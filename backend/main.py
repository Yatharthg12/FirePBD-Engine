"""
FirePBD Engine — FastAPI Application
======================================
Full REST + WebSocket API backend for the FirePBD Engine.

API Surface:
  POST /api/blueprint/upload           Upload PNG or SVG blueprint
  GET  /api/blueprint/{id}/model       Get extracted building model JSON
  POST /api/simulation/run             Start full simulation (async)
  GET  /api/simulation/{id}/status     Poll simulation status
  WS   /api/simulation/{id}/stream     WebSocket: stream real-time snapshots
  GET  /api/simulation/{id}/results    Get full simulation results JSON
  GET  /api/simulation/{id}/report     Download PDF report
  POST /api/simulation/{id}/optimize   Run optimization engine
  GET  /api/health                     Health check

Storage: in-memory dicts (simulation_id → data).
For production, replace with Redis/database.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import (
    API_CORS_ORIGINS,
    API_HOST,
    API_PORT,
    API_RELOAD,
    DEFAULT_GRID_CELL_SIZE_M,
    DEFAULT_MONTE_CARLO_RUNS,
    DEFAULT_SIMULATION_STEPS,
    ENABLE_MONTE_CARLO,
    ENABLE_OPTIMIZATION,
    ENABLE_PDF_REPORT,
    FRONTEND_DIR,
    INPUT_BLUEPRINTS_DIR,
)
from backend.core.constants import MC_DEFAULT_RUNS, SIMULATION_TIMESTEP_S, WS_STREAM_INTERVAL_STEPS
from backend.core.simulation_state import SimulationState
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ─── App Initialisation ───────────────────────────────────────────────────────

app = FastAPI(
    title="FirePBD Engine API",
    description=(
        "Performance-Based Design Fire Safety Simulation System. "
        "AI-assisted blueprint analysis, fire simulation, evacuation modelling, "
        "RSET/ASET calculation, Monte Carlo risk assessment, and PDF reporting."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/app", status_code=307)

# ─── In-Memory Storage ────────────────────────────────────────────────────────
# In production: replace with Redis + database
_blueprints: Dict[str, dict] = {}     # building_id → {model, model_dict, path}
_simulations: Dict[str, dict] = {}    # sim_id → {state, results, report_path}


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    building_id: str
    ignition_zone: Optional[str] = None   # None = auto-select
    n_steps: int = DEFAULT_SIMULATION_STEPS
    dt_s: float = SIMULATION_TIMESTEP_S
    run_monte_carlo: bool = False
    mc_runs: int = MC_DEFAULT_RUNS
    run_optimization: bool = False
    generate_report: bool = False
    cell_size_m: float = DEFAULT_GRID_CELL_SIZE_M


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "1.0.0",
        "blueprints_loaded": len(_blueprints),
        "simulations_run": len(_simulations),
    }


# ─── Blueprint Upload ─────────────────────────────────────────────────────────

@app.post("/api/blueprint/upload")
async def upload_blueprint(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
) -> JSONResponse:
    """
    Upload a building blueprint (PNG, JPG, or SVG).
    Automatically extracts zones, openings, and wall geometry.
    Returns building_id and extraction summary.
    """
    ext = Path(file.filename).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".svg", ".bmp"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    save_path = INPUT_BLUEPRINTS_DIR / f"{file_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Blueprint uploaded: {file.filename} → {save_path}")

    # Extract geometry in background (or synchronously for small files)
    try:
        from backend.agents.blueprint_agent import BlueprintAgent
        agent = BlueprintAgent()
        model = agent.process(str(save_path))
        building_id = model.building_id

        _blueprints[building_id] = {
            "model": model,
            "model_dict": {
                **model.to_dict(),
                "source_url": f"/api/blueprint/{building_id}/source",
            },
            "path": str(save_path),
            "filename": file.filename,
        }

        return JSONResponse({
            "status": "success",
            "building_id": building_id,
            "filename": file.filename,
            "zones": len(model.zones),
            "openings": len(model.openings),
            "exits": len(model.exit_zones),
            "total_area_m2": round(model.total_area_m2, 1),
            "total_occupants": model.total_occupants,
        })

    except Exception as e:
        logger.error(f"Blueprint extraction failed: {e}")
        raise HTTPException(500, f"Blueprint processing failed: {str(e)}")


@app.get("/api/blueprint/{building_id}/model")
async def get_building_model(building_id: str) -> JSONResponse:
    if building_id not in _blueprints:
        raise HTTPException(404, f"Building not found: {building_id}")
    return JSONResponse(_blueprints[building_id]["model_dict"])


@app.get("/api/blueprint/{building_id}/source")
async def get_building_source(building_id: str):
    if building_id not in _blueprints:
        raise HTTPException(404, f"Building not found: {building_id}")
    path = _blueprints[building_id]["path"]
    return FileResponse(path)


@app.get("/api/blueprint/{building_id}/zones")
async def get_building_zones(building_id: str) -> JSONResponse:
    if building_id not in _blueprints:
        raise HTTPException(404, f"Building not found: {building_id}")
    model_dict = _blueprints[building_id]["model_dict"]
    return JSONResponse({
        "building_id": building_id,
        "zones": model_dict.get("zones", []),
        "openings": model_dict.get("openings", []),
        "exit_zone_ids": model_dict.get("exit_zone_ids", []),
    })


# ─── Simulation ───────────────────────────────────────────────────────────────

@app.post("/api/simulation/run")
async def run_simulation(
    req: SimulationRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Start a full simulation run (fire + evacuation + risk + MC + report).
    Returns simulation_id immediately; use /status or /stream for results.
    """
    if req.building_id not in _blueprints:
        raise HTTPException(404, f"Building not found: {req.building_id}")

    sim_id = str(uuid.uuid4())[:12]
    state = SimulationState(
        simulation_id=sim_id,
        building_id=req.building_id,
        ignition_zone=req.ignition_zone or "auto",
        total_steps=req.n_steps,
        dt_s=req.dt_s,
    )
    _simulations[sim_id] = {
        "state": state,
        "status": "queued",
        "results": None,
        "report_path": None,
        "snapshots": [],
    }

    background_tasks.add_task(
        _run_simulation_task,
        sim_id,
        req,
    )

    return JSONResponse({
        "simulation_id": sim_id,
        "status": "queued",
        "message": "Simulation started. Connect to WebSocket for live updates.",
    })


@app.get("/api/simulation/{sim_id}/status")
async def get_simulation_status(sim_id: str) -> JSONResponse:
    if sim_id not in _simulations:
        raise HTTPException(404, f"Simulation not found: {sim_id}")
    sim = _simulations[sim_id]
    state: SimulationState = sim["state"]
    return JSONResponse({
        "simulation_id": sim_id,
        "status": sim["status"],
        "progress_pct": round(
            100 * state.current_step / max(state.total_steps, 1), 1
        ),
        "current_step": state.current_step,
        "total_steps": state.total_steps,
        "sim_time_s": round(state.sim_time_s, 0),
        "aset_s": state.aset_s,
        "rset_s": state.rset_s,
    })


@app.get("/api/simulation/{sim_id}/results")
async def get_simulation_results(sim_id: str) -> JSONResponse:
    if sim_id not in _simulations:
        raise HTTPException(404, f"Simulation not found: {sim_id}")
    sim = _simulations[sim_id]
    if sim["status"] not in ("complete", "error"):
        raise HTTPException(400, f"Simulation not yet complete (status: {sim['status']})")
    return JSONResponse(sim["results"] or {"error": "No results"})


@app.get("/api/simulation/{sim_id}/report")
async def download_report(sim_id: str) -> FileResponse:
    if sim_id not in _simulations:
        raise HTTPException(404)
    report_path = _simulations[sim_id].get("report_path")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(404, "Report not yet generated")
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=Path(report_path).name,
    )


@app.post("/api/simulation/{sim_id}/optimize")
async def run_optimization(sim_id: str) -> JSONResponse:
    """Run optimization agent on existing simulation results."""
    if sim_id not in _simulations:
        raise HTTPException(404)
    sim = _simulations[sim_id]
    if sim["status"] != "complete":
        raise HTTPException(400, "Simulation must be complete before optimization")

    results = sim.get("results", {})
    model = _blueprints.get(sim["state"].building_id, {}).get("model")
    if not model:
        raise HTTPException(404, "Building model not found")

    try:
        from backend.agents.optimization_agent import EvacuationOptimizer
        stored = sim.get("_internal", {})
        optimizer = EvacuationOptimizer(
            model=model,
            graph=stored.get("graph"),
            risk_report=results.get("risk_report", {}),
            evac_summary=results.get("evacuation", {}),
        )
        recs = optimizer.generate()
        return JSONResponse({
            "simulation_id": sim_id,
            "recommendations": [r.to_dict() for r in recs],
            "count": len(recs),
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── WebSocket Stream ─────────────────────────────────────────────────────────

@app.websocket("/api/simulation/{sim_id}/stream")
async def simulation_stream(websocket: WebSocket, sim_id: str) -> None:
    """
    Stream real-time simulation snapshots to the frontend.
    Sends JSON frames every WS_STREAM_INTERVAL_STEPS simulation steps.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for simulation {sim_id}")

    try:
        last_sent = -1
        while True:
            if sim_id not in _simulations:
                await websocket.send_json({"error": "Simulation not found"})
                break

            sim = _simulations[sim_id]
            snapshots = sim.get("snapshots", [])
            state = sim["state"]

            if len(snapshots) > last_sent + 1:
                for snap in snapshots[last_sent + 1:]:
                    await websocket.send_text(json.dumps(snap, default=str))
                    last_sent = len(snapshots) - 1

            if sim["status"] in ("complete", "error"):
                await websocket.send_json({
                    "type": "complete",
                    "status": sim["status"],
                    "summary": state.summary_dict(),
                })
                break

            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {sim_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ─── Simulation Background Task ───────────────────────────────────────────────

async def _run_simulation_task(sim_id: str, req: SimulationRequest) -> None:
    """
    Full simulation pipeline running as a FastAPI background task.
    Runs: Topology → Fire → Evacuation → Risk → MC → Optimization → Report
    """
    sim = _simulations[sim_id]
    state: SimulationState = sim["state"]
    state.start()
    sim["status"] = "running"

    try:
        from backend.agents.blueprint_agent import BlueprintAgent
        from backend.agents.topology_agent import TopologyAgent
        from backend.agents.fire_agent import FireSimulator, FireAnalyzer
        from backend.agents.evacuation_agent import EvacuationSimulator
        from backend.agents.risk_agent import RiskAnalyzer, MonteCarloEngine
        from backend.agents.optimization_agent import EvacuationOptimizer
        from backend.agents.report_agent import ReportGenerator
        from backend.core.simulation_state import StepSnapshot

        model = _blueprints[req.building_id]["model"]

        # ── Build topology ────────────────────────────────────────────────────
        topo = TopologyAgent(cell_size_m=req.cell_size_m)
        graph, grid = topo.build(model, auto_repair=True)

        # ── Fire setup ────────────────────────────────────────────────────────
        fire_sim = FireSimulator(grid, dt_s=req.dt_s)
        fire_analyzer = FireAnalyzer(grid, model.zones, dt_s=req.dt_s)

        # Choose ignition zone
        ignition_zone = req.ignition_zone
        if ignition_zone == "auto" or not ignition_zone:
            non_exits = [z for z in model.zones if not model.zones[z].is_exit]
            exit_ids = list(model.exit_zone_ids)
            best_zone = None
            best_cost = -1.0
            for zone_id in non_exits or list(model.zones.keys()):
                path = graph.shortest_path_to_any_exit(zone_id, exit_ids)
                if not path:
                    continue
                cost = graph._path_cost(path)
                if cost > best_cost:
                    best_cost = cost
                    best_zone = zone_id
            ignition_zone = best_zone or (non_exits[0] if non_exits else list(model.zones.keys())[0])

        state.ignition_zone = ignition_zone
        ignited_count = fire_sim.ignite_zone(ignition_zone)
        state.log_event(f"Fire ignited in zone '{ignition_zone}' ({ignited_count} cells)")

        # ── Evacuation setup ──────────────────────────────────────────────────
        evac_sim = EvacuationSimulator(graph, model, dt_s=req.dt_s)
        evac_sim.populate_from_model()
        state.log_event(
            f"Evacuation: {len(evac_sim.people)} agents, "
            f"{len(model.exit_zone_ids)} exits"
        )

        # ── Main simulation loop ──────────────────────────────────────────────
        zone_status_history = []
        aset_triggered = False

        for step in range(req.n_steps):
            # Fire step
            fire_metrics = fire_sim.step()
            zone_status = fire_analyzer.analyze_zones()
            zone_status_history.append(zone_status)

            # ASET check
            if not aset_triggered and fire_analyzer.compute_aset(zone_status):
                state.log_aset()
                aset_triggered = True

            # Flashover check
            if fire_metrics.get("flashover_detected") and state.flashover_step is None:
                state.log_flashover(ignition_zone)

            # Evacuation step
            evac_metrics = evac_sim.step(zone_status)

            state.advance()

            # RSET check
            rset = evac_sim.compute_rset()
            if rset and state.rset_s is None and evac_metrics.get("all_resolved"):
                state.log_rset()

            # Build snapshot for WebSocket
            if step % WS_STREAM_INTERVAL_STEPS == 0:
                agents = []
                for person in evac_sim.people:
                    zone = model.zones.get(person.current_zone)
                    if zone is None:
                        continue
                    agents.append({
                        "id": person.id,
                        "x": round(zone.centroid.x, 2),
                        "y": round(zone.centroid.y, 2),
                        "status": person.status.value,
                        "zone": person.current_zone,
                    })

                snap = {
                    "type": "step",
                    "step": step,
                    "sim_time_s": state.sim_time_s,
                    "fire": grid.snapshot_compact(),
                    "fire_metrics": fire_metrics,
                    "evac_metrics": evac_metrics,
                    "persons": agents,
                    "grid_size": {
                        "rows": grid.rows,
                        "cols": grid.cols,
                        "cell_m": grid.cell_size_m,
                    },
                    "zone_status": {
                        zid: {
                            "danger": s["danger"],
                            "avg_temp_c": s["avg_temp_c"],
                            "min_visibility_m": s["min_visibility_m"],
                        }
                        for zid, s in zone_status.items()
                    },
                }
                sim["snapshots"].append(snap)

            # Allow other async tasks
            await asyncio.sleep(0)

            if evac_metrics.get("all_resolved"):
                state.log_event(f"All agents resolved at step {step}")
                break

        # ── Risk analysis ─────────────────────────────────────────────────────
        evac_summary = evac_sim.summary()
        risk_report = RiskAnalyzer.generate_report(
            sim_state=state,
            evac_summary=evac_summary,
            zone_status_history=zone_status_history,
            graph=graph,
            model=model,
        )

        # ── Monte Carlo ───────────────────────────────────────────────────────
        mc_result = None
        if req.run_monte_carlo:
            mc_engine = MonteCarloEngine(n_runs=min(req.mc_runs, 200), n_workers=1)
            mc_result = mc_engine.run(model, max_steps=min(req.n_steps, 60), dt_s=req.dt_s)
            risk_report["monte_carlo"] = mc_result.to_dict()

        # ── Optimization ──────────────────────────────────────────────────────
        recommendations = []
        if req.run_optimization:
            optimizer = EvacuationOptimizer(model, graph, risk_report, evac_summary)
            recommendations = optimizer.generate()
            risk_report["recommendations"] = [r.to_dict() for r in recommendations]

        # ── PDF Report ────────────────────────────────────────────────────────
        report_path = None
        if req.generate_report:
            try:
                gen = ReportGenerator(
                    risk_report=risk_report,
                    evac_summary=evac_summary,
                    model=model,
                    sim_state=state,
                    mc_result=mc_result,
                    recommendations=recommendations,
                    zone_status_history=zone_status_history,
                )
                report_path = gen.generate()
                sim["report_path"] = report_path
                state.log_event(f"PDF report generated: {Path(report_path).name}")
            except Exception as e:
                logger.warning(f"Report generation failed: {e}")

        # ── Finalise ─────────────────────────────────────────────────────────
        state.complete()
        state.results = risk_report

        sim["status"] = "complete"
        sim["results"] = {
            "simulation": state.summary_dict(),
            "evacuation": evac_summary,
            "risk_report": risk_report,
            "grid_size": {"rows": grid.rows, "cols": grid.cols, "cell_m": grid.cell_size_m},
            "recommendations_count": len(recommendations),
        }
        sim["_internal"] = {"graph": graph, "grid": grid}

        logger.info(
            f"Simulation {sim_id} complete — "
            f"RSET={state.rset_s}s, ASET={state.aset_s}s, "
            f"risk={risk_report.get('risk_score')}/100"
        )

    except Exception as e:
        logger.error(f"Simulation {sim_id} failed: {traceback.format_exc()}")
        sim["status"] = "error"
        state.error(str(e))
        sim["results"] = {"error": str(e), "traceback": traceback.format_exc()}


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from backend.config import print_config
    print_config()
    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        log_level="info",
    )
