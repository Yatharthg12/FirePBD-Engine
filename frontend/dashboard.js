/**
 * FirePBD Engine — Main Dashboard Controller
 * Handles: API calls, WebSocket streaming, UI state, Chart.js rendering,
 * file upload, drag-and-drop, tab navigation, toast notifications.
 */

const API = 'http://localhost:8000';
let _buildingId = null;
let _simId = null;
let _ws = null;
let _simRunning = false;
let _snapshots = [];
let _charts = {};
let _results = null;
let _modelData = null;
let _selectedIgnitionZone = 'auto';
let _selectedMarkerPoint = null;
let _routeZoneIds = [];
let _routePoints = [];
let _playbackTimer = null;
let _playbackPaused = false;
let _playbackSpeedIndex = 0;
const _playbackSpeeds = [1, 2, 4];

function _tabPanelId(tab) {
  const map = {
    dashboard: 'tabDashboard',
    simulation: 'tabSimulation',
    analysis: 'tabAnalysis',
    montecarlo: 'tabMonteCarlo',
    optimization: 'tabOptimization',
  };
  return map[tab] || `tab${tab.charAt(0).toUpperCase()}${tab.slice(1)}`;
}

function _currentPlaybackSpeed() {
  return _playbackSpeeds[_playbackSpeedIndex];
}

function _setPlaybackControls() {
  return;
}

function _stopPlaybackTimer() {
  if (_playbackTimer) {
    clearTimeout(_playbackTimer);
    _playbackTimer = null;
  }
}

function _schedulePlayback() {
  if (_playbackPaused || _playbackTimer || !_snapshots.length) return;
  _playbackTimer = setTimeout(() => {
    _playbackTimer = null;
    const next = _snapshots.shift();
    if (next) {
      renderStepSnapshot(next);
    }
    if (_snapshots.length) {
      _schedulePlayback();
    }
  }, 220);
}

function _renderSnapshotByStep(step) {
  if (!_stepHistory.length) return;
  let chosen = _stepHistory[0];
  for (const snap of _stepHistory) {
    if (snap.step <= step) chosen = snap;
  }
  if (chosen) renderStepSnapshot(chosen, true);
}

function _renderSnapshotByPercent(percent) {
  if (!_stepHistory.length) return;
  const maxStep = Math.max(..._stepHistory.map(s => s.step), 1);
  const target = Math.round((percent / 100) * maxStep);
  _renderSnapshotByStep(target);
}

let _stepHistory = [];

const _OPENING_TYPE_FACTOR = {
  emergency_exit: 2.0,
  double_door: 1.8,
  archway: 1.5,
  door: 1.0,
  window: 0.3,
  inferred: 0.8,
  default: 1.0,
};

function _zoneById(zoneId) {
  return _modelData?.zones?.find(z => z.id === zoneId) || null;
}

function _exitZones() {
  return (_modelData?.zones || []).filter(z => z.is_exit);
}

function _zoneForWorldPoint(point) {
  if (!_modelData) return null;
  for (const zone of _modelData.zones || []) {
    if (_pointInPolygon(point, zone.polygon)) {
      return zone;
    }
  }
  let best = null;
  let bestDist = Infinity;
  for (const zone of _modelData.zones || []) {
    const dx = zone.centroid.x - point.x;
    const dy = zone.centroid.y - point.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < bestDist) {
      bestDist = dist;
      best = zone;
    }
  }
  return best;
}

function _pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const intersect = ((yi > point.y) !== (yj > point.y)) &&
      (point.x < (xj - xi) * (point.y - yi) / ((yj - yi) || 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function _routeCost(zoneA, zoneB, opening) {
  const pa = zoneA?.centroid || { x: 0, y: 0 };
  const pb = zoneB?.centroid || { x: 0, y: 0 };
  const dx = pa.x - pb.x;
  const dy = pa.y - pb.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const factor = _OPENING_TYPE_FACTOR[opening?.type || 'default'] || 1.0;
  return dist / Math.max((opening?.width || 1.0) * factor, 0.1);
}

function _buildAdjacency() {
  const adjacency = new Map();
  for (const zone of _modelData?.zones || []) {
    adjacency.set(zone.id, []);
  }
  const zones = new Map((_modelData?.zones || []).map(z => [z.id, z]));
  for (const opening of _modelData?.openings || []) {
    const a = zones.get(opening.zone_a);
    const b = zones.get(opening.zone_b);
    if (!a || !b) continue;
    const cost = _routeCost(a, b, opening);
    adjacency.get(a.id).push({ to: b.id, cost });
    adjacency.get(b.id).push({ to: a.id, cost });
  }
  return adjacency;
}

function _dijkstra(startZone, exitIds) {
  const adjacency = _buildAdjacency();
  const dist = new Map([[startZone, 0]]);
  const prev = new Map();
  const visited = new Set();
  const queue = [[0, startZone]];

  const popMin = () => {
    queue.sort((a, b) => a[0] - b[0]);
    return queue.shift();
  };

  while (queue.length) {
    const current = popMin();
    if (!current) break;
    const [cost, zoneId] = current;
    if (visited.has(zoneId)) continue;
    visited.add(zoneId);
    if (exitIds.includes(zoneId)) break;

    for (const edge of adjacency.get(zoneId) || []) {
      const nextCost = cost + edge.cost;
      if (nextCost < (dist.get(edge.to) ?? Infinity)) {
        dist.set(edge.to, nextCost);
        prev.set(edge.to, zoneId);
        queue.push([nextCost, edge.to]);
      }
    }
  }

  let bestExit = null;
  let bestCost = Infinity;
  for (const exitId of exitIds) {
    const c = dist.get(exitId);
    if (c !== undefined && c < bestCost) {
      bestCost = c;
      bestExit = exitId;
    }
  }
  if (!bestExit) return [];

  const path = [bestExit];
  while (path[0] !== startZone) {
    const p = prev.get(path[0]);
    if (!p) break;
    path.unshift(p);
  }
  return path[0] === startZone ? path : [];
}

function _worstCaseIgnitionZone() {
  const exits = _exitZones().map(z => z.id);
  let worst = null;
  let worstCost = -Infinity;
  for (const zone of _modelData?.zones || []) {
    if (zone.is_exit) continue;
    const route = _dijkstra(zone.id, exits);
    const cost = route.length;
    if (cost > worstCost) {
      worstCost = cost;
      worst = zone.id;
    }
  }
  return worst || (_modelData?.zones || [])[0]?.id || 'auto';
}

function _routePointsFor(zoneIds) {
  return zoneIds
    .map(zid => _zoneById(zid))
    .filter(Boolean)
    .map(zone => ({ x: zone.centroid.x, y: zone.centroid.y }));
}

function _stairPath(points) {
  if (!points.length) return [];
  const out = [{ x: points[0].x, y: points[0].y }];
  for (let i = 1; i < points.length; i++) {
    const a = out[out.length - 1];
    const b = points[i];
    if (a.x === b.x || a.y === b.y) {
      out.push({ x: b.x, y: b.y });
      continue;
    }
    const viaX = (a.x + b.x) / 2;
    const viaY = (a.y + b.y) / 2;
    const bend1 = { x: viaX, y: a.y };
    const bend2 = { x: viaX, y: viaY };
    const bend3 = { x: b.x, y: viaY };
    if (bend1.x !== a.x || bend1.y !== a.y) out.push(bend1);
    if (bend2.x !== out[out.length - 1].x || bend2.y !== out[out.length - 1].y) out.push(bend2);
    if (bend3.x !== out[out.length - 1].x || bend3.y !== out[out.length - 1].y) out.push(bend3);
    out.push({ x: b.x, y: b.y });
  }
  return out;
}

function _openingBetween(zoneA, zoneB) {
  return (_modelData?.openings || []).find(o =>
    (o.zone_a === zoneA && o.zone_b === zoneB) ||
    (o.zone_a === zoneB && o.zone_b === zoneA)
  ) || null;
}

function _routeWaypointsFor(zoneIds, anchorPoint = null) {
  if (!zoneIds.length) return anchorPoint ? [anchorPoint] : [];
  const pts = [];
  if (anchorPoint) pts.push({ x: anchorPoint.x, y: anchorPoint.y });
  zoneIds.forEach((zoneId, idx) => {
    const next = zoneIds[idx + 1];
    if (next) {
      const opening = _openingBetween(zoneId, next);
      if (opening?.midpoint) {
        pts.push({ x: opening.midpoint.x, y: opening.midpoint.y });
      } else {
        const a = _zoneById(zoneId)?.centroid;
        const b = _zoneById(next)?.centroid;
        if (a && b) {
          pts.push({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
        }
      }
    } else {
      const zone = _zoneById(zoneId);
      if (zone) {
        const centroid = { x: zone.centroid.x, y: zone.centroid.y };
        if (!pts.length || pts[pts.length - 1].x !== centroid.x || pts[pts.length - 1].y !== centroid.y) {
          pts.push(centroid);
        }
      }
    }
  });
  return pts;
}

function _syncMapMarker(point) {
  _selectedMarkerPoint = point ? { x: point.x, y: point.y } : null;
  if (window.blueprintViewer) window.blueprintViewer.setMarker(_selectedMarkerPoint);
}

function _updateScenarioCard(zoneId, routeZoneIds) {
  const card = document.getElementById('scenarioCard');
  if (!card) return;
  const route = routeZoneIds.length ? routeZoneIds.join(' → ') : 'No path';
  card.innerHTML = `
    <div class="scenario-title">Scenario</div>
    <div class="scenario-body">
      Ignition: <b>${zoneId}</b><br/>
      Exit route: <b>${route}</b><br/>
      <span class="scenario-note">Auto mode uses the worst-case non-exit zone.</span>
    </div>
  `;
}

function _applyRoutePreview(zoneId, worldPoint = null) {
  if (!_modelData) return;
  const actualZoneId = zoneId === 'auto'
    ? (_zoneForWorldPoint(worldPoint || _selectedMarkerPoint || { x: 0, y: 0 })?.id || _worstCaseIgnitionZone())
    : zoneId;
  const exitIds = _exitZones().map(z => z.id);
  const route = actualZoneId ? _dijkstra(actualZoneId, exitIds) : [];
  _selectedIgnitionZone = actualZoneId;
  _routeZoneIds = route;
  if (worldPoint) {
    _selectedMarkerPoint = { x: worldPoint.x, y: worldPoint.y };
  } else if (!_selectedMarkerPoint) {
    const z = _zoneById(actualZoneId);
    _selectedMarkerPoint = z ? { x: z.centroid.x, y: z.centroid.y } : null;
  }
  _routePoints = _stairPath(_routeWaypointsFor(route, _selectedMarkerPoint));

  const select = document.getElementById('ignitionZoneSelect');
  if (select && select.value !== zoneId) select.value = zoneId;

  if (window.blueprintViewer) {
    window.blueprintViewer.setSelection(actualZoneId, _routePoints);
    if (_selectedMarkerPoint) {
      window.blueprintViewer.setMarker(_selectedMarkerPoint);
    }
  }
  if (window.simRenderer) {
    window.simRenderer.setRoute(_routePoints, parseInt(document.getElementById('simSteps').value) || 90);
  }
  _updateScenarioCard(actualZoneId, route);
}

// ─── Utility ──────────────────────────────────────────────────────────────────

function toast(msg, type = 'success') {
  const c = document.getElementById('toastContainer');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  const icons = { success: '✓', error: '✗', warn: '⚠' };
  t.innerHTML = `<span>${icons[type] || '•'}</span><span>${msg}</span>`;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4500);
}

function log(msg, cls = '') {
  const el = document.getElementById('eventLog');
  const empty = el.querySelector('.log-empty');
  if (empty) empty.remove();
  const item = document.createElement('div');
  item.className = `log-item ${cls}`;
  item.textContent = msg;
  el.appendChild(item);
  el.scrollTop = el.scrollHeight;
}

function simLog(msg) {
  const el = document.getElementById('simLog');
  el.textContent += msg + '\n';
  el.scrollTop = el.scrollHeight;
}

function setLoadingProgress(pct, msg = '') {
  document.getElementById('loadingFill').style.width = `${pct}%`;
  if (msg) document.getElementById('loadingMsg').textContent = msg;
}

// ─── API Health Check ─────────────────────────────────────────────────────────

async function checkApiHealth() {
  const dot = document.getElementById('apiStatusDot');
  const label = document.getElementById('apiStatus');
  try {
    const res = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      dot.className = 'status-dot online';
      label.textContent = 'API Online';
    } else {
      throw new Error('Not OK');
    }
  } catch {
    dot.className = 'status-dot error';
    label.textContent = 'API Offline';
  }
}

// ─── Tab Navigation ───────────────────────────────────────────────────────────

document.querySelectorAll('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById(_tabPanelId(tab));
    if (panel) panel.classList.add('active');
  });
});

// ─── File Upload & Drag-Drop ──────────────────────────────────────────────────

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) uploadFile(e.target.files[0]);
});
dropZone.addEventListener('click', e => {
  if (e.target.tagName !== 'LABEL') fileInput.click();
});

async function uploadFile(file) {
  const status = document.getElementById('uploadStatus');
  const progress = document.getElementById('uploadProgress');
  const msg = document.getElementById('uploadMsg');

  status.style.display = 'block';
  progress.style.width = '30%';
  msg.textContent = `Uploading ${file.name}…`;
  simLog(`[UPLOAD] ${file.name} (${(file.size/1024).toFixed(0)} KB)`);

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`${API}/api/blueprint/upload`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    progress.style.width = '100%';
    msg.textContent = `Extraction complete — ${data.zones} zones, ${data.exits} exits`;

    _buildingId = data.building_id;
    simLog(`[EXTRACT] Building ID: ${_buildingId}`);
    simLog(`[EXTRACT] Zones: ${data.zones}, Openings: ${data.openings}, Exits: ${data.exits}`);
    simLog(`[EXTRACT] Area: ${data.total_area_m2} m², Occupants: ${data.total_occupants}`);

    showBuildingSummary(data);
    await loadAndViewBuildingModel(data.building_id);
    document.getElementById('simControls').style.display = 'flex';
    document.getElementById('runSimBtn').disabled = false;
    toast(`Blueprint loaded: ${data.zones} zones, ${data.exits} exits`);

  } catch (err) {
    progress.style.width = '100%';
    progress.style.background = 'var(--danger)';
    msg.textContent = `Error: ${err.message}`;
    toast(`Upload failed: ${err.message}`, 'error');
    simLog(`[ERROR] ${err.message}`);
  }

  setTimeout(() => { status.style.display = 'none'; progress.style.width = '0%'; progress.style.background = ''; }, 3000);
}

function showBuildingSummary(data) {
  document.getElementById('buildingSummary').style.display = 'block';
  document.getElementById('sumZones').textContent = data.zones;
  document.getElementById('sumOpenings').textContent = data.openings;
  document.getElementById('sumExits').textContent = data.exits;
  document.getElementById('sumArea').textContent = data.total_area_m2;
  document.getElementById('sumBuildingId').textContent = data.building_id;
}

async function loadAndViewBuildingModel(buildingId) {
  try {
    const res = await fetch(`${API}/api/blueprint/${buildingId}/model`);
    if (!res.ok) return;
    const model = await res.json();
    _modelData = model;
    document.getElementById('viewerContainer').style.display = 'block';
    const bpCanvas = document.getElementById('blueprintCanvas');
    bpCanvas.width = bpCanvas.parentElement.clientWidth;
    bpCanvas.height = 180;
    if (!window.blueprintViewer) initBlueprintViewer();
    window.blueprintViewer.setModel(model);
    if (model.source_url && window.blueprintViewer.setSourceImage) {
      window.blueprintViewer.setSourceImage(`${API}${model.source_url}`);
    }
    window.blueprintViewer.onZoneSelected = (zoneId) => {
      _applyRoutePreview(zoneId);
    };
    window.blueprintViewer.onPointSelected = (point, zone) => {
      _applyRoutePreview(zone?.id || 'auto', point);
    };
    if (window.simRenderer?.setModel) window.simRenderer.setModel(model);
    if (model.source_url && window.simRenderer.setSourceImage) {
      window.simRenderer.setSourceImage(`${API}${model.source_url}`);
    }
    simLog(`[MODEL] Graph: ${model.zones.length} zones, ${model.openings.length} openings`);
    populateIgnitionZones(model.zones);
    _applyRoutePreview('auto');
  } catch (e) {
    console.warn('Could not load building model:', e);
  }
}

function populateIgnitionZones(zones) {
  const sel = document.getElementById('ignitionZoneSelect');
  sel.innerHTML = '<option value="auto">Auto (worst-case non-exit)</option>';
  (zones || [])
    .filter(z => !z.is_exit)
    .forEach((zone) => {
      const opt = document.createElement('option');
      opt.value = zone.id;
      opt.textContent = `${zone.id} · ${Math.round(zone.area)} m²`;
      sel.appendChild(opt);
    });
  sel.value = 'auto';
}

// ─── Run Simulation ───────────────────────────────────────────────────────────

document.getElementById('runSimBtn').addEventListener('click', runSimulation);

async function runSimulation() {
  if (!_buildingId) { toast('Upload a blueprint first', 'warn'); return; }
  if (_simRunning) { toast('Simulation already running', 'warn'); return; }

  _simRunning = true;
  _snapshots = [];
  _stepHistory = [];
  _playbackPaused = false;
  _playbackSpeedIndex = 0;
  _stopPlaybackTimer();
  _setPlaybackControls();
  document.getElementById('loadingOverlay').style.display = 'flex';
  setLoadingProgress(5, 'Initialising fire physics…');
  document.getElementById('runSimBtn').disabled = true;
  document.getElementById('simPlaceholder').style.display = 'none';

  const body = {
    building_id: _buildingId,
    ignition_zone: document.getElementById('ignitionZoneSelect').value || 'auto',
    n_steps: parseInt(document.getElementById('simSteps').value) || 90,
    dt_s: 5.0,
    run_monte_carlo: document.getElementById('enableMC').checked,
    mc_runs: parseInt(document.getElementById('cfgMcRuns')?.value || 200),
    run_optimization: true,
    generate_report: document.getElementById('enableReport').checked,
    cell_size_m: 1.0,
  };

  simLog(`\n[SIM] Starting simulation — ${body.n_steps} steps, ignition: ${body.ignition_zone}`);

  try {
    const res = await fetch(`${API}/api/simulation/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    _simId = data.simulation_id;
    simLog(`[SIM] ID: ${_simId}`);
    document.getElementById('totalSteps').textContent = body.n_steps;

    setLoadingProgress(10, 'Building spatial graph…');
    connectWebSocket(_simId);

  } catch (err) {
    toast(`Simulation failed: ${err.message}`, 'error');
    simLog(`[ERROR] ${err.message}`);
    _simRunning = false;
    document.getElementById('runSimBtn').disabled = false;
    document.getElementById('loadingOverlay').style.display = 'none';
  }
}

// ─── WebSocket Streaming ──────────────────────────────────────────────────────

function connectWebSocket(simId) {
  const wsUrl = `ws://localhost:8000/api/simulation/${simId}/stream`;
  _ws = new WebSocket(wsUrl);

  _ws.onopen = () => {
    simLog('[WS] Connected — receiving live updates');
    setLoadingProgress(20, 'Fire simulation running…');
  };

  _ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'step') {
      handleStepUpdate(msg);
    } else if (msg.type === 'complete') {
      handleSimulationComplete(msg);
    } else if (msg.error) {
      toast(`Simulation error: ${msg.error}`, 'error');
      simLog(`[ERROR] ${msg.error}`);
      _simRunning = false;
    }
  };

  _ws.onerror = (e) => {
    // WS failed — fall back to polling
    simLog('[WS] WebSocket unavailable — polling status');
    pollSimulationStatus(simId);
  };

  _ws.onclose = () => {
    simLog('[WS] Connection closed');
  };
}

function renderStepSnapshot(msg, fromHistory = false) {
  const step = msg.step;
  const total = parseInt(document.getElementById('totalSteps').textContent) || 1;
  const pct = Math.min(Math.round((step / total) * 100), 100);

  setLoadingProgress(pct, `Step ${step}/${total} — t=${msg.sim_time_s?.toFixed(0)}s`);
  document.getElementById('currentStep').textContent = step;
  document.getElementById('simTimestamp').textContent = `${msg.sim_time_s?.toFixed(0) || 0}s`;
  if (msg.fire && window.simRenderer) {
    if (!window.simRenderer.gridRows && msg.grid_size) {
      window.simRenderer.setGridDimensions(msg.grid_size.rows, msg.grid_size.cols, msg.grid_size.cell_m);
    }
    const r = window.simRenderer;
    if (r.gridRows === 0 && msg.fire.state) {
      r.gridRows = msg.fire.state.length;
      r.gridCols = msg.fire.state[0]?.length || 0;
    }
    if (msg.zone_status && window.simRenderer.setZoneStatus) {
      window.simRenderer.setZoneStatus(msg.zone_status);
    }
    window.simRenderer.render(msg.fire, msg.persons || []);
  }

  const fm = msg.fire_metrics || {};
  document.getElementById('statBurning').textContent = fm.burning_cells ?? 0;
  document.getElementById('statMaxTemp').textContent = `${(fm.max_temp_c ?? 20).toFixed(0)}°C`;
  if (fm.flashover_detected && document.getElementById('statFlashover').textContent === '—') {
    document.getElementById('statFlashover').textContent = `t=${msg.sim_time_s?.toFixed(0)}s`;
    document.getElementById('statFlashover').style.color = 'var(--danger)';
    log(`⚠ FLASHOVER at t=${msg.sim_time_s?.toFixed(0)}s`, 'danger');
  }

  const em = msg.evac_metrics || {};
  const total_agents = em.total_agents || 1;
  const evac_pct = ((em.evacuated ?? 0) / total_agents * 100).toFixed(1);
  document.getElementById('evacBar').style.width = `${evac_pct}%`;
  document.getElementById('statEvacuated').textContent = `${em.evacuated ?? 0} evacuated`;
  document.getElementById('statWaiting').textContent = `${em.waiting ?? 0} waiting`;
  document.getElementById('statDead').textContent = `${em.dead ?? 0} dead`;

  if (msg.zone_status && window.blueprintViewer) {
    window.blueprintViewer.updateDanger(msg.zone_status);
  }

  if (!fromHistory && step % 10 === 0) {
    simLog(`[${msg.sim_time_s?.toFixed(0)}s] Burning: ${fm.burning_cells}, Evac: ${em.evacuated}/${em.total_agents}`);
  }
}

function handleStepUpdate(msg) {
  _stepHistory.push(msg);
  _snapshots.push(msg);
  _schedulePlayback();
}

async function handleSimulationComplete(msg) {
  setLoadingProgress(95, 'Collecting results…');
  simLog('[SIM] Simulation complete — fetching results');

  // Fetch full results
  try {
    const res = await fetch(`${API}/api/simulation/${_simId}/results`);
    if (res.ok) {
      _results = await res.json();
      updateResultsUI(_results);
    }
  } catch(e) { simLog('[WARN] Could not fetch full results'); }

  setLoadingProgress(100, 'Done');
  setTimeout(() => {
    document.getElementById('loadingOverlay').style.display = 'none';
    document.getElementById('downloadBar').style.display = 'flex';
    document.getElementById('runSimBtn').disabled = false;
    _simRunning = false;
    toast('Simulation complete! Results ready.', 'success');
    log('✓ Simulation complete', 'safe');
    buildAnalysisCharts();
    buildMonteCarloCharts();
    buildOptimizationTab();
  }, 800);
}

async function pollSimulationStatus(simId) {
  const maxWait = 600; // 10 min
  let waited = 0;
  const interval = setInterval(async () => {
    waited += 2;
    if (waited > maxWait) { clearInterval(interval); return; }
    try {
      const res = await fetch(`${API}/api/simulation/${simId}/status`);
      if (!res.ok) return;
      const data = await res.json();
      const pct = Math.min(data.progress_pct + 10, 90);
      setLoadingProgress(pct, `Step ${data.current_step}/${data.total_steps}`);
      document.getElementById('currentStep').textContent = data.current_step;
      document.getElementById('simTimestamp').textContent = `${data.sim_time_s}s`;

      if (data.status === 'complete' || data.status === 'error') {
        clearInterval(interval);
        handleSimulationComplete({ summary: data });
      }
    } catch(e) { clearInterval(interval); }
  }, 2000);
}

// ─── Results UI ───────────────────────────────────────────────────────────────

function updateResultsUI(results) {
  const rr = results.risk_report || {};
  const summ = results.simulation || {};

  // Risk score
  const score = rr.risk_score || 0;
  const level = rr.risk_level || '—';
  document.getElementById('riskScore').textContent = score;
   document.getElementById('riskLevel').textContent = `${level} · ${_selectedIgnitionZone || 'scenario'} ignition`;
  const riskCard = document.getElementById('riskCard');
  const scoreColour = score > 70 ? 'var(--danger)' : score > 40 ? 'var(--warn)' : 'var(--safe)';
  document.getElementById('riskScore').style.color = scoreColour;

  // RSET / ASET
  document.getElementById('rsetValue').textContent = rr.rset_s ?? '—';
  document.getElementById('asetValue').textContent = rr.aset_s ?? 'N/A';
  const margin = rr.rset_aset_margin_s;
  document.getElementById('marginValue').textContent = margin !== null && margin !== undefined ? `${margin}` : '—';
  const margAssess = rr.margin_assessment || '';
  const mEl = document.getElementById('marginStatus');
  mEl.textContent = `${margAssess} (required ≥ 120s)`;
  mEl.style.color = margAssess === 'ADEQUATE' ? 'var(--safe)' : margAssess === 'MARGINAL' ? 'var(--warn)' : 'var(--danger)';

  // Log key events
  const events = rr.events || [];
  events.forEach(e => {
    const cls = e.includes('FLASHOVER') ? 'danger' : e.includes('ASET') ? 'warn' : e.includes('RSET') ? 'safe' : '';
    log(e, cls);
  });
}

// ─── Analysis Charts ──────────────────────────────────────────────────────────

function buildAnalysisCharts() {
  if (!_results) return;
  const summary = document.getElementById('analysisSummary');
  const tableBody = document.getElementById('analysisHistoryTable');
  const history = _stepHistory.length ? [..._stepHistory] : [];
  const last = history[history.length - 1] || {};
  const burnSeries = history.map(s => s.fire_metrics?.burning_cells || 0);
  const evacSeries = history.map(s => s.evac_metrics?.evacuated || 0);
  const tempSeries = history.map(s => s.fire_metrics?.max_temp_c || 0);
  if (summary) {
    const routeText = _routeZoneIds.length ? _routeZoneIds.join(' → ') : 'No route';
    const topoZones = _modelData?.zones?.length || 0;
    const topoOpenings = _modelData?.openings?.length || 0;
    const topoExits = _exitZones().length || 0;
    summary.innerHTML = `
      <div class="analysis-metrics">
        <div class="analysis-card"><div class="analysis-val">${topoZones}</div><div class="analysis-label">Blueprint zones</div></div>
        <div class="analysis-card"><div class="analysis-val">${topoOpenings}</div><div class="analysis-label">Openings</div></div>
        <div class="analysis-card"><div class="analysis-val">${topoExits}</div><div class="analysis-label">Exits</div></div>
        <div class="analysis-card"><div class="analysis-val">${_routeZoneIds.length || 0}</div><div class="analysis-label">Route hops</div></div>
      </div>
      <div class="analysis-note">Blueprint topology and graph route inspection. Current ignition: <b>${_selectedIgnitionZone}</b> · Route: <b>${routeText}</b></div>
    `;
  }

  if (tableBody) {
    const rows = history.slice(-8).reverse();
    tableBody.innerHTML = rows.length ? rows.map(step => {
      const temp = step.fire_metrics?.max_temp_c || 0;
      const status = temp > 550 ? 'Critical' : temp > 250 ? 'Elevated' : 'Stable';
      return `
        <tr>
          <td>${step.step}</td>
          <td>${step.fire_metrics?.burning_cells || 0}</td>
          <td>${step.evac_metrics?.evacuated || 0}</td>
          <td>${temp.toFixed(0)}°C</td>
          <td><span class="table-pill ${status.toLowerCase()}">${status}</span></td>
        </tr>
      `;
    }).join('') : '<tr><td colspan="5" class="table-placeholder">No step history yet.</td></tr>';
  }

  const rset = _results.risk_report?.rset_s || 0;
  const aset = _results.risk_report?.aset_s || 0;
  const maxT = _results.simulation?.sim_time_s || 600;

  if (_charts.rsetAset) _charts.rsetAset.destroy();
  _charts.rsetAset = new Chart(document.getElementById('chartRsetAset'), {
    type: 'bar',
    data: {
      labels: ['RSET', 'ASET'],
      datasets: [{
        label: 'Time (s)',
        data: [rset, aset || maxT],
        backgroundColor: ['rgba(249,115,22,0.7)', 'rgba(34,197,94,0.7)'],
        borderColor: ['#f97316', '#22c55e'],
        borderWidth: 2,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.raw}s` } },
        annotation: {
          annotations: {
            required: {
              type: 'line',
              y: rset + 120,
              borderColor: '#f59e0b',
              borderWidth: 2,
              label: { display: true, content: 'Min ASET (RSET+120s)', color: '#f59e0b', position: 'end' },
            }
          }
        }
      },
      scales: {
        y: { beginAtZero: true, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
      }
    }
  });

  if (_charts.analysisTrend) _charts.analysisTrend.destroy();
  _charts.analysisTrend = new Chart(document.getElementById('chartAnalysisTrend'), {
    type: 'line',
    data: {
      labels: history.map(s => s.step),
      datasets: [
        {
          label: 'Burning cells',
          data: burnSeries,
          borderColor: '#f97316',
          backgroundColor: 'rgba(249,115,22,0.18)',
          tension: 0.3,
        },
        {
          label: 'Evacuated',
          data: evacSeries,
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34,197,94,0.18)',
          tension: 0.3,
        },
        {
          label: 'Peak temp (°C)',
          data: tempSeries,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56,189,248,0.18)',
          tension: 0.3,
          yAxisID: 'y1',
        },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#cbd5e1' } } },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y1: {
          position: 'right',
          ticks: { color: '#94a3b8' },
          grid: { drawOnChartArea: false },
        }
      }
    }
  });
}

function buildMonteCarloCharts() {
  const mcSummary = document.getElementById('mcSummary');
  const passCanvas = document.getElementById('chartMcPass');
  if (!_results?.risk_report?.monte_carlo) {
    if (_charts.mcRset) {
      _charts.mcRset.destroy();
      _charts.mcRset = null;
    }
    if (_charts.mcPass) {
      _charts.mcPass.destroy();
      _charts.mcPass = null;
    }
    mcSummary.innerHTML = `
      <div class="mc-placeholder">
        Monte Carlo was disabled for this run. Enable it before starting a simulation to see probabilistic RSET and pass-rate results.
      </div>
    `;
    return;
  }
  const mc = _results.risk_report.monte_carlo;
  const rsetCi = mc.rset?.ci_90pct || [0, 0];

  mcSummary.innerHTML = `
    <div class="mc-explain">Monte Carlo mode summarizes variability across repeated runs of the same scenario.</div>
    <div class="mc-grid">
      <div class="mc-metric"><div class="mc-metric-val">${mc.rset?.mean_s || '—'}s</div><div class="mc-metric-label">RSET Mean</div></div>
      <div class="mc-metric"><div class="mc-metric-val">${mc.rset?.p90_s || '—'}s</div><div class="mc-metric-label">RSET P90</div></div>
      <div class="mc-metric"><div class="mc-metric-val">${mc.evacuation_success?.mean_pct || '—'}%</div><div class="mc-metric-label">Evac Success</div></div>
      <div class="mc-metric"><div class="mc-metric-val">${(mc.pass_rate_pct || 0).toFixed(1)}%</div><div class="mc-metric-label">BS 9999 Pass Rate</div></div>
      <div class="mc-metric"><div class="mc-metric-val">${mc.aset?.mean_s || '—'}s</div><div class="mc-metric-label">ASET Mean</div></div>
      <div class="mc-metric"><div class="mc-metric-val">${mc.mean_dead || 0}</div><div class="mc-metric-label">Mean dead</div></div>
      <div class="mc-metric"><div class="mc-metric-val">${mc.mean_incapacitated || 0}</div><div class="mc-metric-label">Mean incapacitated</div></div>
      <div class="mc-metric"><div class="mc-metric-val">${rsetCi[0]}-${rsetCi[1]}</div><div class="mc-metric-label">90% RSET CI</div></div>
    </div>
    <div class="analysis-note">Uncertainty window: RSET ${rsetCi[0]}s to ${rsetCi[1]}s, ASET mean ${mc.aset?.mean_s || '—'}s, evacuation success P10 ${mc.evacuation_success?.p10_pct || '—'}%.</div>
    `;

  if (passCanvas) {
    if (_charts.mcPass) _charts.mcPass.destroy();
    _charts.mcPass = new Chart(passCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Pass', 'Fail'],
        datasets: [{
          data: [mc.pass_rate_pct || 0, 100 - (mc.pass_rate_pct || 0)],
          backgroundColor: ['rgba(34,197,94,0.8)', 'rgba(239,68,68,0.8)'],
          borderColor: ['#22c55e', '#ef4444'],
          borderWidth: 2,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: '#cbd5e1' } },
        }
      }
    });
  }

  // RSET summary chart from the actual Monte Carlo report.
  const labels = ['Mean RSET', 'P90 RSET', '120s Target'];
  const mean = mc.rset?.mean_s || 0;
  const p90 = mc.rset?.p90_s || 0;

  if (_charts.mcRset) _charts.mcRset.destroy();
  _charts.mcRset = new Chart(document.getElementById('chartMcRset'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Seconds',
        data: [mean, p90, 120],
        backgroundColor: ['rgba(56,189,248,0.55)', 'rgba(245,158,11,0.55)', 'rgba(34,197,94,0.55)'],
        borderColor: ['#38bdf8', '#f59e0b', '#22c55e'],
        borderWidth: 1, borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${ctx.raw.toFixed ? ctx.raw.toFixed(1) : ctx.raw}s` } } },
      scales: {
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
      }
    }
  });
}

// ─── Optimization Tab ─────────────────────────────────────────────────────────

function buildOptimizationTab() {
  const recs = _results?.risk_report?.recommendations || [];
  const list = document.getElementById('recList');
  const header = document.getElementById('optSummary');
  const impact = document.getElementById('optImpact');
  if (header) {
    header.innerHTML = `
      <div class="opt-explain">Optimization recommends the highest-impact changes first so the layout can improve before the next run.</div>
      <div class="opt-meta">Scenario: <b>${_selectedIgnitionZone}</b> · Route length: <b>${_routeZoneIds.length || 0}</b> hops</div>
    `;
  }

  if (impact) {
    const top = recs.slice(0, 3);
    impact.innerHTML = top.length ? top.map(rec => `
      <div class="impact-card">
        <div class="impact-title">${rec.title}</div>
        <div class="impact-bar"><span style="width:${Math.min(100, Math.max(10, rec.estimated_rset_reduction_s || 0))}%"></span></div>
        <div class="impact-meta">${rec.category} · −${rec.estimated_rset_reduction_s?.toFixed(0) || 0}s</div>
      </div>
    `).join('') : '<div class="rec-placeholder">No optimization actions available for this scenario.</div>';
  }

  if (!recs.length) {
    list.innerHTML = `
      <div class="rec-placeholder">
        No recommendations were generated for this run. If you want the optimizer to propose layout changes, enable optimization before starting the simulation.
      </div>
    `;
    return;
  }

  list.innerHTML = recs.map(rec => `
    <div class="rec-card">
      <div>
        <div class="rec-priority ${rec.priority}">${rec.priority}</div>
      </div>
      <div class="rec-content">
        <div class="rec-title">${rec.title}</div>
        <div class="rec-desc">${rec.description?.substring(0, 300)}${(rec.description?.length > 300) ? '…' : ''}</div>
        <div class="rec-meta">
          <span class="rec-reduction">−${rec.estimated_rset_reduction_s?.toFixed(0)}s RSET</span>
          <span>${rec.category}</span>
          <span>${rec.standard_reference || ''}</span>
        </div>
      </div>
    </div>
  `).join('');
}

// ─── Layer Toggles ────────────────────────────────────────────────────────────

document.querySelectorAll('.layer-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const layer = btn.dataset.layer;
    btn.classList.toggle('active');
    if (window.simRenderer) window.simRenderer.toggleLayer(layer);
  });
});

document.getElementById('ignitionZoneSelect')?.addEventListener('change', (e) => {
  _applyRoutePreview(e.target.value);
  if (window.simRenderer) window.simRenderer.setRoute(_routePoints, parseInt(document.getElementById('simSteps').value) || 90);
});

document.getElementById('simSteps')?.addEventListener('input', () => {
  if (window.simRenderer) window.simRenderer.setRoute(_routePoints, parseInt(document.getElementById('simSteps').value) || 90);
});

function _wireBlueprintCanvasInteraction() {
  const canvas = document.getElementById('simCanvas');
  if (!canvas || canvas.dataset.wired === '1') return;
  canvas.dataset.wired = '1';

  const pick = (e) => {
    if (!window.simRenderer?.canvasToWorld || !_modelData) return;
    const rect = canvas.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const py = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const world = window.simRenderer.canvasToWorld(px, py);
    const zone = _zoneForWorldPoint(world) || _worstCaseIgnitionZone();
    _applyRoutePreview(zone?.id || zone, world);
  };

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pick(e);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (e.buttons) pick(e);
  });
}

// ─── Download Report ──────────────────────────────────────────────────────────

document.getElementById('downloadReportBtn')?.addEventListener('click', () => {
  if (!_simId) return;
  window.open(`${API}/api/simulation/${_simId}/report`, '_blank');
});

document.getElementById('viewResultsBtn')?.addEventListener('click', () => {
  if (!_results) return;
  const blob = new Blob([JSON.stringify(_results, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `firepbd_results_${_simId}.json`; a.click();
  URL.revokeObjectURL(url);
});

// ─── Range inputs live display ────────────────────────────────────────────────

const sliders = [
  ['cfgSpreadProb', 'cfgSpreadProbVal', v => v],
  ['cfgWalkSpeed', 'cfgWalkSpeedVal', v => v + ' m/s'],
  ['cfgConfidence', 'cfgConfidenceVal', v => v + '%'],
];
sliders.forEach(([inputId, displayId, fmt]) => {
  const el = document.getElementById(inputId);
  const display = document.getElementById(displayId);
  if (el && display) {
    el.addEventListener('input', () => { display.textContent = fmt(el.value); });
  }
});

// ─── Initialisation ───────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setInterval(checkApiHealth, 30000);
  document.getElementById('runSimBtn').disabled = true;
  _setPlaybackControls();
  _wireBlueprintCanvasInteraction();
  simLog('[FirePBD Engine v1.0] Ready. Upload a blueprint to begin.');
  simLog('[INFO] Supported formats: CubiCasa SVG (recommended), PNG, JPG');
  simLog('[INFO] Backend: http://localhost:8000 | Docs: http://localhost:8000/api/docs');
});
