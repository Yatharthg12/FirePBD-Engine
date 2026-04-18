"""
FirePBD Engine — Professional PDF Report Generator
====================================================
Generates a fully formatted, standards-compliant fire safety analysis report
using ReportLab.

Report Sections:
  1. Cover Page — project info, risk score, compliance status
  2. Executive Summary — key findings, verdict, urgent actions
  3. Building Geometry — zone table, area, occupancy
  4. Fire Scenario — ignition, HRR curve, timeline
  5. Zone Hazard Timeline — temperature / smoke / FED per zone
  6. Evacuation Results — per-agent outcomes, RSET curve
  7. RSET vs ASET Analysis — margin chart, compliance verdict
  8. Monte Carlo Results — probability distributions
  9. Optimization Recommendations — ranked action table
 10. Compliance Mapping — NFPA 101, BS 9999, IS 16009
 11. Appendix — simulation parameters, standards glossary
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

from backend.config import REPORTS_DIR, REPORT_COMPANY_NAME
from backend.core.constants import ASET_SAFETY_MARGIN_S, STANDARDS
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Colour Palette ──────────────────────────────────────────────────────────
C_DARK = colors.HexColor("#0a0e1a")
C_PRIMARY = colors.HexColor("#f97316")
C_CARD = colors.HexColor("#111827")
C_SAFE = colors.HexColor("#22c55e")
C_WARNING = colors.HexColor("#f59e0b")
C_DANGER = colors.HexColor("#ef4444")
C_TEXT = colors.HexColor("#1f2937")
C_SUBTEXT = colors.HexColor("#6b7280")
C_WHITE = colors.white
C_LIGHT_GREY = colors.HexColor("#f3f4f6")
C_BORDER = colors.HexColor("#e5e7eb")

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


class ReportGenerator:
    """
    Generates a comprehensive PDF fire safety analysis report.

    Usage:
        gen = ReportGenerator(risk_report, evac_summary, model, sim_state)
        path = gen.generate(output_dir)
    """

    def __init__(
        self,
        risk_report: dict,
        evac_summary: dict,
        model,
        sim_state,
        mc_result=None,
        recommendations: Optional[List] = None,
        zone_status_history: Optional[List] = None,
    ) -> None:
        self.risk_report = risk_report
        self.evac_summary = evac_summary
        self.model = model
        self.sim_state = sim_state
        self.mc_result = mc_result
        self.recommendations = recommendations or []
        self.zone_status_history = zone_status_history or []

        self._styles = self._build_styles()

    def generate(self, output_dir: Optional[str] = None) -> str:
        """Generate PDF and return the output file path."""
        out_dir = Path(output_dir or REPORTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"FirePBD_Report_{self.model.building_id}_{timestamp}.pdf"
        out_path = str(out_dir / fname)

        doc = SimpleDocTemplate(
            out_path,
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN,
            title=f"FirePBD Engine — Fire Safety Report",
            author=REPORT_COMPANY_NAME,
            subject="Performance-Based Fire Safety Analysis",
        )

        story = []
        story += self._cover_page()
        story.append(PageBreak())
        story += self._executive_summary()
        story.append(PageBreak())
        story += self._building_geometry()
        story += self._fire_scenario()
        story.append(PageBreak())
        story += self._evacuation_results()
        story.append(PageBreak())
        story += self._rset_aset_analysis()
        if self.mc_result:
            story += self._monte_carlo_results()
        if self.recommendations:
            story.append(PageBreak())
            story += self._optimization_section()
        story += self._compliance_mapping()
        story += self._appendix()

        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        logger.info(f"PDF report generated: {out_path}")
        return out_path

    # ─── Cover Page ───────────────────────────────────────────────────────────

    def _cover_page(self) -> list:
        s = self._styles
        story = []

        story.append(Spacer(1, 1.5 * cm))

        # Title block
        story.append(Paragraph("FIRE SAFETY ANALYSIS REPORT", s["cover_title"]))
        story.append(Paragraph("Performance-Based Design (PBD) Assessment", s["cover_subtitle"]))
        story.append(HRFlowable(width="100%", thickness=3, color=C_PRIMARY))
        story.append(Spacer(1, 0.5 * cm))

        # Building info table
        risk_score = self.risk_report.get("risk_score", 0)
        risk_level = self.risk_report.get("risk_level", "UNKNOWN")
        compliant = self.risk_report.get("compliant", False)
        risk_colour = (
            C_DANGER if risk_level in ("CRITICAL", "HIGH")
            else C_WARNING if risk_level == "MEDIUM"
            else C_SAFE
        )

        info_data = [
            ["Building ID", self.risk_report.get("building_id", "—")],
            ["Simulation ID", self.risk_report.get("simulation_id", "—")],
            ["Analysis Date", datetime.now().strftime("%d %B %Y")],
            ["Analysis Time", datetime.now().strftime("%H:%M")],
            ["Prepared By", REPORT_COMPANY_NAME],
            ["Standard", "BS 9999:2017 / NFPA 101:2021 / ISO 13571:2012"],
        ]
        info_table = Table(info_data, colWidths=[5 * cm, 12 * cm])
        info_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), C_SUBTEXT),
            ("TEXTCOLOR", (1, 0), (1, -1), C_TEXT),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LIGHT_GREY, C_WHITE]),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.8 * cm))

        # Risk score box
        verdict = "COMPLIANT" if compliant else "NON-COMPLIANT"
        v_colour = C_SAFE if compliant else C_DANGER

        summary_data = [
            ["OVERALL RISK SCORE", f"{risk_score}/100"],
            ["RISK LEVEL", risk_level],
            ["COMPLIANCE VERDICT", verdict],
            ["RSET", f"{self.risk_report.get('rset_s', '—')} s"],
            ["ASET", f"{self.risk_report.get('aset_s', '—')} s"],
            ["RSET/ASET MARGIN", f"{self.risk_report.get('rset_aset_margin_s', '—')} s"],
        ]
        summary_table = Table(summary_data, colWidths=[9 * cm, 8 * cm])
        summary_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 12),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LIGHT_GREY, C_WHITE]),
            ("TEXTCOLOR", (0, 0), (0, -1), C_TEXT),
            ("TEXTCOLOR", (1, 1), (1, 1), risk_colour),
            ("TEXTCOLOR", (1, 2), (1, 2), v_colour),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.5 * cm))

        # Disclaimer
        disclaimer = (
            "<i>This report has been generated by FirePBD Engine, an AI-assisted "
            "fire safety simulation system. Results are based on computational models "
            "and are for decision-support purposes. All critical safety decisions must "
            "be reviewed by a qualified fire safety engineer in accordance with local "
            "regulations. FirePBD Engine does not replace a full engineering assessment.</i>"
        )
        story.append(Paragraph(disclaimer, s["disclaimer"]))
        return story

    # ─── Executive Summary ────────────────────────────────────────────────────

    def _executive_summary(self) -> list:
        s = self._styles
        story = [Paragraph("1. Executive Summary", s["h1"]), HRFlowable(width="100%", thickness=1, color=C_BORDER)]

        risk_score = self.risk_report.get("risk_score", 0)
        risk_level = self.risk_report.get("risk_level", "UNKNOWN")
        margin = self.risk_report.get("rset_aset_margin_s")
        evac_pct = self.risk_report.get("evacuation_success_pct", 0)
        rset = self.risk_report.get("rset_s")
        aset = self.risk_report.get("aset_s")

        summary_text = (
            f"The fire safety simulation of building <b>{self.model.building_id}</b> "
            f"yielded an overall <b>risk score of {risk_score}/100</b> ({risk_level}). "
            f"The simulation modelled fire propagation from the specified ignition point "
            f"and evacuation of {self.evac_summary.get('total_agents', 0)} occupants "
            f"over {self.sim_state.current_step} simulation steps "
            f"({self.sim_state.current_step * 5:.0f} seconds)."
        )
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(summary_text, s["body"]))

        # Key metrics
        metrics = [
            ["Metric", "Value", "Requirement", "Status"],
            ["RSET", f"{rset:.0f}s" if rset else "—", "< ASET − 120s", "PASS" if (margin or 0) >= ASET_SAFETY_MARGIN_S else "FAIL"],
            ["ASET", f"{aset:.0f}s" if aset else "Not reached", "—", "—"],
            ["RSET/ASET Margin", f"{margin:.0f}s" if margin else "—", f"≥ {ASET_SAFETY_MARGIN_S:.0f}s (BS 9999)", "PASS" if (margin or 0) >= ASET_SAFETY_MARGIN_S else "FAIL"],
            ["Evacuation Success", f"{evac_pct:.1f}%", "100%", "PASS" if evac_pct >= 95 else "FAIL"],
            ["Dead", str(self.evac_summary.get("dead", 0)), "0", "PASS" if self.evac_summary.get("dead", 0) == 0 else "FAIL"],
        ]
        t = self._coloured_table(metrics, [5 * cm, 4 * cm, 5 * cm, 3 * cm])
        story.append(Spacer(1, 0.4 * cm))
        story.append(t)

        # Urgent actions
        critical_recs = [r for r in self.recommendations if r.priority == "CRITICAL"]
        if critical_recs:
            story.append(Spacer(1, 0.5 * cm))
            story.append(Paragraph("⚠ Urgent Actions Required:", s["h3"]))
            for rec in critical_recs[:5]:
                story.append(Paragraph(f"• <b>{rec.title}</b>: {rec.description[:200]}...", s["body"]))

        return story

    # ─── Building Geometry ────────────────────────────────────────────────────

    def _building_geometry(self) -> list:
        s = self._styles
        story = [
            Paragraph("2. Building Geometry", s["h1"]),
            HRFlowable(width="100%", thickness=1, color=C_BORDER),
            Spacer(1, 0.3 * cm),
        ]

        story.append(Paragraph(
            f"Total floor area: {self.model.total_area_m2:.1f} m² | "
            f"Zones: {len(self.model.zones)} | "
            f"Openings: {len(self.model.openings)} | "
            f"Exits: {len(self.model.exit_zones)}",
            s["body"]
        ))
        story.append(Spacer(1, 0.3 * cm))

        zone_data = [["Zone ID", "Area (m²)", "Max Occ.", "Fuel Load (MJ/m²)", "Exit"]]
        for zone in list(self.model.zones.values())[:20]:
            zone_data.append([
                zone.id,
                f"{zone.area:.1f}",
                str(zone.max_occupants),
                f"{zone.fuel_load_density:.0f}",
                "✓" if zone.is_exit else "—",
            ])
        story.append(self._coloured_table(zone_data, [3.5 * cm, 3 * cm, 3 * cm, 4 * cm, 2 * cm]))

        return story

    # ─── Fire Scenario ────────────────────────────────────────────────────────

    def _fire_scenario(self) -> list:
        s = self._styles
        story = [
            Spacer(1, 0.5 * cm),
            Paragraph("3. Fire Scenario", s["h1"]),
            HRFlowable(width="100%", thickness=1, color=C_BORDER),
            Spacer(1, 0.3 * cm),
        ]

        flashover_t = self.risk_report.get("flashover_time_s")
        story.append(Paragraph(
            f"<b>Ignition zone:</b> {self.sim_state.ignition_zone} | "
            f"<b>Flashover:</b> {'t=' + str(flashover_t) + 's' if flashover_t else 'Not observed'} | "
            f"<b>Simulation duration:</b> {self.sim_state.sim_time_s:.0f}s",
            s["body"]
        ))

        story.append(Spacer(1, 0.3 * cm))
        events_text = " → ".join(self.sim_state.events[:10])
        if events_text:
            story.append(Paragraph(f"<b>Event timeline:</b> {events_text}", s["body"]))

        return story

    # ─── Evacuation Results ───────────────────────────────────────────────────

    def _evacuation_results(self) -> list:
        s = self._styles
        story = [
            Paragraph("4. Evacuation Results", s["h1"]),
            HRFlowable(width="100%", thickness=1, color=C_BORDER),
            Spacer(1, 0.3 * cm),
        ]

        ev = self.evac_summary
        story.append(Paragraph(
            f"Total agents: {ev.get('total_agents', 0)} | "
            f"Evacuated: {ev.get('evacuated', 0)} "
            f"({ev.get('evacuation_success_pct', 0):.1f}%) | "
            f"Incapacitated: {ev.get('incapacitated', 0)} | "
            f"Dead: {ev.get('dead', 0)} | "
            f"Mean RSET: {ev.get('mean_evac_time_s', '—')}s",
            s["body"]
        ))

        # Agent outcome table (first 15)
        persons = ev.get("persons", [])[:15]
        if persons:
            p_data = [["Agent ID", "Start Zone", "Status", "Time (s)", "FED", "Reroutes"]]
            for p in persons:
                p_data.append([
                    p.get("id", "—"),
                    p.get("start_zone", "—"),
                    p.get("status", "—"),
                    str(p.get("time_taken_s", "—")),
                    f"{p.get('fed_accumulated', 0):.3f}",
                    str(p.get("reroute_count", 0)),
                ])
            story.append(Spacer(1, 0.3 * cm))
            story.append(self._coloured_table(p_data, [3 * cm, 3 * cm, 3.5 * cm, 2.5 * cm, 2 * cm, 2 * cm]))

        return story

    # ─── RSET/ASET Analysis ───────────────────────────────────────────────────

    def _rset_aset_analysis(self) -> list:
        s = self._styles
        story = [
            Paragraph("5. RSET vs ASET Analysis", s["h1"]),
            HRFlowable(width="100%", thickness=1, color=C_BORDER),
            Spacer(1, 0.3 * cm),
        ]

        rset = self.risk_report.get("rset_s")
        aset = self.risk_report.get("aset_s")
        margin = self.risk_report.get("rset_aset_margin_s")
        assessment = self.risk_report.get("margin_assessment", "UNKNOWN")

        story.append(Paragraph(
            "<b>Framework:</b> BS 9999:2017 Annex D — Available Safe Egress Time vs "
            "Required Safe Egress Time. For a building to be deemed safe: "
            "ASET − RSET ≥ 120 seconds.",
            s["body"]
        ))
        story.append(Spacer(1, 0.3 * cm))

        colour_map = {"ADEQUATE": C_SAFE, "MARGINAL": C_WARNING, "INADEQUATE": C_DANGER, "UNKNOWN": C_SUBTEXT}
        a_colour = colour_map.get(assessment, C_SUBTEXT)

        data = [
            ["Parameter", "Value", "Notes"],
            ["RSET (Required Safe Egress Time)", f"{rset:.0f}s" if rset else "—", f"Includes ×{1.25} safety factor (BS 9999)"],
            ["ASET (Available Safe Egress Time)", f"{aset:.0f}s" if aset else "Not reached", "Time to tenability breach"],
            ["ASET − RSET Margin", f"{margin:.0f}s" if margin else "—", f"Required: ≥ {ASET_SAFETY_MARGIN_S:.0f}s"],
            ["Compliance Assessment", assessment, "BS 9999:2017 Annex D"],
        ]
        story.append(self._coloured_table(data, [6 * cm, 4 * cm, 7 * cm]))

        return story

    # ─── Monte Carlo ──────────────────────────────────────────────────────────

    def _monte_carlo_results(self) -> list:
        s = self._styles
        mc = self.mc_result
        story = [
            PageBreak(),
            Paragraph("6. Monte Carlo Probabilistic Analysis", s["h1"]),
            HRFlowable(width="100%", thickness=1, color=C_BORDER),
            Spacer(1, 0.3 * cm),
        ]

        story.append(Paragraph(
            f"<b>{mc.n_runs} simulation runs</b> with varied ignition location, "
            f"occupant speed (SFPE), fire load (±20%), and reaction time (BS 9999).",
            s["body"]
        ))

        mc_data = [
            ["Metric", "Value"],
            ["RSET Mean", f"{mc.rset_mean_s:.0f}s"],
            ["RSET 90th Percentile (conservative)", f"{mc.rset_p90_s:.0f}s"],
            ["RSET 95th Percentile", f"{mc.rset_p95_s:.0f}s"],
            ["RSET 90% Confidence Interval", f"[{mc.rset_ci_low:.0f}s, {mc.rset_ci_high:.0f}s]"],
            ["ASET Mean", f"{mc.aset_mean_s:.0f}s"],
            ["ASET 10th Percentile (worst case)", f"{mc.aset_p10_s:.0f}s"],
            ["Evacuation Success Mean", f"{mc.evacuation_success_mean_pct:.1f}%"],
            ["Evacuation Success P10", f"{mc.evacuation_success_p10_pct:.1f}%"],
            ["BS 9999 Pass Rate", f"{mc.pass_rate * 100:.1f}%"],
        ]
        story.append(Spacer(1, 0.3 * cm))
        story.append(self._coloured_table(mc_data, [10 * cm, 7 * cm]))

        return story

    # ─── Optimization Recommendations ────────────────────────────────────────

    def _optimization_section(self) -> list:
        s = self._styles
        story = [
            Paragraph("7. Optimization Recommendations", s["h1"]),
            HRFlowable(width="100%", thickness=1, color=C_BORDER),
            Spacer(1, 0.3 * cm),
        ]

        rec_data = [["ID", "Priority", "Title", "RSET Reduction", "Reference"]]
        for rec in self.recommendations:
            rec_data.append([
                rec.rec_id,
                rec.priority,
                rec.title[:60] + ("..." if len(rec.title) > 60 else ""),
                f"−{rec.estimated_rset_reduction_s:.0f}s",
                rec.standard_reference[:30],
            ])

        story.append(self._coloured_table(
            rec_data,
            [1.5 * cm, 2.5 * cm, 8 * cm, 2.5 * cm, 3.5 * cm]
        ))

        return story

    # ─── Compliance Mapping ───────────────────────────────────────────────────

    def _compliance_mapping(self) -> list:
        s = self._styles
        story = [
            PageBreak(),
            Paragraph("8. Compliance Standards Reference", s["h1"]),
            HRFlowable(width="100%", thickness=1, color=C_BORDER),
            Spacer(1, 0.3 * cm),
        ]

        std_data = [["Standard", "Description", "Application"]]
        std_apps = {
            "NFPA_101": "Means of egress, occupancy load, exit width",
            "BS_9999": "RSET/ASET framework, evacuation strategy, fire compartmentation",
            "ISO_13571": "FED incapacitation model, tenability criteria",
            "SFPE": "Speed-density model, HRR data, smoke toxicity yields",
            "NFPA_557": "Fuel load density reference values",
            "IS_16009": "Indian standard for fire hazard assessment methodology",
        }
        for key, desc in STANDARDS.items():
            std_data.append([key, desc, std_apps.get(key, "Reference")])

        story.append(self._coloured_table(std_data, [3 * cm, 9 * cm, 6 * cm]))
        return story

    # ─── Appendix ─────────────────────────────────────────────────────────────

    def _appendix(self) -> list:
        s = self._styles
        story = [
            PageBreak(),
            Paragraph("Appendix — Simulation Parameters", s["h1"]),
            HRFlowable(width="100%", thickness=1, color=C_BORDER),
            Spacer(1, 0.3 * cm),
        ]

        params = [
            ["Parameter", "Value"],
            ["Simulation Timestep", f"{self.sim_state.dt_s}s"],
            ["Total Steps", str(self.sim_state.current_step)],
            ["Grid Cell Size", f"0.5m × 0.5m"],
            ["Fire Spread Model", "Vectorised NumPy, NFPA HRR constants"],
            ["Evacuation Model", "SFPE speed-density, ISO 13571 FED"],
            ["Ignition Zone", self.sim_state.ignition_zone],
            ["Total Agents", str(self.evac_summary.get("total_agents", 0))],
        ]
        story.append(self._coloured_table(params, [8 * cm, 9 * cm]))
        return story

    # ─── Header/Footer ────────────────────────────────────────────────────────

    def _header_footer(self, canvas, doc) -> None:
        canvas.saveState()
        # Header line
        canvas.setStrokeColor(C_PRIMARY)
        canvas.setLineWidth(2)
        canvas.line(MARGIN, PAGE_H - 1.5 * cm, PAGE_W - MARGIN, PAGE_H - 1.5 * cm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(C_SUBTEXT)
        canvas.drawString(MARGIN, PAGE_H - 1.2 * cm, f"FirePBD Engine — Fire Safety Analysis Report")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.2 * cm, f"Building: {self.model.building_id}")
        # Footer
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 1.5 * cm, PAGE_W - MARGIN, 1.5 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(MARGIN, 0.8 * cm, f"CONFIDENTIAL — {REPORT_COMPANY_NAME}")
        canvas.drawCentredString(PAGE_W / 2, 0.8 * cm, datetime.now().strftime("%d %b %Y"))
        canvas.drawRightString(PAGE_W - MARGIN, 0.8 * cm, f"Page {doc.page}")
        canvas.restoreState()

    # ─── Style Builder ────────────────────────────────────────────────────────

    def _build_styles(self) -> dict:
        base = getSampleStyleSheet()
        return {
            "cover_title": ParagraphStyle(
                "cover_title", fontSize=26, fontName="Helvetica-Bold",
                textColor=C_DARK, spaceAfter=6, alignment=TA_CENTER
            ),
            "cover_subtitle": ParagraphStyle(
                "cover_subtitle", fontSize=13, fontName="Helvetica",
                textColor=C_SUBTEXT, spaceAfter=20, alignment=TA_CENTER
            ),
            "h1": ParagraphStyle(
                "h1", fontSize=16, fontName="Helvetica-Bold",
                textColor=C_DARK, spaceBefore=12, spaceAfter=6
            ),
            "h2": ParagraphStyle(
                "h2", fontSize=13, fontName="Helvetica-Bold",
                textColor=C_TEXT, spaceBefore=8, spaceAfter=4
            ),
            "h3": ParagraphStyle(
                "h3", fontSize=11, fontName="Helvetica-Bold",
                textColor=C_TEXT, spaceBefore=6, spaceAfter=3
            ),
            "body": ParagraphStyle(
                "body", fontSize=9, fontName="Helvetica",
                textColor=C_TEXT, spaceAfter=4, leading=14
            ),
            "disclaimer": ParagraphStyle(
                "disclaimer", fontSize=8, fontName="Helvetica-Oblique",
                textColor=C_SUBTEXT, spaceBefore=10, spaceAfter=4
            ),
        }

    def _coloured_table(self, data: list, col_widths: list) -> Table:
        t = Table(data, colWidths=col_widths)
        style = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT_GREY]),
            ("GRID", (0, 0), (-1, -1), 0.3, C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]
        # Colour PASS/FAIL in rows
        for i, row in enumerate(data[1:], 1):
            for j, cell in enumerate(row):
                if str(cell) == "PASS":
                    style.append(("TEXTCOLOR", (j, i), (j, i), C_SAFE))
                    style.append(("FONTNAME", (j, i), (j, i), "Helvetica-Bold"))
                elif str(cell) == "FAIL":
                    style.append(("TEXTCOLOR", (j, i), (j, i), C_DANGER))
                    style.append(("FONTNAME", (j, i), (j, i), "Helvetica-Bold"))
                elif str(cell) == "CRITICAL":
                    style.append(("TEXTCOLOR", (j, i), (j, i), C_DANGER))
                    style.append(("FONTNAME", (j, i), (j, i), "Helvetica-Bold"))
                elif str(cell) == "HIGH":
                    style.append(("TEXTCOLOR", (j, i), (j, i), C_WARNING))
        t.setStyle(TableStyle(style))
        return t
