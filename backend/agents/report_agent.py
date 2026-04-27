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

import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Line, Rect, String
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

    # ─── Report Helpers ─────────────────────────────────────────────────────

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _format_seconds(self, value) -> str:
        return "—" if value is None else f"{self._safe_float(value):.0f}s"

    def _zone_history_stats(self) -> List[dict]:
        if not self.zone_status_history:
            return []

        zones = defaultdict(lambda: {
            "zone_id": "",
            "peak_temp": float("-inf"),
            "min_visibility": float("inf"),
            "danger_hits": 0,
            "samples": 0,
            "last_temp": None,
            "last_visibility": None,
            "last_danger": "UNKNOWN",
        })

        for snapshot in self.zone_status_history:
            for zone_id, status in snapshot.items():
                rec = zones[zone_id]
                rec["zone_id"] = zone_id
                temp = self._safe_float(status.get("avg_temp_c"), 0.0)
                vis = self._safe_float(status.get("min_visibility_m"), 999.0)
                danger = str(status.get("danger", "UNKNOWN")).upper()
                rec["peak_temp"] = max(rec["peak_temp"], temp)
                rec["min_visibility"] = min(rec["min_visibility"], vis)
                rec["danger_hits"] += 1 if danger in {"HIGH", "EXTREME", "UNTENABLE"} else 0
                rec["samples"] += 1
                rec["last_temp"] = temp
                rec["last_visibility"] = vis
                rec["last_danger"] = danger

        stats = list(zones.values())
        stats.sort(key=lambda r: (r["peak_temp"], r["danger_hits"], -r["min_visibility"]), reverse=True)
        return stats

    def _chart_wrapper(self, title: str, drawing, caption: str) -> list:
        story = [Paragraph(title, self._styles["h2"])]
        if drawing is not None:
            story.extend([drawing, Spacer(1, 0.1 * cm)])
        story.append(Paragraph(caption, self._styles["body"]))
        return story

    def _make_metric_cards(self) -> Table:
        data = [
            [
                Paragraph("<b>Risk Score</b>", self._styles["body"]),
                Paragraph("<b>RSET</b>", self._styles["body"]),
                Paragraph("<b>ASET</b>", self._styles["body"]),
                Paragraph("<b>Evacuation Success</b>", self._styles["body"]),
            ],
            [
                Paragraph(f"{self.risk_report.get('risk_score', 0)}/100", self._styles["h3"]),
                Paragraph(self._format_seconds(self.risk_report.get("rset_s")), self._styles["h3"]),
                Paragraph(self._format_seconds(self.risk_report.get("aset_s")), self._styles["h3"]),
                Paragraph(f"{self.evac_summary.get('evacuation_success_pct', 0):.1f}%", self._styles["h3"]),
            ],
        ]
        table = Table(data, colWidths=[4.2 * cm] * 4)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("BACKGROUND", (0, 1), (-1, 1), C_LIGHT_GREY),
            ("BOX", (0, 0), (-1, -1), 0.8, C_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, C_BORDER),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return table

    def _make_evacuation_pie(self):
        total = int(self.evac_summary.get("total_agents", 0))
        evacuated = int(self.evac_summary.get("evacuated", 0))
        incapacitated = int(self.evac_summary.get("incapacitated", 0))
        dead = int(self.evac_summary.get("dead", 0))
        if total <= 0 or (evacuated + incapacitated + dead) <= 0:
            return None

        drawing = Drawing(460, 220)
        pie = Pie()
        pie.x = 110
        pie.y = 15
        pie.width = 170
        pie.height = 170
        pie.data = [evacuated, incapacitated, dead]
        pie.labels = [f"Evacuated {evacuated}", f"Incapacitated {incapacitated}", f"Dead {dead}"]
        pie.slices.strokeWidth = 0.5
        pie.slices[0].fillColor = C_SAFE
        pie.slices[1].fillColor = C_WARNING
        pie.slices[2].fillColor = C_DANGER
        drawing.add(pie)
        drawing.add(String(20, 192, "Occupant outcome split", fontName="Helvetica-Bold", fontSize=11, fillColor=C_TEXT))
        return drawing

    def _make_zone_hazard_chart(self):
        stats = self._zone_history_stats()[:8]
        if not stats:
            return None

        drawing = Drawing(500, 240)
        chart = VerticalBarChart()
        chart.x = 40
        chart.y = 45
        chart.height = 150
        chart.width = 420
        chart.data = [[round(max(item["peak_temp"], 0.0), 1) for item in stats]]
        chart.categoryAxis.categoryNames = [item["zone_id"][:10] for item in stats]
        chart.categoryAxis.labels.boxAnchor = "n"
        chart.categoryAxis.labels.angle = 35
        chart.categoryAxis.labels.dy = -10
        chart.categoryAxis.labels.fontSize = 7
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueStep = max(100, math.ceil(max(item["peak_temp"] for item in stats) / 5 / 50) * 50)
        chart.valueAxis.labelTextFormat = "%.0f°C"
        chart.bars[0].fillColor = C_PRIMARY
        chart.barWidth = 18
        chart.groupSpacing = 12
        drawing.add(chart)
        drawing.add(String(40, 215, "Hottest zones by peak temperature", fontName="Helvetica-Bold", fontSize=11, fillColor=C_TEXT))
        return drawing

    def _make_rset_aset_chart(self):
        rset = self._safe_float(self.risk_report.get("rset_s"), 0.0)
        aset = self._safe_float(self.risk_report.get("aset_s"), 0.0)
        if not rset and not aset:
            return None

        drawing = Drawing(500, 220)
        chart = HorizontalBarChart()
        chart.x = 105
        chart.y = 35
        chart.height = 130
        chart.width = 330
        chart.data = [[rset, aset]]
        chart.categoryAxis.categoryNames = ["RSET", "ASET"]
        chart.categoryAxis.labels.fontSize = 9
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(rset, aset) * 1.2 if max(rset, aset) > 0 else 1
        chart.valueAxis.labelTextFormat = "%.0fs"
        chart.bars[0].fillColor = C_DANGER
        chart.barWidth = 18
        drawing.add(chart)
        drawing.add(String(40, 190, "Egress time comparison", fontName="Helvetica-Bold", fontSize=11, fillColor=C_TEXT))
        return drawing

    def _make_mc_histogram(self):
        if not self.mc_result or not getattr(self.mc_result, "run_details", None):
            return None

        rset_values = [r.rset_s for r in self.mc_result.run_details if getattr(r, "rset_s", None) is not None]
        if not rset_values:
            return None

        lo = min(rset_values)
        hi = max(rset_values)
        if math.isclose(lo, hi):
            hi = lo + 1
        bins = 6
        step = (hi - lo) / bins
        counts = [0] * bins
        for val in rset_values:
            idx = min(int((val - lo) / step), bins - 1)
            counts[idx] += 1

        labels = []
        for i in range(bins):
            start = lo + i * step
            end = lo + (i + 1) * step
            labels.append(f"{start:.0f}-{end:.0f}")

        drawing = Drawing(500, 240)
        chart = VerticalBarChart()
        chart.x = 40
        chart.y = 45
        chart.height = 150
        chart.width = 420
        chart.data = [counts]
        chart.categoryAxis.categoryNames = labels
        chart.categoryAxis.labels.angle = 35
        chart.categoryAxis.labels.dy = -10
        chart.categoryAxis.labels.fontSize = 7
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueStep = max(1, math.ceil(max(counts) / 4) or 1)
        chart.bars[0].fillColor = C_PRIMARY
        chart.barWidth = 18
        drawing.add(chart)
        drawing.add(String(40, 215, "Monte Carlo RSET distribution", fontName="Helvetica-Bold", fontSize=11, fillColor=C_TEXT))
        return drawing

    def _make_bottleneck_chart(self):
        bottlenecks = self.risk_report.get("bottlenecks") or []
        if not bottlenecks:
            return None

        labels = []
        values = []
        for item in bottlenecks[:5]:
            labels.append(str(item.get("zone_a") or item.get("zone_id") or item.get("type", "N/A"))[:12])
            values.append(self._safe_float(item.get("centrality_score"), 0.0))

        if not values:
            return None

        drawing = Drawing(500, 220)
        chart = HorizontalBarChart()
        chart.x = 130
        chart.y = 35
        chart.height = 130
        chart.width = 310
        chart.data = [values]
        chart.categoryAxis.categoryNames = labels
        chart.categoryAxis.labels.fontSize = 8
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(values) * 1.2 if max(values) > 0 else 1
        chart.bars[0].fillColor = C_WARNING
        chart.barWidth = 16
        drawing.add(chart)
        drawing.add(String(30, 190, "Main bottleneck zones", fontName="Helvetica-Bold", fontSize=11, fillColor=C_TEXT))
        return drawing

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

        story.append(self._make_metric_cards())
        story.append(Spacer(1, 0.6 * cm))

        # Risk score box
        verdict = "COMPLIANT" if compliant else "NON-COMPLIANT"
        v_colour = C_SAFE if compliant else C_DANGER

        summary_data = [
            ["OVERALL RISK SCORE", f"{risk_score}/100"],
            ["RISK LEVEL", risk_level],
            ["COMPLIANCE VERDICT", verdict],
            ["RSET", self._format_seconds(self.risk_report.get("rset_s"))],
            ["ASET", self._format_seconds(self.risk_report.get("aset_s"))],
            ["RSET/ASET MARGIN", self._format_seconds(self.risk_report.get("rset_aset_margin_s"))],
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
            ["RSET", self._format_seconds(rset), "< ASET − 120s", "PASS" if margin is not None and margin >= ASET_SAFETY_MARGIN_S else "FAIL"],
            ["ASET", self._format_seconds(aset) if aset is not None else "Not reached", "—", "—"],
            ["RSET/ASET Margin", self._format_seconds(margin), f"≥ {ASET_SAFETY_MARGIN_S:.0f}s (BS 9999)", "PASS" if margin is not None and margin >= ASET_SAFETY_MARGIN_S else "FAIL"],
            ["Evacuation Success", f"{evac_pct:.1f}%", "100%", "PASS" if evac_pct >= 95 else "FAIL"],
            ["Dead", str(self.evac_summary.get("dead", 0)), "0", "PASS" if self.evac_summary.get("dead", 0) == 0 else "FAIL"],
        ]
        t = self._coloured_table(metrics, [5 * cm, 4 * cm, 5 * cm, 3 * cm])
        story.append(Spacer(1, 0.4 * cm))
        story.append(t)

        risk_factors = [
            ["Driver", "Observed value", "Engineering note"],
            ["Occupant outcome", f"{evac_pct:.1f}%", "Lower success increases life-safety risk"],
            ["RSET", self._format_seconds(rset), "Evacuation completion time"],
            ["ASET", self._format_seconds(aset), "Tenability limit"],
            ["Margin", self._format_seconds(margin), "Positive margin indicates available safety buffer"],
        ]
        story.append(Spacer(1, 0.35 * cm))
        story.append(self._coloured_table(risk_factors, [5 * cm, 4.5 * cm, 7 * cm]))

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

        stats = self._zone_history_stats()
        if stats:
            story.append(Spacer(1, 0.4 * cm))
            story.append(Paragraph("Zone hazard snapshot", s["h2"]))
            story.append(Paragraph(
                "The table below ranks zones by peak observed temperature and visibility collapse "
                "across the simulation history.",
                s["body"]
            ))

            hazard_rows = [["Zone", "Peak Temp", "Min Visibility", "Danger Hits", "Last State"]]
            for row in stats[:10]:
                hazard_rows.append([
                    row["zone_id"],
                    f"{row['peak_temp']:.1f}°C",
                    f"{row['min_visibility']:.1f} m",
                    str(row["danger_hits"]),
                    row["last_danger"],
                ])
            story.append(self._coloured_table(hazard_rows, [3 * cm, 3 * cm, 3 * cm, 2.5 * cm, 4 * cm]))
            story.append(Spacer(1, 0.25 * cm))
            zone_chart = self._make_zone_hazard_chart()
            if zone_chart is not None:
                story.append(zone_chart)

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

        if self.risk_report.get("bottlenecks"):
            story.append(Spacer(1, 0.25 * cm))
            bottleneck_text = ", ".join(
                f"{b.get('zone_a') or b.get('zone_id')}" for b in self.risk_report.get("bottlenecks", [])[:5]
            )
            story.append(Paragraph(
                f"<b>Key congestion zones:</b> {bottleneck_text}",
                s["body"]
            ))

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

        outcome_table = [
            ["Outcome", "Count", "Share"],
            ["Evacuated", str(ev.get("evacuated", 0)), f"{ev.get('evacuation_success_pct', 0):.1f}%"],
            ["Incapacitated", str(ev.get("incapacitated", 0)), f"{(100 * ev.get('incapacitated', 0) / max(ev.get('total_agents', 0), 1)):.1f}%"],
            ["Dead", str(ev.get("dead", 0)), f"{(100 * ev.get('dead', 0) / max(ev.get('total_agents', 0), 1)):.1f}%"],
        ]
        story.append(Spacer(1, 0.25 * cm))
        story.append(self._coloured_table(outcome_table, [5 * cm, 4 * cm, 4 * cm]))

        evac_chart = self._make_evacuation_pie()
        if evac_chart is not None:
            story.append(Spacer(1, 0.2 * cm))
            story.append(evac_chart)

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

        chart = self._make_rset_aset_chart()
        if chart is not None:
            story.append(Spacer(1, 0.35 * cm))
            story.append(chart)

        interpretation = {
            "ADEQUATE": "The available safe egress time is comfortably above the evacuation demand; the design is in the safe band.",
            "MARGINAL": "The safety margin exists but is tight; this would usually trigger engineering review and mitigation.",
            "INADEQUATE": "Evacuation demand exceeds available tenability; the scenario is not acceptable without design changes.",
            "UNKNOWN": "One or both inputs were unavailable, so the margin could not be assessed reliably.",
        }.get(assessment, "No interpretation available.")
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(f"<b>Engineering interpretation:</b> {interpretation}", s["body"]))

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

        hist = self._make_mc_histogram()
        if hist is not None:
            story.append(Spacer(1, 0.35 * cm))
            story.append(hist)

        if getattr(mc, "run_details", None):
            pass_count = sum(1 for r in mc.run_details if getattr(r, "passed", False))
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                f"<b>Run outcomes:</b> {pass_count}/{mc.n_runs} runs met the BS 9999 margin criterion.",
                s["body"]
            ))

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

        bottleneck_chart = self._make_bottleneck_chart()
        if bottleneck_chart is not None:
            story.append(Spacer(1, 0.35 * cm))
            story.append(bottleneck_chart)

        if self.recommendations:
            impact_rows = [["Priority", "Expected effect", "Engineering rationale"]]
            for rec in self.recommendations[:6]:
                impact_rows.append([
                    rec.priority,
                    f"−{rec.estimated_rset_reduction_s:.0f}s RSET",
                    rec.description[:120] + ("..." if len(rec.description) > 120 else ""),
                ])
            story.append(Spacer(1, 0.2 * cm))
            story.append(self._coloured_table(impact_rows, [3 * cm, 4.5 * cm, 9.5 * cm]))

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

        glossary = [
            ["Output", "Meaning"],
            ["RSET", "Time required for the occupant population to complete evacuation."],
            ["ASET", "Time until tenability limits are exceeded in the most exposed zone."],
            ["FED", "Fractional effective dose used to estimate incapacitation risk."],
            ["Bottleneck", "A zone or opening with high path centrality or congestion pressure."],
            ["Monte Carlo pass rate", "Share of randomized runs that satisfied the safety margin criterion."],
        ]
        story.append(Spacer(1, 0.35 * cm))
        story.append(Paragraph("Data dictionary", s["h2"]))
        story.append(self._coloured_table(glossary, [5 * cm, 12 * cm]))
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
