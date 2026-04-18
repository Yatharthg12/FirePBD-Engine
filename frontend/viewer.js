/**
 * FirePBD Engine — Blueprint Viewer
 * Renders extracted building model on the blueprint canvas:
 *   - Zone polygons (colour-coded by danger)
 *   - Exit markers
 *   - Opening markers (doors)
 *   - Zone ID labels
 * Scales to fit the container dynamically.
 */

const ZONE_COLOURS = {
  LOW:        { fill: 'rgba(96,165,250,0.08)',  stroke: '#7dd3fc' },
  MEDIUM:     { fill: 'rgba(56,189,248,0.10)', stroke: '#38bdf8' },
  HIGH:       { fill: 'rgba(251,191,36,0.08)',  stroke: '#fbbf24' },
  UNTENABLE:  { fill: 'rgba(248,113,113,0.10)',  stroke: '#fb7185' },
  exit:       { fill: 'rgba(56,189,248,0.10)', stroke: '#38bdf8' },
  default:    { fill: 'rgba(148,163,184,0.06)', stroke: '#94a3b8' },
};

class BlueprintViewer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.model = null;
    this.zoneDangerMap = {};  // zone_id → danger level
    this.selectedZoneId = null;
    this.routePoints = [];
    this.markerPoint = null;
    this.onZoneSelected = null;
    this.onPointSelected = null;
    this.scale = 1;
    this.offset = { x: 0, y: 0 };
    this._dragging = false;
    this.sourceImage = null;
  }

  setModel(model) {
    this.model = model;
    this._computeTransform();
    this.render();
  }

  setSourceImage(url) {
    if (!url) return;
    const img = new Image();
    img.onload = () => {
      this.sourceImage = img;
      this.render();
    };
    img.src = url;
  }

  setSelection(zoneId, routePoints = []) {
    this.selectedZoneId = zoneId;
    this.routePoints = Array.isArray(routePoints) ? routePoints : [];
    this.render();
  }

  setMarker(point) {
    this.markerPoint = point ? { x: point.x, y: point.y } : null;
    this.render();
  }

  updateDanger(zoneStatus) {
    this.zoneDangerMap = {};
    for (const [zid, status] of Object.entries(zoneStatus)) {
      this.zoneDangerMap[zid] = status.danger || 'LOW';
    }
    this.render();
  }

  _computeTransform() {
    if (!this.model) return;
    const bb = this.model.bounding_box;
    const w = bb.max_x - bb.min_x;
    const h = bb.max_y - bb.min_y;
    const pad = 20;
    const cw = this.canvas.width - pad * 2;
    const ch = this.canvas.height - pad * 2;
    this.scale = Math.min(cw / Math.max(w, 0.01), ch / Math.max(h, 0.01));
    this.offset = {
      x: pad + (cw - w * this.scale) / 2 - bb.min_x * this.scale,
      y: pad + (ch - h * this.scale) / 2 - bb.min_y * this.scale,
    };
  }

  worldToCanvas(x, y) {
    return {
      x: x * this.scale + this.offset.x,
      y: y * this.scale + this.offset.y,
    };
  }

  canvasToWorld(x, y) {
    return {
      x: (x - this.offset.x) / this.scale,
      y: (y - this.offset.y) / this.scale,
    };
  }

  render() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = '#050816';
    ctx.fillRect(0, 0, cw, ch);

    if (!this.model) return;

    if (this.sourceImage) {
      ctx.save();
      ctx.globalAlpha = 0.35;
      ctx.drawImage(this.sourceImage, 0, 0, cw, ch);
      ctx.restore();
    }

    // Draw zones
    for (const zone of this.model.zones) {
      this._drawZone(zone);
    }

    // Draw openings (doors)
    for (const opening of this.model.openings) {
      this._drawOpening(opening);
    }

    this._drawFrame();

    if (this.routePoints.length > 1) {
      this._drawRoute();
    }

    // Draw exit badges
    for (const zone of this.model.zones) {
      if (zone.is_exit) this._drawExitBadge(zone);
    }

    if (this.selectedZoneId) {
      const zone = this.model.zones.find(z => z.id === this.selectedZoneId);
      if (zone) this._drawSelectedMarker(zone);
    }

    if (this.markerPoint) {
      this._drawMarker(this.markerPoint);
    }
  }

  _drawZone(zone) {
    const ctx = this.ctx;
    const pts = zone.polygon;
    if (!pts || pts.length < 2) return;

    const danger = this.zoneDangerMap[zone.id] || (zone.is_exit ? 'exit' : 'default');
    const colours = ZONE_COLOURS[danger] || ZONE_COLOURS.default;

    ctx.beginPath();
    const first = this.worldToCanvas(pts[0].x, pts[0].y);
    ctx.moveTo(first.x, first.y);
    for (let i = 1; i < pts.length; i++) {
      const p = this.worldToCanvas(pts[i].x, pts[i].y);
      ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fillStyle = zone.is_exit ? 'rgba(56,189,248,0.03)' : 'rgba(7,12,20,0.03)';
    ctx.fill();
    ctx.strokeStyle = colours.stroke;
    ctx.lineWidth = zone.is_exit ? 2.6 : 1.4;
    ctx.stroke();

    // Zone label
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    const lp = this.worldToCanvas(cx, cy);
    ctx.fillStyle = zone.is_exit ? '#dbeafe' : '#cbd5e1';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(zone.id, lp.x, lp.y);
  }

  _drawOpening(opening) {
    const ctx = this.ctx;
    if (!opening.midpoint) return;
    const p = this.worldToCanvas(opening.midpoint.x, opening.midpoint.y);
    ctx.beginPath();
    ctx.arc(p.x, p.y, opening.is_exit_door ? 5.5 : 4, 0, Math.PI * 2);
    ctx.fillStyle = opening.is_exit_door ? '#38bdf8' : '#f97316';
    ctx.fill();
  }

  _drawExitBadge(zone) {
    const ctx = this.ctx;
    const pts = zone.polygon;
    if (!pts || pts.length === 0) return;
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    const p = this.worldToCanvas(cx, cy);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('EXIT', p.x, p.y - 8);
  }

  _drawRoute() {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = '#7dd3fc';
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 8;
    ctx.setLineDash([10, 8]);
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    this.routePoints.forEach((pt, idx) => {
      const p = this.worldToCanvas(pt.x, pt.y);
      if (idx === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#e0f2fe';
    const head = this.worldToCanvas(this.routePoints[0].x, this.routePoints[0].y);
    ctx.beginPath();
    ctx.arc(head.x, head.y, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  _drawSelectedMarker(zone) {
    const ctx = this.ctx;
    const pts = zone.polygon;
    if (!pts || pts.length === 0) return;
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    const p = this.worldToCanvas(cx, cy);
    ctx.save();
    ctx.strokeStyle = '#93c5fd';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = 'rgba(96,165,250,0.2)';
    ctx.fill();
    ctx.restore();
  }

  _drawMarker(point) {
    const ctx = this.ctx;
    const p = this.worldToCanvas(point.x, point.y);
    ctx.save();
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 18;
    ctx.fillStyle = '#e0f2fe';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  _drawFrame() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    ctx.save();
    ctx.strokeStyle = 'rgba(56,189,248,0.28)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(10, 10, cw - 20, ch - 20);
    ctx.strokeStyle = 'rgba(56,189,248,0.08)';
    for (let x = 10; x < cw - 10; x += 28) {
      ctx.beginPath();
      ctx.moveTo(x, 10);
      ctx.lineTo(x, ch - 10);
      ctx.stroke();
    }
    for (let y = 10; y < ch - 10; y += 28) {
      ctx.beginPath();
      ctx.moveTo(10, y);
      ctx.lineTo(cw - 10, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  hitTestZone(worldX, worldY) {
    if (!this.model) return null;
    for (const zone of this.model.zones) {
      if (BlueprintViewer.pointInPolygon({ x: worldX, y: worldY }, zone.polygon)) {
        return zone;
      }
    }
    return null;
  }

  static pointInPolygon(point, polygon) {
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

  nearestZone(point) {
    if (!this.model) return null;
    let best = null;
    let bestDist = Infinity;
    for (const zone of this.model.zones) {
      const cx = zone.centroid.x;
      const cy = zone.centroid.y;
      const dx = cx - point.x;
      const dy = cy - point.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < bestDist) {
        bestDist = dist;
        best = zone;
      }
    }
    return best;
  }
}

// Export global instance
window.blueprintViewer = null;

function initBlueprintViewer() {
  const canvas = document.getElementById('blueprintCanvas');
  if (!canvas) return;
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = 200;
  window.blueprintViewer = new BlueprintViewer('blueprintCanvas');
  const pickPoint = (e) => {
    if (!window.blueprintViewer?.model) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const world = window.blueprintViewer.canvasToWorld(x, y);
    const zone = window.blueprintViewer.hitTestZone(world.x, world.y) || window.blueprintViewer.nearestZone(world);
    window.blueprintViewer.setMarker(world);
    if (zone && window.blueprintViewer.onZoneSelected) {
      window.blueprintViewer.onZoneSelected(zone.id, world);
    }
    if (window.blueprintViewer.onPointSelected) {
      window.blueprintViewer.onPointSelected(world, zone);
    }
  };

  canvas.addEventListener('pointerdown', (e) => {
    window.blueprintViewer._dragging = true;
    canvas.setPointerCapture(e.pointerId);
    pickPoint(e);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!window.blueprintViewer._dragging) return;
    pickPoint(e);
  });
  const stopDrag = () => { window.blueprintViewer._dragging = false; };
  canvas.addEventListener('pointerup', stopDrag);
  canvas.addEventListener('pointerleave', stopDrag);
}

document.addEventListener('DOMContentLoaded', initBlueprintViewer);
