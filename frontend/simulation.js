/**
 * FirePBD Engine — Grid Simulation Renderer
 * Renders the fire simulation grid on the main canvas.
 *
 * Layers (toggleable):
 *   fire   — burning cells (orange glow)
 *   smoke  — smoke density (dark grey opacity)
 *   agents — agent positions (coloured by status)
 *   heatmap — temperature colour map (blue→red)
 *
 * Uses direct pixel manipulation via ImageData for performance.
 */

class SimulationRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.layers = { fire: true, smoke: true, agents: true, heatmap: false };
    this.model = null;      // BuildingModel for zone overlay
    this.lastSnap = null;   // Most recent grid snapshot
    this.agents = [];       // [{x,y,status}, ...]
    this.gridRows = 0;
    this.gridCols = 0;
    this.cellM = 1.0;
    this._imgData = null;
    this.routePoints = [];
    this.routeSteps = 0;
    this.blueprintModel = null;
    this.zoneStatus = {};
    this._bpTransform = null;
    this.sourceImage = null;
  }

  setGridDimensions(rows, cols, cellM) {
    this.gridRows = rows;
    this.gridCols = cols;
    this.cellM = cellM;
  }

  toggleLayer(layer) {
    this.layers[layer] = !this.layers[layer];
  }

  setRoute(points, steps = 0) {
    this.routePoints = Array.isArray(points) ? points : [];
    this.routeSteps = steps || 0;
    if (this.lastSnap) {
      this.render(this.lastSnap, this.agents);
    }
  }

  setModel(model) {
    this.blueprintModel = model || null;
    if (this.lastSnap) {
      this.render(this.lastSnap, this.agents);
    }
  }

  setSourceImage(url) {
    if (!url) return;
    const img = new Image();
    img.onload = () => {
      this.sourceImage = img;
      if (this.lastSnap) {
        this.render(this.lastSnap, this.agents);
      }
    };
    img.src = url;
  }

  setZoneStatus(zoneStatus) {
    this.zoneStatus = zoneStatus || {};
    if (this.lastSnap) {
      this.render(this.lastSnap, this.agents);
    }
  }

  canvasToWorld(x, y) {
    if (!this._bpTransform) return { x, y };
    return {
      x: (x - this._bpTransform.offsetX) / this._bpTransform.scale,
      y: (y - this._bpTransform.offsetY) / this._bpTransform.scale,
    };
  }

  worldToCanvas(x, y) {
    if (!this._bpTransform) return { x, y };
    return {
      x: x * this._bpTransform.scale + this._bpTransform.offsetX,
      y: y * this._bpTransform.scale + this._bpTransform.offsetY,
    };
  }

  render(snap, agents) {
    if (!snap) return;
    this.lastSnap = snap;
    if (agents) this.agents = agents;

    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const ctx = this.ctx;

    if (!this.gridRows || !this.gridCols) {
      ctx.fillStyle = '#050816';
      ctx.fillRect(0, 0, cw, ch);
      return;
    }

    const cellW = cw / this.gridCols;
    const cellH = ch / this.gridRows;

    // Clear
    ctx.fillStyle = '#050816';
    ctx.fillRect(0, 0, cw, ch);

    const state = snap.state;
    const temp = snap.temperature;
    const smoke = snap.smoke;
    const vis = snap.visibility;

    if (this.blueprintModel?.zones?.length) {
      this._renderBlueprint(snap, agents || this.agents);
      return;
    }

    // Draw cells
    for (let r = 0; r < this.gridRows; r++) {
      const row_s = state[r];
      const row_t = temp ? temp[r] : null;
      const row_sk = smoke ? smoke[r] : null;

      for (let c = 0; c < this.gridCols; c++) {
        const s = row_s[c];
        const x = c * cellW;
        const y = r * cellH;

        // Wall
        if (s === 3) {
          ctx.fillStyle = '#1e293b';
          ctx.fillRect(x, y, cellW, cellH);
          continue;
        }

        // Opening
        if (s === 4) {
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(x, y, cellW, cellH);
          continue;
        }

        // Normal cell smoke
        if (this.layers.smoke && row_sk) {
          const sk = row_sk[c];
          if (sk > 0.1) {
            const opacity = Math.min(sk / 80, 0.85);
            ctx.fillStyle = `rgba(50,50,60,${opacity.toFixed(2)})`;
            ctx.fillRect(x, y, cellW, cellH);
          }
        }

        // Temperature heatmap
        if (this.layers.heatmap && row_t) {
          const t = row_t[c];
          if (t > 25) {
            const col = this._tempToColour(t);
            ctx.fillStyle = col;
            ctx.fillRect(x, y, cellW, cellH);
          }
        }

        // BURNING (state==1)
        if (this.layers.fire && s === 1) {
          const t = row_t ? row_t[c] : 300;
          const intensity = Math.min((t - 100) / 900, 1.0);
          const r_val = Math.round(255);
          const g_val = Math.round(100 - intensity * 80);
          const b_val = 0;
          const alpha = 0.6 + intensity * 0.4;
          ctx.fillStyle = `rgba(${r_val},${g_val},${b_val},${alpha.toFixed(2)})`;
          ctx.fillRect(x, y, cellW, cellH);

          // Glow
          const grd = ctx.createRadialGradient(x + cellW/2, y + cellH/2, 0, x + cellW/2, y + cellH/2, cellW * 2);
          grd.addColorStop(0, `rgba(255,${Math.round(g_val)},0,0.3)`);
          grd.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = grd;
          ctx.fillRect(x - cellW, y - cellH, cellW * 3, cellH * 3);
        }

        // BURNED (state==2)
        if (s === 2) {
          ctx.fillStyle = 'rgba(40,20,10,0.8)';
          ctx.fillRect(x, y, cellW, cellH);
        }
      }
    }

    // Draw agents
    if (this.layers.agents && this.agents.length) {
      this._drawAgents(cellW, cellH);
    }

    if (this.routePoints.length > 1) {
      this._drawRoute(cellW, cellH, snap.step || 0);
    }

    // Draw grid overlay (thin lines)
    if (cellW > 4 && cellH > 4) {
      ctx.strokeStyle = 'rgba(255,255,255,0.03)';
      ctx.lineWidth = 0.5;
      for (let r = 0; r <= this.gridRows; r += 5) {
        ctx.beginPath(); ctx.moveTo(0, r * cellH); ctx.lineTo(cw, r * cellH); ctx.stroke();
      }
      for (let c = 0; c <= this.gridCols; c += 5) {
        ctx.beginPath(); ctx.moveTo(c * cellW, 0); ctx.lineTo(c * cellW, ch); ctx.stroke();
      }
    }
  }

  _drawAgents(cellW, cellH) {
    const ctx = this.ctx;
    const statusColour = {
      MOVING: '#22c55e',
      WAITING: '#f59e0b',
      REACTING: '#60a5fa',
      EVACUATED: '#94a3b8',
      INCAPACITATED: '#f97316',
      DEAD: '#6b7280',
    };
    for (const agent of this.agents) {
      const x = agent.x / this.cellM * cellW + cellW / 2;
      const y = agent.y / this.cellM * cellH + cellH / 2;
      const colour = statusColour[agent.status] || '#60a5fa';
      ctx.beginPath();
      ctx.arc(x, y, Math.max(cellW * 0.6, 3), 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.globalAlpha = 0.85;
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }
  }

  _drawRoute(cellW, cellH, step) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = '#7dd3fc';
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 8;
    ctx.lineWidth = 3.2;
    ctx.setLineDash([10, 7]);
    ctx.beginPath();
    this.routePoints.forEach((pt, idx) => {
      const x = pt.x / this.cellM * cellW + cellW / 2;
      const y = pt.y / this.cellM * cellH + cellH / 2;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    const progress = this.routeSteps > 0 ? Math.min(step / this.routeSteps, 1) : 0;
    const marker = this._routePointAt(progress, cellW, cellH);
    if (marker) {
      ctx.fillStyle = '#f8fafc';
      ctx.beginPath();
      ctx.arc(marker.x, marker.y, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
  }

  _renderBlueprint(snap, agents) {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    ctx.fillStyle = '#050816';
    ctx.fillRect(0, 0, cw, ch);

    const bb = this.blueprintModel.bounding_box;
    const w = bb.max_x - bb.min_x || 1;
    const h = bb.max_y - bb.min_y || 1;
    const pad = 18;
    const scale = Math.min((cw - pad * 2) / w, (ch - pad * 2) / h);
    const offsetX = pad + ((cw - pad * 2) - w * scale) / 2 - bb.min_x * scale;
    const offsetY = pad + ((ch - pad * 2) - h * scale) / 2 - bb.min_y * scale;
    const toCanvas = (x, y) => ({ x: x * scale + offsetX, y: y * scale + offsetY });
    this._bpTransform = { scale, offsetX, offsetY };

    if (this.sourceImage) {
      ctx.save();
      ctx.globalAlpha = 0.28;
      ctx.drawImage(this.sourceImage, 0, 0, cw, ch);
      ctx.restore();
    }

    ctx.save();
    ctx.strokeStyle = 'rgba(56,189,248,0.08)';
    for (let x = 0; x < cw; x += 24) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
    }
    for (let y = 0; y < ch; y += 24) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(56,189,248,0.22)';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(10, 10, cw - 20, ch - 20);
    ctx.restore();

    const zoneMap = new Map((this.blueprintModel.zones || []).map(z => [z.id, z]));
    for (const zone of this.blueprintModel.zones) {
      const pts = zone.polygon;
      if (!pts || pts.length < 2) continue;
      const status = this.zoneStatus[zone.id] || {};
      const danger = (status.danger || (zone.is_exit ? 'exit' : 'LOW')).toString().toUpperCase();
      const color = danger === 'EXIT' ? '#38bdf8' :
        danger === 'HIGH' ? '#fb7185' :
        danger === 'MEDIUM' ? '#f59e0b' :
        danger === 'UNTENABLE' ? '#ef4444' :
        '#60a5fa';

      ctx.beginPath();
      const first = toCanvas(pts[0].x, pts[0].y);
      ctx.moveTo(first.x, first.y);
      for (let i = 1; i < pts.length; i++) {
        const p = toCanvas(pts[i].x, pts[i].y);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      const haze = this.layers.smoke && status.danger && danger !== 'LOW' && danger !== 'EXIT'
        ? 'rgba(148,163,184,0.08)'
        : zone.is_exit ? 'rgba(56,189,248,0.05)' : 'rgba(6,10,19,0.22)';
      ctx.fillStyle = haze;
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = zone.is_exit ? 2.4 : 1.2;
      ctx.stroke();

      const label = toCanvas(zone.centroid.x, zone.centroid.y);
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(zone.id, label.x, label.y);

      if (this.layers.fire && status.danger && danger !== 'LOW' && danger !== 'EXIT') {
        ctx.strokeStyle = 'rgba(249,115,22,0.28)';
        ctx.lineWidth = 5;
        ctx.stroke();
      }

      if (this.layers.heatmap && status.avg_temp_c != null) {
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText(`${Math.round(status.avg_temp_c)}°`, label.x, label.y + 12);
      }
    }

    for (const opening of this.blueprintModel.openings || []) {
      if (!opening.midpoint) continue;
      const p = toCanvas(opening.midpoint.x, opening.midpoint.y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, opening.is_exit_door ? 5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = opening.is_exit_door ? '#38bdf8' : '#94a3b8';
      ctx.fill();
    }

    if (this.layers.agents && agents?.length) {
      ctx.fillStyle = '#22c55e';
      for (const agent of agents) {
        const zone = zoneMap.get(agent.zone);
        const p = zone ? toCanvas(zone.centroid.x, zone.centroid.y) : toCanvas(agent.x, agent.y);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (this.routePoints.length > 1) {
      ctx.save();
      ctx.strokeStyle = '#7dd3fc';
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 8;
      ctx.lineWidth = 3.2;
      ctx.setLineDash([9, 6]);
      ctx.beginPath();
      this.routePoints.forEach((pt, idx) => {
        const p = toCanvas(pt.x, pt.y);
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      ctx.restore();

      const marker = this.routePoints[0];
      const p = toCanvas(marker.x, marker.y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6.5, 0, Math.PI * 2);
      ctx.fillStyle = '#e0f2fe';
      ctx.fill();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  _routePointAt(progress, cellW, cellH) {
    if (!this.routePoints.length) return null;
    const pts = this.routePoints.map(pt => ({
      x: pt.x / this.cellM * cellW + cellW / 2,
      y: pt.y / this.cellM * cellH + cellH / 2,
    }));
    const total = pts.length - 1;
    if (total <= 0) return pts[0];
    const idx = Math.min(Math.floor(progress * total), total - 1);
    const local = (progress * total) - idx;
    const a = pts[idx];
    const b = pts[idx + 1];
    return {
      x: a.x + (b.x - a.x) * local,
      y: a.y + (b.y - a.y) * local,
    };
  }

  _tempToColour(temp) {
    // Blue → Cyan → Green → Yellow → Orange → Red
    const t = Math.min(Math.max(temp - 25, 0) / 975, 1.0);
    const r = Math.round(t < 0.5 ? 0 : (t - 0.5) * 2 * 255);
    const g = Math.round(t < 0.5 ? t * 2 * 255 : (1 - (t - 0.5) * 2) * 255);
    const b = Math.round(t < 0.25 ? 255 : (0.5 - t) * 4 * 255);
    return `rgba(${r},${g},${Math.max(0, b)},0.5)`;
  }

  showPlaceholder() {
    document.getElementById('simPlaceholder').style.display = 'flex';
  }

  hidePlaceholder() {
    document.getElementById('simPlaceholder').style.display = 'none';
  }
}

window.simRenderer = null;

function initSimRenderer() {
  const canvas = document.getElementById('simCanvas');
  if (!canvas) return;
  function resize() {
    const wrapper = canvas.parentElement;
    canvas.width = wrapper.clientWidth;
    canvas.height = wrapper.clientHeight;
    if (window.simRenderer && window.simRenderer.lastSnap) {
      window.simRenderer.render(window.simRenderer.lastSnap, window.simRenderer.agents);
    }
  }
  window.simRenderer = new SimulationRenderer('simCanvas');
  resize();
  window.addEventListener('resize', resize);
}

document.addEventListener('DOMContentLoaded', initSimRenderer);
