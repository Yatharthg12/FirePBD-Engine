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
    this.layers = { fire: true, smoke: false, agents: true, heatmap: false };
    this.model = null;      // BuildingModel for zone overlay
    this.lastSnap = null;   // Most recent grid snapshot
    this.agents = [];       // [{x,y,status}, ...]
    this.gridRows = 0;
    this.gridCols = 0;
    this.cellM = 1.0;
    this._imgData = null;
    this.blueprintModel = null;
    this.zoneStatus = {};
    this._bpTransform = null;
  }

  setGridDimensions(rows, cols, cellM) {
    this.gridRows = rows;
    this.gridCols = cols;
    this.cellM = cellM;
  }

  toggleLayer(layer) {
    this.layers[layer] = !this.layers[layer];
  }

  setModel(model) {
    this.blueprintModel = model || null;
    if (this.lastSnap) {
      this.render(this.lastSnap, this.agents);
    }
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
          if (sk > 3) {
            const opacity = Math.min(sk / 180, 0.18);
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
      const status = String(agent.status || 'REACTING').toUpperCase();
      const x = agent.x / this.cellM * cellW + cellW / 2;
      const y = agent.y / this.cellM * cellH + cellH / 2;
      const colour = statusColour[status] || '#60a5fa';
      const radius = status === 'EVACUATED' || status === 'DEAD'
        ? Math.max(cellW * 0.22, 1.8)
        : Math.max(cellW * 0.48, 2.5);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.globalAlpha = status === 'EVACUATED' ? 0.42 : status === 'DEAD' ? 0.28 : 0.78;
      ctx.fill();
      if (status === 'EVACUATED') {
        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.lineWidth = 0.75;
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;
    }
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

    const origin = snap.grid_origin || { x: 0, y: 0 };
    const cellM = snap.grid_size?.cell_m || this.cellM || 1.0;
    const drawCell = (mx, my, fillStyle, alpha = 1.0) => {
      const a = toCanvas(mx - cellM / 2, my - cellM / 2);
      const b = toCanvas(mx + cellM / 2, my + cellM / 2);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fillStyle;
      ctx.fillRect(a.x, a.y, b.x - a.x, b.y - a.y);
      ctx.restore();
    };

    const state = snap.state || [];
    const temp = snap.temperature || [];
    const smoke = snap.smoke || [];
    for (let r = 0; r < state.length; r++) {
      const rowS = state[r] || [];
      const rowT = temp[r] || [];
      const rowSk = smoke[r] || [];
      for (let c = 0; c < rowS.length; c++) {
        const cell = rowS[c];
        const mx = (c + 0.5) * cellM - origin.x;
        const my = (r + 0.5) * cellM - origin.y;
        if (cell === 3) {
          drawCell(mx, my, 'rgba(11,18,34,0.95)', 1.0);
          continue;
        }
        const sk = rowSk[c] || 0;
        if (this.layers.smoke && sk > 5) {
          const smokeAlpha = Math.min(0.01 + sk / 6000, 0.08);
          const p = toCanvas(mx, my);
          const radius = Math.max(8, cellM * scale * 1.4);
          ctx.save();
          ctx.globalCompositeOperation = 'source-over';
          const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius * 2.2);
          grd.addColorStop(0, `rgba(176,185,199,${smokeAlpha})`);
          grd.addColorStop(0.6, `rgba(113,128,150,${smokeAlpha * 0.5})`);
          grd.addColorStop(1, 'rgba(17,24,39,0)');
          ctx.fillStyle = grd;
          ctx.beginPath();
          ctx.arc(p.x, p.y, radius * 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
        if (this.layers.heatmap && rowT[c] > 25) {
          drawCell(mx, my, this._tempToColour(rowT[c]), Math.min(0.12 + rowT[c] / 1200, 0.35));
        }
        if (this.layers.fire && cell === 1) {
          const t = rowT[c] || 300;
          const intensity = Math.min((t - 100) / 900, 1.0);
          drawCell(mx, my, `rgba(255,${Math.round(120 - intensity * 90)},0,1)`, 0.45 + intensity * 0.4);
          ctx.save();
          const p = toCanvas(mx, my);
          const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, cellM * scale * 2.2);
          grd.addColorStop(0, `rgba(255,190,0,0.35)`);
          grd.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = grd;
          ctx.fillRect(p.x - cellM * scale * 2, p.y - cellM * scale * 2, cellM * scale * 4, cellM * scale * 4);
          ctx.restore();
        } else if (cell === 2) {
          drawCell(mx, my, 'rgba(32,20,14,1)', 0.62);
        }
      }
    }

    if (this.blueprintModel.walls?.length) {
      ctx.save();
      ctx.strokeStyle = 'rgba(15,23,42,0.95)';
      ctx.shadowColor = 'rgba(56,189,248,0.18)';
      ctx.shadowBlur = 8;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineWidth = 4.5;
      for (const wall of this.blueprintModel.walls) {
        const a = toCanvas(wall.x1, wall.y1);
        const b = toCanvas(wall.x2, wall.y2);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    ctx.save();
    ctx.strokeStyle = 'rgba(148,163,184,0.04)';
    for (let x = 0; x < cw; x += 28) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
    }
    for (let y = 0; y < ch; y += 28) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
    }
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
      ctx.fillStyle = zone.is_exit ? 'rgba(56,189,248,0.03)' : 'rgba(6,10,19,0.02)';
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = zone.is_exit ? 2.4 : 1.2;
      ctx.stroke();

      const label = toCanvas(zone.centroid.x, zone.centroid.y);
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(zone.id, label.x, label.y);

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
      const statusColour = {
        MOVING: '#22c55e',
        WAITING: '#f59e0b',
        REACTING: '#60a5fa',
        EVACUATED: '#94a3b8',
        INCAPACITATED: '#f97316',
        DEAD: '#6b7280',
      };
      ctx.save();
      ctx.globalAlpha = 0.88;
      for (const agent of agents) {
        const status = String(agent.status || 'REACTING').toUpperCase();
        const p = toCanvas(agent.x, agent.y);
        ctx.beginPath();
        const radius = status === 'EVACUATED' || status === 'DEAD'
          ? Math.max(2.0, cellM * scale * 0.12)
          : Math.max(3.5, cellM * scale * 0.18);
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = statusColour[status] || '#60a5fa';
        ctx.globalAlpha = status === 'EVACUATED' ? 0.4 : status === 'DEAD' ? 0.25 : 0.9;
        ctx.fill();
        if (status === 'EVACUATED') {
          ctx.strokeStyle = 'rgba(255,255,255,0.18)';
          ctx.lineWidth = 0.75;
          ctx.stroke();
        }
      }
      ctx.restore();
    }
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
