"""
FirePBD Engine — Physical & Simulation Constants
=================================================
References:
  - NFPA 72: National Fire Alarm and Signaling Code
  - NFPA 101: Life Safety Code (2021)
  - NFPA 557: Standard for Determination of Fire Loads
  - SFPE Handbook of Fire Protection Engineering, 5th Edition
  - ISO 13571:2012 — Life-threatening components of fire (FED model)
  - BS 9999:2017 — Fire safety in the design of buildings
  - IS 16009:2010 — Fire hazard assessment methodology (India)
"""

# ─── Simulation Resolution ────────────────────────────────────────────────────
SIMULATION_TIMESTEP_S: float = 5.0      # seconds per simulation step
DEFAULT_GRID_CELL_SIZE_M: float = 1.0   # metres represented by one grid cell
MAX_SIMULATION_STEPS: int = 360          # 360 × 5s = 30 min max simulation

# ─── Environmental Baseline ───────────────────────────────────────────────────
AMBIENT_TEMPERATURE_C: float = 20.0     # °C
AMBIENT_OXYGEN_PERCENT: float = 20.9    # % O₂ in fresh air
ATMOSPHERIC_PRESSURE_PA: float = 101325.0

# ─── Heat Release Rate Presets (kW/m²) — NFPA 557 / SFPE Ch. 3-1 ───────────
HRR_PER_M2_KW: dict = {
    "wood":         250,
    "carpet":       150,
    "furniture":    500,
    "office":       250,
    "retail":       400,
    "storage":      600,
    "kitchen":      350,
    "corridor":      80,
    "default":      250,
}

# ─── Fuel Load Density (MJ/m²) — NFPA 557 Table 5.1 ─────────────────────────
FUEL_LOAD_DENSITY_MJ_M2: dict = {
    "residential":  600,
    "office":       420,
    "retail":       600,
    "storage":     1200,
    "corridor":     200,
    "assembly":     160,
    "kitchen":      350,
    "default":      420,
}

# ─── Fire Spread Physics ──────────────────────────────────────────────────────
BASE_SPREAD_PROB: float = 0.25          # baseline ignition prob per step per burning neighbour
OPENING_SPREAD_MULTIPLIER: float = 3.0  # through open doorways — faster spread
WALL_SPREAD_BLOCK: bool = True          # walls block fire spread (set False to disable)
SMOKE_DIFFUSION_COEFF: float = 0.15     # fraction diffusing to neighbours per step
CO_DIFFUSION_COEFF: float = 0.12
HEAT_DECAY_FACTOR: float = 0.97        # ambient heat loss per step

# ─── Temperature Thresholds (°C) ─────────────────────────────────────────────
TEMP_IGNITION_C: float = 300.0          # auto-ignition threshold
TEMP_FLASHOVER_C: float = 600.0         # upper-layer flashover (NFPA 72 Annex B)
TEMP_BURNOUT_C: float = 1100.0          # cell considered fully burned out
TEMP_TENABILITY_MAX_C: float = 60.0     # ISO 13571 human tenability limit

# ─── Combustion Yields (g/g fuel) — SFPE Table 2-6.8, Douglas 1994 ──────────
SMOKE_YIELD_G_G: float = 0.07           # well-ventilated wood combustion
CO_YIELD_G_G: float = 0.004
CO2_YIELD_G_G: float = 1.498

# Fuel burn rate per cell per timestep (fraction of total fuel load)
FUEL_BURN_RATE_PER_STEP: float = 0.05

# ─── Smoke & Visibility ───────────────────────────────────────────────────────
EXTINCTION_COEFF_K: float = 0.08        # m⁻¹ per smoke unit (empirical default)
VISIBILITY_ILLUMINATED_SIGNS_M: float = 10.0  # SFPE min for illuminated exit signs
VISIBILITY_DARK_M: float = 5.0          # minimum in dark conditions

# ─── CO Concentration (ppm) Calculation ──────────────────────────────────────
# CO mass → ppm conversion at STP:  1 g/m³ CO ≈ 800 ppm
CO_G_M3_TO_PPM: float = 800.0

# ─── Fractional Effective Dose (FED) — ISO 13571 ────────────────────────────
# CO incapacitation model (Purser 1988):
#   FED_CO per step = (CO_ppm / 35000) × (dt / 60)  [simplified linear model]
CO_FED_NORMALISER: float = 35000.0      # ppm·min to incapacitation
FED_INCAPACITATION_THRESHOLD: float = 1.0

# Heat FED — table: (temperature_°C, time_to_incapacitation_s)
HEAT_FED_TABLE: list = [
    (60,   7200),   # 2 hours
    (80,   7200),   # still tolerable short-term
    (100,  2700),   # 45 min
    (120,   900),   # 15 min
    (150,   300),   # 5 min
    (200,   120),   # 2 min
    (300,    30),   # 30 s
    (600,     5),   # flashover → near-immediate incapacitation
]

# O₂ depletion — incapacitation below 12%, immediate danger below 8%
O2_INCAPACITATION_PERCENT: float = 12.0
O2_DANGER_PERCENT: float = 8.0
# FED contribution from O₂ depletion (simplified: linear above threshold)
O2_FED_RATE_AT_12: float = 1.0 / 300.0  # 1 FED unit per 5 min at 12% O₂

# ─── Tenability Limits (ISO 13571 / SFPE 3-14) ───────────────────────────────
TENABILITY_CO_MAX_PPM: float = 1400.0   # ~30 min exposure limit
TENABILITY_SMOKE_MAX: float = 100.0     # smoke density index
TENABILITY_VISIBILITY_MIN_M: float = 5.0
TENABILITY_TEMP_MAX_C: float = 60.0    # alias used by fire_agent and report
# Minimum passable opening width (metres)
MIN_OPENING_WIDTH_M: float = 0.6

# ─── Evacuation Physics (SFPE Handbook) ──────────────────────────────────────
# Free-flow speed distribution — SFPE Table 3-13.1
WALK_SPEED_MEAN_M_S: float = 1.2
WALK_SPEED_STD_M_S: float = 0.25
WALK_SPEED_MIN_M_S: float = 0.3
WALK_SPEED_MAX_M_S: float = 1.8

# Fruin (1971) speed-density: v = v_free × max(0, 1 − k×D)
FRUIN_SPEED_K: float = 0.266           # (m/person·s)⁻¹
MAX_CROWD_DENSITY_P_M2: float = 3.8    # practical level-of-service capacity

# Door flow rate — Nelson & MacLennan (SFPE 3-13.19)
DOOR_FLOW_RATE_P_M_S: float = 1.3      # persons per (metre of width × second)

# Body radius for personal space in congestion model
PERSON_BODY_RADIUS_M: float = 0.25

# Smoke-visibility → speed fraction (SFPE 3-13.10)
VISIBILITY_SPEED_CURVE: list = [        # (visibility_m, speed_fraction_of_free)
    (0.0,  0.10),
    (2.0,  0.30),
    (5.0,  0.60),
    (10.0, 1.00),
]

# Reaction time distribution (pre-movement time) — BS 9999 Annex D
T_DETECTION_S: float = 30.0            # automatic detection system delay
T_WARNING_S: float = 10.0              # alarm-to-PA warning broadcast
T_REACTION_MEAN_S: float = 45.0        # occupant reaction time (mean)
T_REACTION_STD_S: float = 15.0         # reaction time std deviation

# ─── RSET / ASET Safety Margins ──────────────────────────────────────────────
ASET_SAFETY_MARGIN_S: float = 120.0    # required: ASET − RSET ≥ 120 s (BS 9999)
RSET_BUFFER_FRACTION: float = 1.25     # RSET × 1.25 for engineering safety factor

# ─── Risk Score Weights ───────────────────────────────────────────────────────
RISK_WEIGHT_RSET_ASET: float = 0.35
RISK_WEIGHT_EVAC_SUCCESS: float = 0.30
RISK_WEIGHT_BOTTLENECK: float = 0.20
RISK_WEIGHT_DEAD_ZONES: float = 0.15

# ─── Occupancy Load Defaults (persons/m²) — NFPA 101 Table 7.3.1.2 ──────────
OCCUPANCY_LOAD_P_M2: dict = {
    "residential":  0.05,    # gross area
    "office":       0.09,
    "retail":       0.56,
    "assembly":     0.65,
    "corridor":     0.10,
    "storage":      0.03,
    "default":      0.09,
}

# ─── Compliance Standards Registry ───────────────────────────────────────────
STANDARDS: dict = {
    "NFPA_101":  "NFPA 101 Life Safety Code (2021 edition)",
    "NFPA_72":   "NFPA 72 National Fire Alarm and Signaling Code (2022)",
    "NFPA_557":  "NFPA 557 Standard for Determination of Fire Loads (2016)",
    "BS_9999":   "BS 9999:2017 Code of practice for fire safety in buildings",
    "IS_16009":  "IS 16009:2010 Fire hazard assessment methodology (India)",
    "ISO_13571": "ISO 13571:2012 Life-threatening components of fire — FED",
    "ISO_13943": "ISO 13943:2017 Fire safety — Vocabulary",
    "SFPE":      "SFPE Handbook of Fire Protection Engineering, 5th Edition (2016)",
}

# ─── Monte Carlo Defaults ─────────────────────────────────────────────────────
MC_DEFAULT_RUNS: int = 500
MC_CONFIDENCE_LEVEL: float = 0.90      # 90th percentile for conservative RSET
MC_RANDOM_SEED: int = 42               # reproducibility (can be overridden)

# ─── Grid State Encoding ──────────────────────────────────────────────────────
CELL_NORMAL: int = 0
CELL_BURNING: int = 1
CELL_BURNED: int = 2
CELL_WALL: int = 3
CELL_OPENING: int = 4                  # door/window gap in wall

# ─── API Configuration ────────────────────────────────────────────────────────
API_MAX_BLUEPRINT_SIZE_MB: int = 20
API_SIMULATION_TIMEOUT_S: int = 300
WS_STREAM_INTERVAL_STEPS: int = 10     # send WebSocket update every N steps
