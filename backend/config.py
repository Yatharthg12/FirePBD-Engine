"""
FirePBD Engine — Central Configuration
=======================================
All runtime configuration, paths, and feature flags live here.
Override via environment variables (loaded from .env).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.resolve()
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATASETS_DIR = ROOT_DIR / "datasets"
DATA_DIR = BACKEND_DIR / "data"
AI_MODELS_DIR = BACKEND_DIR / "ai_models"
DOCS_DIR = ROOT_DIR / "docs"

# ─── Data Directories ─────────────────────────────────────────────────────────
INPUT_BLUEPRINTS_DIR = DATA_DIR / "input_blueprints"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SIMULATION_CACHE_DIR = OUTPUTS_DIR / "simulations"

# Ensure output directories exist
for _d in [INPUT_BLUEPRINTS_DIR, PROCESSED_DIR, OUTPUTS_DIR, REPORTS_DIR, SIMULATION_CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Dataset Paths ────────────────────────────────────────────────────────────
CUBICASA_DIR = DATASETS_DIR / "CubiCasa5K" / "cubicasa5k"
CUBICASA_HIGH_QUALITY = CUBICASA_DIR / "high_quality"
CUBICASA_TRAIN_TXT = CUBICASA_DIR / "train.txt"
CUBICASA_VAL_TXT = CUBICASA_DIR / "val.txt"
CUBICASA_TEST_TXT = CUBICASA_DIR / "test.txt"

# ─── AI Model Paths ───────────────────────────────────────────────────────────
DOOR_DETECTOR_MODEL = AI_MODELS_DIR / "door_detector.pt"
SEGMENTATION_MODEL = AI_MODELS_DIR / "segmentation_model.pt"

# ─── API Settings ─────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
API_CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
API_MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))

# ─── Simulation Defaults ──────────────────────────────────────────────────────
DEFAULT_GRID_CELL_SIZE_M: float = float(os.getenv("GRID_CELL_SIZE", "1.0"))
DEFAULT_SIMULATION_STEPS: int = int(os.getenv("SIM_STEPS", "240"))  # 240×5s=20min
DEFAULT_MONTE_CARLO_RUNS: int = int(os.getenv("MC_RUNS", "500"))
MAX_WORKER_PROCESSES: int = int(os.getenv("MAX_WORKERS", "4"))

# ─── Blueprint Processing ─────────────────────────────────────────────────────
# Minimum zone area to keep (m²) — filter tiny artefacts
MIN_ZONE_AREA_M2: float = 2.0
# Contour approximation epsilon (fraction of arc length) for polygon simplification
CONTOUR_APPROX_EPSILON: float = 0.02
# Maximum gap in wall to classify as door/window (cells)
DOOR_GAP_MAX_CELLS: int = 8
DOOR_GAP_MIN_CELLS: int = 2
# Minimum opening width (m) to consider a passable connection
MIN_OPENING_WIDTH_M: float = 0.6

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ─── Report Settings ─────────────────────────────────────────────────────────
REPORT_COMPANY_NAME: str = os.getenv("COMPANY_NAME", "FirePBD Engine")
REPORT_LOGO_PATH: str = os.getenv("REPORT_LOGO", "")
REPORT_PAGE_SIZE: str = "A4"

# ─── Feature Flags ────────────────────────────────────────────────────────────
ENABLE_MONTE_CARLO: bool = os.getenv("ENABLE_MC", "true").lower() == "true"
ENABLE_OPTIMIZATION: bool = os.getenv("ENABLE_OPT", "true").lower() == "true"
ENABLE_PDF_REPORT: bool = os.getenv("ENABLE_PDF", "true").lower() == "true"
USE_SVG_PARSER: bool = True             # Prefer SVG over OpenCV for CubiCasa
USE_VECTORISED_FIRE: bool = True        # NumPy vectorised vs Python loop

def print_config() -> None:
    """Print active configuration to stdout — useful for debugging."""
    print("=" * 60)
    print("FirePBD Engine — Active Configuration")
    print("=" * 60)
    print(f"  ROOT_DIR:         {ROOT_DIR}")
    print(f"  API:              {API_HOST}:{API_PORT}")
    print(f"  GRID CELL SIZE:   {DEFAULT_GRID_CELL_SIZE_M}m")
    print(f"  SIM STEPS:        {DEFAULT_SIMULATION_STEPS}")
    print(f"  MC RUNS:          {DEFAULT_MONTE_CARLO_RUNS}")
    print(f"  LOG LEVEL:        {LOG_LEVEL}")
    print("=" * 60)
