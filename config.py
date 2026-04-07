

import os
from pathlib import Path

class BaseConfig:
    # App
    APP_NAME = "AI Traffic Congestion Analysis"
    VERSION = "1.0.0"
    DEBUG = False

    # Fixed Colab tree
    BASE_DIR = BASE_DIR = Path(os.getenv("APP_DIR", Path(__file__).resolve().parent))
    UPLOAD_FOLDER  = BASE_DIR / "uploads"
    OUTPUT_FOLDER  = BASE_DIR / "outputs"
    FRONTEND_FOLDER = BASE_DIR / "frontend"
    LOGS_FOLDER    = BASE_DIR / "logs"

    # Server
    HOST = "0.0.0.0"
    PORT = 5000
    WORKERS = 1
    THREADS = 8
    TIMEOUT = 120

    # Uploads
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".flv"}

    # Models (defaults)
    DEFAULT_DETECT_MODEL = "yolov8n.pt"
    DEFAULT_SEG_MODEL = None
    DETECT_CONFIDENCE = 0.25
    DETECT_IOU = 0.5
    SEG_CONFIDENCE = 0.15
    SEG_IMAGE_SIZE = 960


class TrafficConfig:
    """Traffic analysis specific configuration"""
    TIME_WINDOW_SECONDS = 15.0
    MAX_LANES = 4
    MIN_LANE_AREA_PIXELS = 8000

    LANE_CAPACITY_VPH = 1800.0
    COUNTING_LINE_POSITION = 0.62
    COUNTING_LINE_WIDTH = 0.90

    LANE_LINE_DILATION_PIXELS = 10
    CONNECTED_COMPONENT_MIN_AREA = 5000

    CONGESTION_THRESHOLDS = {
        "yellow": {"occupancy": 0.25, "volume_capacity": 0.70, "crosswalk_blocking": 0.10},
        "red":    {"occupancy": 0.45, "volume_capacity": 0.95, "crosswalk_blocking": 0.30},
    }

    VEHICLE_CLASS_MAPPING = {
        "car":        ["car","auto","automobile","sedan","suv","hatchback"],
        "bus":        ["bus","coach"],
        "truck":      ["truck","lorry","semi","trailer","freight"],
        "motorcycle": ["motorcycle","motorbike","bike","scooter","moped"],
    }


class RuntimeConfig:
    """Runtime configuration with env overrides"""
    def __init__(self):
        self.base = BaseConfig()
        self._load_env_overrides()

    def _load_env_overrides(self):
        # Dirs
        self.app_dir = Path(os.getenv("APP_DIR", str(self.base.BASE_DIR)))
        self.upload_dir = self.app_dir / "uploads"
        self.output_dir = self.app_dir / "outputs"
        self.frontend_dir = self.app_dir / "frontend"

        # Models
        self.detect_weights = os.getenv("DETECT_WEIGHTS", "")
        self.seg_weights = os.getenv("SEG_WEIGHTS", "")

        # Server
        self.host = os.getenv("HOST", self.base.HOST)
        self.port = int(os.getenv("PORT", self.base.PORT))
        self.workers = int(os.getenv("WORKERS", self.base.WORKERS))
        self.threads = int(os.getenv("THREADS", self.base.THREADS))

        # Analysis
        self.time_window = float(os.getenv("TIME_WINDOW_SECONDS", TrafficConfig.TIME_WINDOW_SECONDS))
        self.max_lanes = int(os.getenv("MAX_LANES", TrafficConfig.MAX_LANES))
        self.lane_capacity = float(os.getenv("LANE_CAPACITY_VPH", TrafficConfig.LANE_CAPACITY_VPH))

        # Debug flags
        self.debug = os.getenv("DEBUG", "false").lower() in ("true","1","yes")
        self.draw_windows = False  # no GUI in Colab

    def init_app_directories(self):
        for d in (self.upload_dir, self.output_dir, self.frontend_dir, self.base.LOGS_FOLDER):
            Path(d).mkdir(parents=True, exist_ok=True)

    def get_model_paths(self):
        det = self.detect_weights if (self.detect_weights and os.path.exists(self.detect_weights)) else self.base.DEFAULT_DETECT_MODEL
        seg = self.seg_weights if (self.seg_weights and os.path.exists(self.seg_weights)) else self.base.DEFAULT_SEG_MODEL
        return {"detect": det, "segment": seg}

    def print_config_summary(self):
        print("=" * 60)
        print(f" {self.base.APP_NAME} v{self.base.VERSION}")
        print("=" * 60)
        print(f" App Directory: {self.app_dir}")
        print(f" Upload Directory: {self.upload_dir}")
        print(f" Output Directory: {self.output_dir}")
        print(f" Server: {self.host}:{self.port}")
        print(f" Workers: {self.workers}, Threads: {self.threads}")
        print(f" Detection Model: {self.detect_weights or 'Default YOLO'}")
        print(f" Segmentation Model: {self.seg_weights or 'Disabled'}")
        print(f" Time Window: {self.time_window}s | Max Lanes: {self.max_lanes} | Lane Capacity: {self.lane_capacity} vph")
        print("=" * 60)

# Global instance
config = RuntimeConfig()


