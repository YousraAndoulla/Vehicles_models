"""
Utility functions for Traffic Congestion Analysis System
Testing, validation, debugging, and helper functions
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Union

import requests
import cv2
import numpy as np


def check_system_requirements() -> bool:
    """Check if system meets requirements for traffic analysis."""
    print("Checking system requirements...")
    req = {
        "python_version": sys.version_info >= (3, 8),
        "opencv_available": True,
        "torch_available": True,
        "ultralytics_available": True,
    }

    # OpenCV
    try:
        import cv2 as _cv2  # noqa: F401
        req["opencv_version"] = cv2.__version__
    except Exception:
        req["opencv_available"] = False
        req["opencv_version"] = "missing"

    # PyTorch
    try:
        import torch  # noqa: F401
        req["torch_version"] = torch.__version__
        req["cuda_available"] = torch.cuda.is_available()
    except Exception:
        req["torch_available"] = False
        req["torch_version"] = "missing"
        req["cuda_available"] = False

    # Ultralytics
    try:
        from ultralytics import YOLO  # noqa: F401
        req["ultralytics_available"] = True
    except Exception:
        req["ultralytics_available"] = False

    # Print summary
    print("\nSystem Requirements:")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  OpenCV:  {req.get('opencv_version', 'missing')}")
    print(f"  PyTorch: {req.get('torch_version', 'missing')}")
    if req.get("cuda_available"):
        print("  CUDA:    available")
    else:
        print("  CUDA:    not available")
    print(f"  Ultralytics: {'ok' if req['ultralytics_available'] else 'missing'}")

    return all(
        (
            req["python_version"],
            req["opencv_available"],
            req["torch_available"],
            req["ultralytics_available"],
        )
    )


def get_file_info(filepath: Union[str, Path]) -> Dict:
    """Get basic file information."""
    p = Path(filepath)
    if not p.exists():
        return {"exists": False}
    s = p.stat()
    return {
        "exists": True,
        "name": p.name,
        "size": s.st_size,
        "size_mb": s.st_size / (1024 * 1024),
        "modified": time.ctime(s.st_mtime),
        "extension": p.suffix.lower(),
    }


def create_test_image(width: int = 640, height: int = 480, save_path: str | None = None) -> str:
    """Create a simple synthetic image for quick testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # colorful rectangles to mimic vehicles
    cv2.rectangle(img, (100, 200), (180, 280), (0, 255, 0), -1)
    cv2.rectangle(img, (300, 180), (420, 300), (255, 0, 0), -1)
    cv2.rectangle(img, (500, 150), (580, 250), (0, 0, 255), -1)

    # road-ish background and lane line
    cv2.rectangle(img, (0, 350), (width, height), (100, 100, 100), -1)
    cv2.line(img, (0, 400), (width, 400), (255, 255, 255), 2)

    if save_path is None:
        save_path = "uploads/test_image.jpg"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)
    return save_path


def test_processing_pipeline(test_file: str | None = None) -> Dict:
    """Run the pipeline on a synthetic (or provided) image and report results."""
    print("Testing processing pipeline...")
    if test_file is None:
        test_file = create_test_image()
        print(f"  Created test image: {test_file}")

    try:
        from pipeline import process_media

        out_path = "outputs/test_output.jpg"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        result = process_media(
            input_path=test_file,
            output_path=out_path,
            kind="image",
            detect_w="",
            seg_w="",
        )
        dt = time.time() - t0

        # pipeline returns (path, metrics)
        if isinstance(result, tuple):
            result_path, metrics = result
        else:
            result_path, metrics = result, None

        if result_path and os.path.exists(result_path):
            info = get_file_info(result_path)
            print(f"  OK in {dt:.2f}s -> {info['size_mb']:.2f} MB")
            return {
                "success": True,
                "processing_time": dt,
                "output_path": result_path,
                "metrics": metrics or {},
            }
        return {"success": False, "error": "Output file not created"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_server_health(base_url: str = "http://127.0.0.1:5000") -> Dict:
    """Ping /health and / to ensure the server is up."""
    print(f"Testing server health at {base_url}...")
    res: Dict = {}

    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        res["health"] = {
            "status_code": r.status_code,
            "success": r.status_code == 200,
            "response": (r.json() if r.status_code == 200 else r.text),
        }
    except Exception as e:
        res["health"] = {"success": False, "error": str(e)}

    try:
        r = requests.get(base_url, timeout=5)
        res["main_page"] = {
            "status_code": r.status_code,
            "success": r.status_code == 200,
            "content_length": len(r.content),
        }
    except Exception as e:
        res["main_page"] = {"success": False, "error": str(e)}

    return res


def run_full_system_test() -> Dict:
    """Run requirements check, print config, run a small pipeline test."""
    print("Running full system test")
    print("=" * 60)

    results: Dict = {}
    results["system_requirements"] = check_system_requirements()

    try:
        from config import config
        config.print_config_summary()
    except Exception as e:
        print(f"Config summary failed: {e}")

    results["pipeline_test"] = test_processing_pipeline()

    print("\nSummary")
    print("-" * 60)
    ok = int(bool(results["system_requirements"])) + int(bool(results["pipeline_test"].get("success", False)))
    print(f" Passed {ok}/2 checks")

    return results
