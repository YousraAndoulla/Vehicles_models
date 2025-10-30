"""
Flask application for Traffic Congestion Analysis System
"""
import os
import sys
import re
import uuid
import mimetypes
import traceback
import base64
from flask import Flask, request, jsonify, Response, abort, send_file, send_from_directory
from werkzeug.utils import secure_filename
import time
import json
import queue
import threading
import base64

# Global dict to store active streams
active_streams = {}

# Add app directory to path
APP_DIR = os.environ.get("APP_DIR", "/content/drive/MyDrive/Project_PFE/app")
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
OUTPUT_DIR = os.path.join(APP_DIR, "outputs")
FRONT_DIR = os.path.join(APP_DIR, "frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRONT_DIR, exist_ok=True)

DETECT_WEIGHTS = os.environ.get("DETECT_WEIGHTS", "")
SEG_WEIGHTS = os.environ.get("SEG_WEIGHTS", "")

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
ALLOWED_VIDEO = {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi"}

def _ext(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()

_process_media = None
_process_stream_with_output = None

def process_media(*args, **kwargs):
    global _process_media
    if _process_media is None:
        try:
            from pipeline import process_media as PM
            _process_media = PM
        except Exception as e:
            print("[ERROR] Failed to import pipeline: " + str(e))
            raise RuntimeError("Pipeline import failed: " + str(e))
    return _process_media(*args, **kwargs)

def process_stream_with_output(*args, **kwargs):
    global _process_stream_with_output
    if _process_stream_with_output is None:
        try:
            from pipeline import process_stream_with_output as PSW
            _process_stream_with_output = PSW
        except Exception as e:
            print("[ERROR] Failed to import pipeline: " + str(e))
            raise RuntimeError("Pipeline import failed: " + str(e))
    return _process_stream_with_output(*args, **kwargs)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

@app.errorhandler(Exception)
def handle_error(e):
    """Catch all unhandled exceptions and return JSON"""
    print("[ERROR] Unhandled exception: " + str(e))
    traceback.print_exc()
    return jsonify(error="Server error: " + str(e)), 500

@app.after_request
def add_common_headers(resp):
    resp.headers.setdefault("Accept-Ranges", "bytes")
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp

@app.get("/health")
def health():
    from pathlib import Path
    def info(p):
        if not p:
            return {"path":"", "exists": False}
        q = Path(p)
        return {
            "path": str(q),
            "exists": q.is_file(),
            "size_mb": round(q.stat().st_size/1e6, 3) if q.is_file() else 0.0
        }
    return jsonify(status="ok", detect=info(DETECT_WEIGHTS), seg=info(SEG_WEIGHTS)), 200


@app.route("/assets/<path:fname>")
def assets(fname):
    path = os.path.join(FRONT_DIR, "assets", fname)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(os.path.join(FRONT_DIR, "assets"), fname)

@app.route("/<path:fname>")
def frontend_file(fname):
    path = os.path.join(FRONT_DIR, fname)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(FRONT_DIR, fname)

@app.get("/")
def index():
    return send_from_directory(FRONT_DIR, "index.html")

def _iter_file_range(path, start, end, chunk_size=1024*1024):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data

@app.route("/outputs/<path:fname>", methods=["GET","HEAD"])
def outputs(fname):
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.isfile(path):
        abort(404)
    ext = os.path.splitext(path)[1].lower()
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    if ext in (".mp4",".m4v"):
        mime = "video/mp4"
    if not mime.startswith("video/"):
        if request.method == "HEAD":
            resp = Response(status=200, mimetype=mime)
            resp.headers["Content-Length"] = os.path.getsize(path)
            return resp
        return send_from_directory(OUTPUT_DIR, fname, as_attachment=False)
    file_size = os.path.getsize(path)
    if request.method == "HEAD":
        resp = Response(status=200, mimetype=mime)
        resp.headers["Content-Length"] = file_size
        resp.headers["Accept-Ranges"] = "bytes"
        return resp
    range_header = request.headers.get("Range")
    if not range_header:
        return send_file(path, mimetype=mime, conditional=True)
    m = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not m:
        return send_file(path, mimetype=mime, conditional=True)
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else file_size-1
    start = max(0,start)
    end = min(end,file_size-1)
    if start > end:
        return Response(status=416, headers={"Content-Range": "bytes */" + str(file_size)})
    length = end-start+1
    resp = Response(_iter_file_range(path,start,end), status=206, mimetype=mime, direct_passthrough=True)
    resp.headers["Content-Range"] = "bytes " + str(start) + "-" + str(end) + "/" + str(file_size)
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(length)
    return resp

@app.post("/api/process")
def api_process():
    try:
        t0 = time.time()

        if "file" not in request.files:
            return jsonify(error="No file uploaded"), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify(error="No file selected"), 400

        original_name = secure_filename(file.filename)
        ext = _ext(original_name)

        if ext in ALLOWED_IMAGE:
            kind = "image"
        elif ext in ALLOWED_VIDEO:
            kind = "video"
        else:
            return jsonify(error="Unsupported file type: " + ext), 400

        upload_id = str(uuid.uuid4())
        input_filename = upload_id + ext
        input_path = os.path.join(UPLOAD_DIR, input_filename)

        print("[UPLOAD] Saving " + original_name + " (" + kind + ") to " + input_path)
        file.save(input_path)

        output_name = upload_id + "_output" + ext
        output_path = os.path.join(OUTPUT_DIR, output_name)

        print("[PROCESS] Starting " + kind + " processing...")

        result = process_media(input_path, output_path, kind, DETECT_WEIGHTS, SEG_WEIGHTS)

        if isinstance(result, tuple):
            result_path, metrics = result
        else:
            result_path, metrics = result, None

        if not os.path.exists(result_path):
            return jsonify(error="No output generated"), 500

        try:
            os.remove(input_path)
        except:
            pass

        elapsed = time.time() - t0
        print("[OK] " + str(round(elapsed, 1)) + "s")

        if metrics:
            try:
                from database import get_database
                db = get_database()

                if db and db.client:
                    db_data = {
                        "user_id": request.form.get("user_id", "anonymous"),
                        "file_name": original_name,
                        "file_type": kind,
                        "vehicle_count": metrics.get("vehicles", 0),
                        "occupancy": metrics.get("occupancy", 0.0),
                        "flow_rate_vph": metrics.get("flow_rate_vph", 0.0),
                        "congestion_level": metrics.get("congestion_level", 0),
                        "processing_duration": elapsed
                    }

                    analysis_id = db.save_analysis(db_data)
                    print("[DB] Saved analysis: " + str(analysis_id))
                else:
                    print("[DB] Database not available")

            except Exception as e:
                print("[DB] Failed to save: " + str(e))

        return jsonify(
            success=True,
            kind=kind,
            output="/outputs/" + output_name,
            size=os.path.getsize(result_path),
            filename=output_name,
            metrics=metrics,
            processing_time=round(elapsed, 1)
        ), 200

    except Exception as e:
        print("[ERROR] " + str(e))
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@app.route("/api/analysis/history", methods=["GET"])
def api_get_history():
    """Get analysis history for a user"""
    try:
        user_id = request.args.get("user_id", "anonymous")
        limit = min(int(request.args.get("limit", 50)), 100)

        from database import get_database
        db = get_database()

        if not db or not db.client:
            return jsonify(history=[], message="Database not available"), 200

        history = db.get_history(user_id, limit)

        return jsonify(
            success=True,
            history=history,
            count=len(history)
        ), 200

    except Exception as e:
        print("[HISTORY] Error: " + str(e))
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@app.route("/api/stream/start", methods=["POST"])
def api_stream_start():
    """Start a real-time stream with live frames"""
    try:
        data = request.get_json()
        stream_url = data.get("stream_url", "").strip()
        duration = int(data.get("duration", 3600))

        if not stream_url:
            return jsonify(error="No URL"), 400

        stream_id = str(uuid.uuid4())

        frame_queue = queue.Queue(maxsize=10)
        metrics_queue = queue.Queue()

        active_streams[stream_id] = {
            "url": stream_url,
            "frame_queue": frame_queue,
            "metrics_queue": metrics_queue,
            "active": True
        }

        def process_stream_live():
            try:
                process_stream_with_output(
                    stream_url,
                    duration,
                    DETECT_WEIGHTS,
                    SEG_WEIGHTS,
                    frame_queue,
                    metrics_queue,
                    stream_id
                )
            finally:
                active_streams[stream_id]["active"] = False

        thread = threading.Thread(target=process_stream_live, daemon=True)
        thread.start()

        return jsonify(success=True, stream_id=stream_id), 200

    except Exception as e:
        print("[STREAM] Error: " + str(e))
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@app.route("/api/stream/<stream_id>/frames")
def api_stream_frames(stream_id):
    """Server-Sent Events endpoint for live frames"""

    def generate_frames():
        if stream_id not in active_streams:
            yield "data: " + json.dumps({"error": "Stream not found"}) + "\n\n"
            return

        stream = active_streams[stream_id]
        frame_queue = stream["frame_queue"]
        
        print(f"[SSE] Starting frame stream for {stream_id}")
        frame_count = 0

        while stream["active"] or not frame_queue.empty():
            try:
                frame_data = frame_queue.get(timeout=2.0)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"[SSE] Sent {frame_count} frames for {stream_id}")
                
                yield "data: " + json.dumps(frame_data) + "\n\n"

            except queue.Empty:
                yield ": heartbeat\n\n"
                continue
            except Exception as e:
                print(f"[SSE] Frame error: {e}")
                yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
                break

        print(f"[SSE] Stream {stream_id} complete. Total frames: {frame_count}")
        
        if frame_count > 0:
            yield "data: " + json.dumps({"done": True, "total_frames": frame_count}) + "\n\n"
        else:
            yield "data: " + json.dumps({"error": "No frames received from stream"}) + "\n\n"

    return Response(
        generate_frames(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

@app.route("/api/stream/<stream_id>/metrics")
def api_stream_metrics(stream_id):
    """Get current metrics for a stream"""

    if stream_id not in active_streams:
        return jsonify(error="Stream not found"), 404

    stream = active_streams[stream_id]
    metrics_queue = stream["metrics_queue"]

    metrics = []
    while not metrics_queue.empty():
        try:
            metrics.append(metrics_queue.get_nowait())
        except queue.Empty:
            break

    return jsonify(
        success=True,
        metrics=metrics,
        active=stream["active"]
    ), 200

@app.route("/api/stream/<stream_id>/stop", methods=["POST"])
def api_stream_stop(stream_id):
    """Stop a stream"""

    if stream_id in active_streams:
        active_streams[stream_id]["active"] = False
        del active_streams[stream_id]
        return jsonify(success=True), 200

    return jsonify(error="Stream not found"), 404

@app.route("/api/realtime/sessions", methods=["GET"])
def api_realtime_sessions():
    """Get list of real-time monitoring sessions"""
    try:
        user_id = request.args.get("user_id", "anonymous")
        from database import get_database
        db = get_database()

        if not db or not db.client:
            return jsonify(sessions=[], message="Database not available"), 200

        sessions = db.get_realtime_sessions(user_id, limit=20)
        return jsonify(success=True, sessions=sessions), 200

    except Exception as e:
        print("[SESSIONS] Error: " + str(e))
        return jsonify(error=str(e)), 500

@app.route("/api/realtime/metrics/<session_id>", methods=["GET"])
def api_realtime_metrics(session_id):
    """Get metrics for a specific session"""
    try:
        from database import get_database
        db = get_database()

        if not db or not db.client:
            return jsonify(metrics=[], message="Database not available"), 200

        metrics = db.get_session_metrics(session_id)
        return jsonify(success=True, metrics=metrics, count=len(metrics)), 200

    except Exception as e:
        print("[METRICS] Error: " + str(e))
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    print("=" * 60)
    print("Traffic Analysis Server")
    print("=" * 60)
    print("App: " + APP_DIR)
    print("Detect: " + (DETECT_WEIGHTS or "Default"))
    print("Segment: " + (SEG_WEIGHTS or "None"))
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
