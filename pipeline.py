
import os, cv2, subprocess, random
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from ultralytics import YOLO
import time
from pathlib import Path
import hashlib

def _sha256(p):
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "unknown"

def _log_model(m, label="MODEL"):
    # m is a YOLO model
    ckpt = getattr(m, "ckpt_path", None)
    if ckpt:
        print(f"[{label}] {Path(ckpt).resolve()}  sha256={_sha256(ckpt)}  task={getattr(m, 'task', '?')}")
    else:
        print(f"[{label}] (no ckpt_path) task={getattr(m, 'task', '?')}")


# ----- determinism (so same video => same results) -----
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
random.seed(0)
np.random.seed(0)
try:
    import torch
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

# -------------------- CONFIG --------------------
TRACKER_CFG = "bytetrack.yaml"
DELTA_T = 15.0                         # seconds per bin (driven by frame count)
MAX_LANES = 4
LANE_DILATE_PX = 10
MIN_LANE_AREA = 8000
LANE_CAPACITY_VPH = 1800.0

YELLOW_THRESH = dict(occ=0.25, vc=0.70, cross=0.10)
RED_THRESH    = dict(occ=0.45, vc=0.95, cross=0.30)

VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]


# Map any class name to your 4 vehicle types; everything else -> None
VEHICLE_NAME_MAP = {
    "car":"car","auto":"car","automobile":"car","sedan":"car","suv":"car","hatchback":"car",
    "bus":"bus","coach":"bus",
    "truck":"truck","lorry":"truck","semi":"truck","trailer":"truck","freight":"truck",
    "motorcycle":"motorcycle","motorbike":"motorcycle","bike":"motorcycle","scooter":"motorcycle","moped":"motorcycle",
}
def _as_vehicle(name: str):
    n = name.lower().strip()
    for k, v in VEHICLE_NAME_MAP.items():
        if k in n:
            return v
    return None


# -------------------- HELPERS --------------------
def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def ccw(p, q, r):
    return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

def line_intersects(a, b, c, d):
    return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

def _transcode_to_h264(path: str):
    tmp = path + ".h264.mp4"
    try:
        subprocess.run(
            ["ffmpeg","-y","-i",path,"-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","23",
             "-c:a","aac","-b:a","128k","-movflags","+faststart",tmp],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception:
            pass

# -------------------- SEGMENTATION --------------------
def infer_seg_indices(seg_model):
    idx = {"road": 0, "lane": 1, "cross": 2}
    if seg_model is None:
        return idx
    names = {int(i): str(n).lower() for i, n in (getattr(seg_model.model, "names", {}) or {}).items()}
    def find(keys, default):
        for i, n in names.items():
            if any(k in n for k in keys):
                return i
        return default
    idx["road"] = find(["road","drivable","asphalt","street","roadway"], idx["road"])
    idx["lane"] = find(["lane","lane-line","lanes","marking","divider","centerline"], idx["lane"])
    idx["cross"] = find(["pedestrian","cross","zebra","crosswalk","cross-walk"], idx["cross"])
    print("[SEG] infer_seg_indices ->", idx, "| names:", names)
    return idx


def extract_masks(seg_result, H, W, idx_map):
    if seg_result.masks is None or seg_result.boxes is None:
        return None, None, None
    classes = seg_result.boxes.cls.int().cpu().numpy().tolist()
    masks = seg_result.masks.data.cpu().numpy()
    def collect_masks(target_idx):
        combined = None
        for mask, cls in zip(masks, classes):
            if int(cls) == int(target_idx):
                binary = (mask > 0.20).astype(np.uint8)
                combined = binary if combined is None else np.maximum(combined, binary)
        if combined is None:
            return None
        if combined.shape != (H, W):
            combined = cv2.resize(combined, (W, H), interpolation=cv2.INTER_NEAREST)
        return combined
    return (
        collect_masks(idx_map["road"]),
        collect_masks(idx_map["lane"]),
        collect_masks(idx_map["cross"]),
    )


# --- Size-aware parameters based on frame size ---
def _size_aware_params(W, H):
    area = W * H
    dilate_px = max(3, int(0.008 * W))              # ~0.8% of width
    min_lane_area = max(2000, int(0.012 * area))    # ~1.2% of frame
    min_lane_width = max(20, int(0.06 * W))         # ~6% of width
    return dilate_px, min_lane_area, min_lane_width

def split_road_into_lanes(road_mask, lane_line_mask, W, H):
    """
    Split drivable area into lanes using lane-line barriers.
    Falls back to a horizontal-histogram method if lane lines are weak/missing.
    Returns a list of binary masks sorted left→right.
    """
    if road_mask is None:
        return []

    road = (road_mask > 0).astype(np.uint8)
    dilate_px, min_lane_area, min_lane_width = _size_aware_params(W, H)

    # Primary path: use lane-line barriers
    if lane_line_mask is not None and lane_line_mask.any():
        lanes = road.copy()
        lines = (lane_line_mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
        barriers = cv2.dilate(lines, kernel, iterations=1)
        lanes[barriers == 1] = 0

        num, labels = cv2.connectedComponents(lanes)
        masks = []
        for i in range(1, num):
            m = (labels == i).astype(np.uint8)
            if int(m.sum()) >= min_lane_area:
                masks.append(m)

        # sort left→right
        def x_centroid(m):
            ys, xs = np.where(m > 0)
            return xs.mean() if xs.size else 1e9
        masks.sort(key=x_centroid)

        if len(masks) >= 2:  # CHANGE: require at least 2 lanes
            return masks
        elif len(masks) == 1 and int((masks[0] > 0).sum()) > 0.5 * int((road > 0).sum()):
            # If only 1 big lane found, it's likely the whole road - try fallback
            pass
        elif len(masks) >= 1:  # NEW: if 1 lane but small, keep it
            return masks

    # Fallback: histogram slice at the counting line
    y = int(0.62 * H)
    stripe = road[max(0, y-4):min(H, y+5), :]
    horiz = stripe.mean(axis=0)

    runs, on, start = [], False, 0
    for x in range(W):
        if horiz[x] > 0.15 and not on:  # CHANGE: 0.1 -> 0.15 (more strict)
            on, start = True, x
        elif (horiz[x] <= 0.15 or x == W-1) and on:
            end = x if horiz[x] <= 0.15 else x+1
            if end - start >= min_lane_width * 1.2:  # CHANGE: add 1.2x multiplier
                runs.append((start, end))
            on = False

    masks = []
    for (xs, xe) in runs:
        m = np.zeros((H, W), np.uint8)
        m[:, xs:xe] = 1
        m &= road
        if m.sum() >= 0.8 * min_lane_area:  # CHANGE: 0.7 -> 0.8 (more strict)
            masks.append(m)

    masks.sort(key=lambda m: np.where(m>0)[1].mean() if (m>0).any() else 1e9)
    return masks[:MAX_LANES]

def calculate_active_lanes(lane_masks, road_mask):
    """
    Count lanes relative to road area (robust to resolution/zoom).
    """
    if road_mask is None or not lane_masks:
        return 1
    road_area = max(1, int((road_mask > 0).sum()))
    active = 0
    for m in lane_masks:
        a = int((m > 0).sum())
        if a >= 0.12 * road_area:  # each lane must be ≥12% of road area
            active += 1
    return max(1, active)

def classify_congestion_level(flow_bin, occupancy_bin, crosswalk_blocking, active_lanes):
    bin_hours = DELTA_T / 3600.0
    hourly_volume = (flow_bin / bin_hours) if bin_hours > 0 else 0.0
    total_capacity = active_lanes * LANE_CAPACITY_VPH
    vc_ratio = (hourly_volume / total_capacity) if total_capacity > 0 else 0.0
    if (occupancy_bin >= RED_THRESH["occ"] or vc_ratio >= RED_THRESH["vc"] or crosswalk_blocking >= RED_THRESH["cross"]):
        return 2  # red
    if (occupancy_bin >= YELLOW_THRESH["occ"] or vc_ratio >= YELLOW_THRESH["vc"] or crosswalk_blocking >= YELLOW_THRESH["cross"]):
        return 1  # yellow
    return 0      # green

def overlay_congestion_status(frame, level, metrics_text=None):
    colors = [(50,220,50),(0,255,255),(20,20,230)]  # G, Y, R in BGR
    status = ["FREE FLOW","MODERATE","CONGESTED"][level]
    color = colors[level]
    box_h = 80 if metrics_text else 60
    cv2.rectangle(frame, (10,10), (450,10+box_h), color, -1)
    cv2.putText(frame, f"TRAFFIC: {status}", (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 3)
    if metrics_text:
        cv2.putText(frame, metrics_text, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    return frame

# -------------------- ACCUMULATOR --------------------
@dataclass
class TrafficBinStats:
    timestamps: list = field(default_factory=list)
    vehicle_counts: dict = field(default_factory=lambda: defaultdict(int))
    total_flow: int = 0
    occupancy_samples: list = field(default_factory=list)
    crosswalk_blocking_samples: list = field(default_factory=list)
    lane_flow_counts: list = field(default_factory=lambda: [0] * MAX_LANES)
    def add_occupancy_sample(self, x): self.occupancy_samples.append(float(x))
    def add_crosswalk_blocking(self, x): self.crosswalk_blocking_samples.append(float(x))
    def increment_vehicle_class(self, k): self.vehicle_counts[k] += 1
    def finalize_bin(self, t0, t1):
        avg_occ = float(np.mean(self.occupancy_samples)) if self.occupancy_samples else 0.0
        avg_xw  = float(np.mean(self.crosswalk_blocking_samples)) if self.crosswalk_blocking_samples else 0.0
        return {"t_start": t0, "t_end": t1, "occupancy": avg_occ, "flow": int(self.total_flow), "crosswalk_blocked": avg_xw}

# -------------------- FALLBACK ROAD MASK --------------------
def _road_proxy_mask(veh_mask: np.ndarray, H: int, W: int, cross: np.ndarray | None = None) -> np.ndarray:
    proxy = np.zeros((H, W), np.uint8)

    # assume lower half is drivable if we have nothing else
    y0 = int(0.45 * H)
    proxy[y0:H, :] = 1

    # expand vehicles to approximate drivable area
    k = max(5, W // 80)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    veh_d = cv2.dilate((veh_mask > 0).astype(np.uint8), ker, iterations=3)
    proxy = np.maximum(proxy, veh_d)

    # include crosswalks if present
    if cross is not None:
        proxy[cross > 0] = 1

    num, labels = cv2.connectedComponents(proxy)
    if num <= 1:
        return proxy

    # prefer the largest component that touches the bottom row
    best_label, best_area = 0, 0
    bottom_row = labels[H - 1, :]
    touching = set(bottom_row.tolist())
    for lab in range(1, num):
        if lab in touching:
            area = int((labels == lab).sum())
            if area > best_area:
                best_area, best_label = area, lab

    # if none touch the bottom, take the largest anywhere
    if best_label == 0:
        for lab in range(1, num):
            area = int((labels == lab).sum())
            if area > best_area:
                best_area, best_label = area, lab

    proxy = (labels == best_label).astype(np.uint8)
    proxy = cv2.morphologyEx(proxy, cv2.MORPH_CLOSE, ker, iterations=1)
    return proxy



# -------------------- IMAGE (metrics = vehicles, occ, xwalk only) --------------------
def process_image(input_path: str, output_path: str, detect_w: str = "", seg_w: str = ""):
    det = YOLO(detect_w) if detect_w else YOLO("yolov8n.pt")
    _log_model(det, "DETECT(image)")

    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {input_path}")
    H, W = img.shape[:2]

    # 1) VEHICLE DETECTION
    res = det.predict(source=img, conf=0.25, iou=0.5, imgsz=960, verbose=False)[0]
    names = getattr(det.model, "names", {}) or {}

    # keep vehicle classes only
    keep = []
    if res.boxes is not None and len(res.boxes):
        for c in res.boxes.cls.int().cpu().numpy():
            keep.append(_as_vehicle(str(names.get(int(c), c))) is not None)
        keep = np.array(keep, dtype=bool)
        res.boxes = res.boxes[keep] if keep.any() else res.boxes[:0]

    # draw detections
    ann = res.plot(img=img.copy(), line_width=2)

    # build vehicle mask + counts
    veh_mask = np.zeros((H, W), np.uint8)
    vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
    total_veh = 0
    if res.boxes is not None and len(res.boxes):
        bbs = res.boxes.xyxy.cpu().numpy()
        cls_np = res.boxes.cls.int().cpu().numpy()
        for bb, cid in zip(bbs, cls_np):
            vtype = _as_vehicle(str(names.get(int(cid), cid)))
            if not vtype:
                continue
            x1, y1, x2, y2 = map(int, bb.tolist())
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
            cv2.rectangle(veh_mask, (x1, y1), (x2, y2), 1, -1)
            vehicle_counts[vtype] += 1
            total_veh += 1

    # 2) ROAD / CROSSWALK SEGMENTATION (for metrics)
    road = cross = None
    if seg_w and os.path.exists(seg_w):
        seg_model = YOLO(seg_w)
        _log_model(seg_model, "SEG(image)")
        s = seg_model.predict(source=img, imgsz=960, conf=0.15,
                              retina_masks=True, verbose=False)[0]
        seg_idx = infer_seg_indices(seg_model)
        road, _, cross = extract_masks(s, H, W, seg_idx)

        min_area = 0.01 * H * W  # 1% of frame; tune if needed
        if road is None or int((road > 0).sum()) < min_area:
            road = _road_proxy_mask(veh_mask, H, W, cross)

        # ---- DEBUG PRINTS ----
        print("[SEG] weights:", seg_w, "exists=", os.path.exists(seg_w))
        print("[SEG] names:", getattr(seg_model.model, "names", {}))
        road_area  = int((road  > 0).sum()) if road  is not None else 0
        cross_area = int((cross > 0).sum()) if cross is not None else 0
        print("[SEG] areas  road/cross:", road_area, cross_area)

    # 3) METRICS
    occ = 0.0
    if road is not None and road.any():
        road_area = int((road > 0).sum())
        inter = (veh_mask & (road > 0).astype(np.uint8)).sum()
        occ = float(inter / max(road_area, 1))

    xwalk = 0.0
    if cross is not None and cross.any():
        cross_area = int((cross > 0).sum())
        inter = (veh_mask & (cross > 0)).sum()
        xwalk = float(inter / max(cross_area, 1))

    # 4) BANNER
    level = classify_congestion_level(flow_bin=0, occupancy_bin=occ,
                                      crosswalk_blocking=xwalk, active_lanes=1)
    label_map = {0: "Optimal Flow", 1: "Moderate", 2: "Heavy Congestion"}
    ann = overlay_congestion_status(ann, level, metrics_text=f"occ={occ*100:.1f}%  xwalk={xwalk*100:.1f}%")

    metrics = {
        "vehicles": int(total_veh),
        "by_class": vehicle_counts,
        "occupancy": float(occ),
        "crosswalk_blocked": float(xwalk),
        "congestion_level": int(level),
        "congestion_label": label_map[level],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, ann)
    return output_path, metrics


# -------------------- VIDEO --------------------
def process_video(input_path: str, output_path: str, detect_w: str = "", seg_w: str = ""):
    det = YOLO(detect_w) if detect_w else YOLO("yolov8n.pt")
    _log_model(det, "DETECT(video)")
    seg_model = YOLO(seg_w) if seg_w else None
    if seg_model:
        _log_model(seg_model, "SEG(video)")


    # *** DEBUG PRINTS here ***
    if seg_model is None:
        print("[SEG] seg_model=None (segmentation disabled or bad path)")
        seg_indices = {"road": 0, "lane": 1, "cross": 2}
    else:
        print("[SEG] names:", getattr(seg_model.model, "names", {}))
        seg_indices = infer_seg_indices(seg_model)
        print("[SEG] indices:", seg_indices)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    counting_y = int(0.62 * height)
    line_start = (int(0.05 * width), counting_y)
    line_end   = (int(0.95 * width), counting_y)

    previous_positions = {}
    bin_stats = TrafficBinStats()
    bin_frames = max(1, int(DELTA_T * fps))
    frames_in_bin = 0

    all_occ = []
    all_xw = []
    total_cross = 0
    active_lanes_hist = []
    bin_levels = []
    current_level = None
    current_text = None

    frame_idx = 0
    stream = det.track(
        source=input_path,
        tracker=TRACKER_CFG,
        stream=True,
        conf=0.25,
        iou=0.5,
        persist=True,
        imgsz=960,
        vid_stride=1,           # <<< determinism: process every frame
        verbose=False
    )

    for fr in stream:
        frame = fr.orig_img
        if frame is None:
            break
        frame_idx += 1
        H, W = frame.shape[:2]

        road = lane_lines = cross = None
        if seg_model is not None:
            seg = seg_model.predict(source=frame, imgsz=960, conf=0.15,
                                    retina_masks=True, verbose=False)[0]
            road, lane_lines, cross = extract_masks(seg, H, W, seg_indices)

        # --- build a uint8 vehicle mask for morphology ---
        veh_mask_u8 = np.zeros((H, W), np.uint8)
        if fr.boxes is not None and len(fr.boxes) > 0:
            for b in fr.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b.tolist())
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
                cv2.rectangle(veh_mask_u8, (x1, y1), (x2, y2), 1, -1)

        k = max(5, W // 80)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        veh_mask_u8 = cv2.dilate(veh_mask_u8, ker, iterations=3)

        if road is None or int((road > 0).sum()) < 0.01 * H * W:
            road = _road_proxy_mask(veh_mask=veh_mask_u8, H=H, W=W, cross=cross)

        lane_masks = split_road_into_lanes(road, lane_lines, W, H)
        lane_masks = (lane_masks + [np.zeros((H, W), np.uint8)] * MAX_LANES)[:MAX_LANES]

        veh_mask = veh_mask_u8.copy()

        occ = 0.0
        if road is not None and road.sum()>0:
            area = int((road>0).sum())
            inter = (veh_mask & (road>0).astype(np.uint8)).sum()
            occ = float(inter / max(area,1))

        xw = 0.0
        if cross is not None and cross.sum()>0:
            area = int((cross>0).sum())
            inter = ((veh_mask & (cross>0))).sum()
            xw = float(inter / max(area,1))

        lane_flows = [0] * MAX_LANES
        crossings = 0
        if fr.boxes is not None and len(fr.boxes)>0:
            class_ids = fr.boxes.cls.int().cpu().numpy()
            track_ids = fr.boxes.id
            track_ids = track_ids.int().cpu().numpy() if track_ids is not None else np.array([-1]*len(class_ids))
            bboxes = fr.boxes.xyxy.cpu().numpy()
            names = getattr(det.model, "names", {}) or {}
            for bb, cid, tid in zip(bboxes, class_ids, track_ids):
                # light class mapping (optional counts)
                name = str(names.get(int(cid), cid)).lower()
                if   "car" in name or "auto" in name: vtype = "car"
                elif "motor" in name or "bike" in name: vtype = "motorcycle"
                elif "bus" in name: vtype = "bus"
                elif "truck" in name or "lorry" in name: vtype = "truck"
                else: vtype = None
                if vtype: bin_stats.increment_vehicle_class(vtype)

                cur = bbox_center(bb)
                lane_idx = None
                for i, lm in enumerate(lane_masks):
                    yy = int(min(max(int(cur[1]), 0), H-1))
                    xx = int(min(max(int(cur[0]), 0), W-1))
                    if lm[yy, xx]: lane_idx = i; break

                if tid in previous_positions:
                    prv = previous_positions[tid]
                    if line_intersects(prv, cur, line_start, line_end):
                        crossings += 1
                        if lane_idx is not None:
                            lane_flows[lane_idx] += 1
                previous_positions[tid] = cur

        # accumulate
        bin_stats.add_occupancy_sample(occ)
        bin_stats.add_crosswalk_blocking(xw)
        bin_stats.total_flow += crossings
        bin_stats.timestamps.append(frame_idx / max(fps, 1.0))
        for i, v in enumerate(lane_flows): bin_stats.lane_flow_counts[i] += v

        all_occ.append(occ); all_xw.append(xw); total_cross += crossings
        frames_in_bin += 1

        # finalize deterministic bin
        if frames_in_bin >= bin_frames:
            f = bin_stats.finalize_bin(bin_stats.timestamps[0], bin_stats.timestamps[-1])
            active_lanes = calculate_active_lanes(lane_masks, road)
            active_lanes_hist.append(active_lanes)
            current_level = classify_congestion_level(
                flow_bin=bin_stats.total_flow,
                occupancy_bin=f["occupancy"],
                crosswalk_blocking=f["crosswalk_blocked"],
                active_lanes=active_lanes
            )
            bin_levels.append(current_level)
            # annotate text
            bin_hours = DELTA_T / 3600.0
            vph = bin_stats.total_flow / bin_hours if bin_hours > 0 else 0.0
            vc  = vph / (active_lanes * LANE_CAPACITY_VPH) if active_lanes > 0 else 0.0
            current_text = (f"occ={f['occupancy']*100:.1f}%  "
                           f"v/c={vc:.2f} "
                           f"xwalk={f['crosswalk_blocked']*100:.1f}%")


            # reset
            bin_stats = TrafficBinStats()
            frames_in_bin = 0

        # draw & write
        annotated = fr.plot()
        if current_level is not None:
            annotated = overlay_congestion_status(annotated, current_level, current_text)
        cv2.line(annotated, line_start, line_end, (0,255,255), 3)
        vw.write(annotated)

    vw.release()

    # --- final summary for frontend (single source of truth) ---
    duration_s = frame_idx / max(fps, 1.0)
    vph_total = (total_cross / max(duration_s, 1e-6)) * 3600.0
    avg_occupancy = float(np.mean(all_occ)) if all_occ else 0.0
    active_final = int(round(np.mean(active_lanes_hist))) if active_lanes_hist else 1

    if bin_levels:
        final_level = int(bin_levels[-1])          # last-bin status drives UI + metrics
    else:
        final_level = int(current_level) if current_level is not None else 0

    label_map = {0: "Optimal Flow", 1: "Moderate", 2: "Heavy Congestion"}
    summary = {
        "vehicles": int(total_cross),
        "active_lanes": int(max(1, active_final)),
        "occupancy": float(avg_occupancy),
        "flow_rate_vph": float(vph_total),
        "congestion_level": final_level,
        "congestion_label": label_map.get(final_level)
    }

    _transcode_to_h264(output_path)
    return output_path, summary

def process_stream_with_output(stream_url: str, duration: int, detect_w: str, seg_w: str,
                                frame_queue, metrics_queue, stream_id):
    """Process stream and output frames + metrics in real-time via queues"""
    print(f"[LIVE] Starting stream {stream_id}: {stream_url}")

    det = YOLO(detect_w) if detect_w else YOLO("yolov8n.pt")
    _log_model(det, "DETECT(live)")
    seg_model = YOLO(seg_w) if seg_w else None
    if seg_model:
        _log_model(seg_model, "SEG(live)")


    if seg_model is None:
        seg_indices = {"road": 0, "lane": 1, "cross": 2}
    else:
        seg_indices = infer_seg_indices(seg_model)

    # Handle YouTube URLs with IMPROVED settings
    actual_stream_url = stream_url
    if 'youtube.com' in stream_url or 'youtu.be' in stream_url:
        try:
            import yt_dlp
            ydl_opts = {
                'format': 'best[height<=480][ext=mp4]/best[height<=480]/best',
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
                # NEW: Add buffer and retry options
                'http_chunk_size': 10485760,  # 10MB chunks
                'retries': 10,
                'fragment_retries': 10,
                'extractor_retries': 10,
                'socket_timeout': 30,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(stream_url, download=False)
                actual_stream_url = info.get('url')
                print(f"[LIVE] Extracted URL: {actual_stream_url[:100]}...")
        except Exception as e:
            print(f"[LIVE] YouTube error: {e}")
            return

    # NEW: Open stream with better settings
    cap = cv2.VideoCapture(actual_stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Keep only 3 frames in buffer
    
    if not cap.isOpened():
        print(f"[LIVE] Cannot open stream")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[LIVE] Stream: {width}x{height} @ {fps}fps")

    # Setup
    SNAPSHOT_INTERVAL = 10.0
    snapshot_frames = max(1, int(SNAPSHOT_INTERVAL * fps))

    bin_stats = TrafficBinStats()
    previous_positions = {}
    frames_in_bin = 0
    total_frames = 0
    start_time = time.time()

    counting_y = int(0.62 * height)
    line_start = (int(0.05 * width), counting_y)
    line_end = (int(0.95 * width), counting_y)

    # NEW: Adaptive frame skip
    frame_skip = 2
    processing_times = []
    
    db_session_id = None
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 30

    try:
        while True:
            elapsed = time.time() - start_time
            
            if elapsed >= duration:
                print(f"[LIVE] Duration reached: {elapsed:.1f}s")
                break

            # NEW: Retry on read failure
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"[LIVE] Frame read failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("[LIVE] Too many consecutive failures, stopping")
                    break
                    
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            total_frames += 1

            # Adaptive frame skipping
            if total_frames % frame_skip != 0:
                continue

            frame_start_time = time.time()
            H, W = frame.shape[:2]

            # NEW: Lower resolution for speed
            det_result = det.track(
                source=frame,
                tracker=TRACKER_CFG,
                conf=0.25,
                iou=0.5,
                persist=True,
                imgsz=320,  # Was 480
                verbose=False
            )

            if not det_result or len(det_result) == 0:
                continue

            fr = det_result[0]

            # Build vehicle mask
            veh_mask = np.zeros((H, W), np.uint8)
            if fr.boxes is not None and len(fr.boxes) > 0:
                for bbox in fr.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, bbox.tolist())
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(W-1, x2); y2 = min(H-1, y2)
                    cv2.rectangle(veh_mask, (x1, y1), (x2, y2), 1, -1)

            k = max(5, W // 80)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            veh_mask = cv2.dilate(veh_mask, ker, iterations=2)

            # NEW: Segment only every 5th frame
            road = cross = lane_lines = None
            if seg_model is not None and frames_in_bin % 5 == 0:
                seg_result = seg_model.predict(
                    source=frame, imgsz=320, conf=0.15,  # Was 480
                    retina_masks=True, verbose=False
                )[0]
                road, lane_lines, cross = extract_masks(seg_result, H, W, seg_indices)

            if road is None or int((road > 0).sum()) < 0.01 * H * W:
                road = _road_proxy_mask(veh_mask, H, W, cross)

            lane_masks = split_road_into_lanes(road, lane_lines, W, H)
            lane_masks = (lane_masks + [np.zeros((H, W), np.uint8)] * MAX_LANES)[:MAX_LANES]

            # Calculate metrics
            occ = 0.0
            if road is not None and road.sum() > 0:
                road_area = int((road > 0).sum())
                vehicle_on_road = (veh_mask & (road > 0).astype(np.uint8)).sum()
                occ = float(vehicle_on_road / max(road_area, 1))

            xw = 0.0
            if cross is not None and cross.sum() > 0:
                cross_area = int((cross > 0).sum())
                vehicles_on_cross = (veh_mask & (cross > 0)).sum()
                xw = float(vehicles_on_cross / max(cross_area, 1))

            # Count crossings
            crossings = 0
            if fr.boxes is not None and len(fr.boxes) > 0:
                track_ids = fr.boxes.id
                track_ids = track_ids.int().cpu().numpy() if track_ids is not None else np.array([-1] * len(fr.boxes))
                bboxes = fr.boxes.xyxy.cpu().numpy()

                for bbox, tid in zip(bboxes, track_ids):
                    cur_pos = bbox_center(bbox)

                    if tid in previous_positions:
                        prev_pos = previous_positions[tid]
                        if line_intersects(prev_pos, cur_pos, line_start, line_end):
                            crossings += 1

                    previous_positions[tid] = cur_pos

            # Accumulate
            bin_stats.add_occupancy_sample(occ)
            bin_stats.add_crosswalk_blocking(xw)
            bin_stats.total_flow += crossings
            frames_in_bin += 1

            # Draw annotated frame
            annotated = fr.plot() if hasattr(fr, 'plot') else frame.copy()
            cv2.line(annotated, line_start, line_end, (0, 255, 255), 2)

            # Add metrics overlay
            active_lanes = calculate_active_lanes(lane_masks, road)
            level = classify_congestion_level(
                flow_bin=bin_stats.total_flow,
                occupancy_bin=occ,
                crosswalk_blocking=xw,
                active_lanes=active_lanes
            )

            current_vph = bin_stats.total_flow / (elapsed/3600) if elapsed > 0 else 0
            total_capacity = active_lanes * LANE_CAPACITY_VPH
            vc_ratio = (current_vph / total_capacity) if total_capacity > 0 else 0.0

            overlay_text = f"occ={occ*100:.1f}%  v/c={vc_ratio:.2f}  xwalk={xw*100:.1f}%"
            annotated = overlay_congestion_status(annotated, level, overlay_text)

            # NEW: Lower JPEG quality for speed
            import base64
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Send frame (non-blocking)
            try:
                frame_queue.put_nowait({
                    'frame': frame_b64,
                    'timestamp': elapsed,
                    'vehicles': bin_stats.total_flow,
                    'active_lanes': active_lanes,
                    'occupancy': occ,
                    'crosswalk_blocked': xw,
                    'flow_rate_vph': current_vph,
                    'congestion_level': level
                })
            except:
                pass

            # Snapshot every 10 seconds
            if frames_in_bin >= snapshot_frames:
                avg_occ = float(np.mean(bin_stats.occupancy_samples)) if bin_stats.occupancy_samples else 0.0
                avg_xw = float(np.mean(bin_stats.crosswalk_blocking_samples)) if bin_stats.crosswalk_blocking_samples else 0.0

                bin_hours = SNAPSHOT_INTERVAL / 3600.0
                flow_vph = bin_stats.total_flow / bin_hours if bin_hours > 0 else 0.0

                snapshot = {
                    "timestamp": elapsed,
                    "vehicles": int(bin_stats.total_flow),
                    "active_lanes": int(active_lanes),
                    "occupancy": float(avg_occ),
                    "crosswalk_blocked": float(avg_xw),
                    "flow_rate_vph": float(flow_vph),
                    "congestion_level": int(level),
                    "congestion_label": ["Optimal Flow", "Moderate", "Heavy Congestion"][level]
                }

                metrics_queue.put(snapshot)
                
                avg_fps = 1/np.mean(processing_times) if processing_times else 0
                print(f"[LIVE] t={elapsed:.1f}s | vehicles={bin_stats.total_flow} | occ={avg_occ*100:.1f}% | fps={avg_fps:.1f}")

                try:
                    from database import get_database
                    db = get_database()
                    if db and db.client:
                        db_session_id = db.save_realtime_metrics(
                            [snapshot],
                            stream_url,
                            "anonymous",
                            session_id=db_session_id
                        )
                except Exception as e:
                    print(f"[DB] ✗ Failed to save: {e}")

                bin_stats = TrafficBinStats()
                frames_in_bin = 0

            # NEW: Adaptive frame skip adjustment
            processing_time = time.time() - frame_start_time
            processing_times.append(processing_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            
            avg_processing = np.mean(processing_times)
            target_fps = fps / frame_skip
            
            if avg_processing > 1.0 / target_fps:
                frame_skip = min(frame_skip + 1, 5)
                print(f"[LIVE] Increasing frame skip to {frame_skip}")
            elif avg_processing < 0.5 / target_fps and frame_skip > 2:
                frame_skip = max(frame_skip - 1, 2)
                print(f"[LIVE] Decreasing frame skip to {frame_skip}")

    except Exception as e:
        print(f"[LIVE] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        print(f"[LIVE] Stream {stream_id} finished. DB session: {db_session_id}")

# -------------------- ENTRY --------------------
def process_media(input_path: str, output_path: str, kind: str, detect_w: str = "", seg_w: str = ""):
    if kind == "image":
        return process_image(input_path, output_path, detect_w, seg_w)
    elif kind == "video":
        return process_video(input_path, output_path, detect_w, seg_w)
    else:
        raise ValueError("Unsupported processing kind: must be 'image' or 'video'")
        

     