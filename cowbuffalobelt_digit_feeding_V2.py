#!/usr/bin/env python3
"""
Full Pipeline: Animal + Belt + Digit + Feeding (with masks)
- Belt numbers assigned directly based on digit group containment in animal box.
- Unique belt assignment per farm.
- Graceful Ctrl+C to save CSVs.
"""
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque, Counter
import csv
from datetime import datetime
import signal

# ===========================
#  CONFIGURATION
# ===========================
ANIMAL_MODEL_PATH = "cowbuffalobelt.pt"
DIGIT_MODEL_PATH = "final_digit_model.pt"
FEEDING_MODEL_PATH = "feeding_model.pt"

CONF_ANIMAL = 0.25
IOU_ANIMAL = 0.4
CONF_DIGIT = 0.25
DIGIT_NMS_IOU = 0.45
DIGIT_MODE = "auto"
DIGIT_USE_TTA = False
CONF_FEED = 0.5

STABLE_FRAMES = 6
STABILITY_THRESHOLD = 0.5
CSV_FILE = "animal_ids_log.csv"
FEEDING_EVENTS_CSV = "feeding_events_log.csv"
FEEDING_SUMMARY_CSV = "feeding_summary.csv"
SCALE_FACTOR = 1.0

ANIMAL_FONT_SCALE = 0.7
DIGIT_FONT_SCALE = 0.6
CROP_DISPLAY_MAX = 800

DEBUG = True
MAX_MISSED_FRAMES = 30

# Distance thresholds for feeding association (not used for digit assignment now)
FEEDING_TO_ANIMAL_MAX_DIST = 120

used_belt_numbers = set()

# Global for graceful interrupt
cap = None
out = None
frame_count = 0
fps = 30

def signal_handler(sig, frame):
    print("\n\n⚠️ KeyboardInterrupt received. Saving data before exit...")
    if cap is not None:
        cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    generate_feeding_summary()
    print("✅ Partial data saved. Exiting.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ===========================
#  DIGIT DETECTION FUNCTIONS (unchanged)
# ===========================
def gamma_correction(img, gamma=1.5):
    lut = np.array([(i / 255.0) ** (1.0/gamma) * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(img, lut)

def clahe_enhance(img, clip_limit=2.0, grid_size=(8,8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def sharpen(img, strength=1.0):
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]]) * strength
    return cv2.filter2D(img, -1, kernel)

def preprocess(img, mode="auto"):
    if mode == "normal":
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if mode == "auto":
        if mean_brightness < 80:
            mode = "dark"
        elif blur_score < 100:
            mode = "blurry"
        else:
            mode = "normal"
    if mode == "dark":
        enhanced = gamma_correction(img, gamma=1.8)
        enhanced = clahe_enhance(enhanced, clip_limit=3.0)
        return enhanced
    elif mode == "blurry":
        enhanced = sharpen(img, strength=1.2)
        enhanced = clahe_enhance(enhanced, clip_limit=2.0)
        return enhanced
    return img

def tta_predict(model, img, conf_thresh):
    res0 = model(img, conf=conf_thresh)[0]
    img_flip = cv2.flip(img, 1)
    res1 = model(img_flip, conf=conf_thresh)[0]
    all_boxes = []
    h, w = img.shape[:2]
    for box in res0.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        all_boxes.append((x1, y1, x2, y2, cls, conf))
    for box in res1.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        nx1 = w - x2
        nx2 = w - x1
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        all_boxes.append((nx1, y1, nx2, y2, cls, conf))
    final = []
    used = [False] * len(all_boxes)
    for i, (x1, y1, x2, y2, cls, conf) in enumerate(all_boxes):
        if used[i]:
            continue
        group = [(cls, conf)]
        used[i] = True
        cx_i = (x1 + x2) // 2
        cy_i = (y1 + y2) // 2
        for j, (xx1, yy1, xx2, yy2, c2, conf2) in enumerate(all_boxes):
            if used[j]:
                continue
            cx_j = (xx1 + xx2) // 2
            cy_j = (yy1 + yy2) // 2
            dist = ((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2) ** 0.5
            if dist < 30:
                group.append((c2, conf2))
                used[j] = True
        votes = defaultdict(list)
        for c, cf in group:
            votes[c].append(cf)
        best_class = max(votes.items(), key=lambda item: np.mean(item[1]))[0]
        avg_conf = np.mean(votes[best_class])
        final.append((x1, y1, x2, y2, best_class, avg_conf))
    return final

def nms_merge(boxes, iou_threshold=DIGIT_NMS_IOU):
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        to_remove = []
        for i, box in enumerate(boxes):
            x1 = max(best[0], box[0])
            y1 = max(best[1], box[1])
            x2 = min(best[2], box[2])
            y2 = min(best[3], box[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            if inter_area == 0:
                continue
            best_area = (best[2] - best[0]) * (best[3] - best[1])
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            iou = inter_area / (best_area + box_area - inter_area)
            if iou > iou_threshold:
                to_remove.append(i)
        for idx in reversed(to_remove):
            boxes.pop(idx)
    return keep

def draw_digit_detections(image, detections, font_scale=DIGIT_FONT_SCALE, thickness=1, base_offset=15):
    boxes_info = []
    for (x1, y1, x2, y2, digit, conf) in detections:
        label = f"{digit}:{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        boxes_info.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'label': label, 'tw': tw, 'th': th,
            'digit': digit, 'conf': conf
        })
    boxes_info.sort(key=lambda b: b['y1'])
    placed = []
    for b in boxes_info:
        label_y_above = b['y1'] - base_offset
        label_y_below = b['y2'] + base_offset + b['th']
        above_ok = (label_y_above > b['th'] + 5)
        if above_ok:
            for py, ph in placed:
                if abs(label_y_above - py) < (b['th'] + ph) // 2:
                    above_ok = False
                    break
        if above_ok:
            b['label_y'] = label_y_above
            placed.append((label_y_above, b['th']))
        else:
            label_y = label_y_below
            collision = True
            attempts = 0
            while collision and attempts < 5:
                collision = False
                for py, ph in placed:
                    if abs(label_y - py) < (b['th'] + ph) // 2:
                        collision = True
                        label_y += base_offset
                        break
                attempts += 1
            b['label_y'] = label_y
            placed.append((label_y, b['th']))
        cv2.rectangle(image, (b['x1'], b['y1']), (b['x2'], b['y2']), (0,255,0), 2)
        tx, ty = b['x1'], b['label_y']
        cv2.rectangle(image, (tx-2, ty - b['th'] - 2), (tx + b['tw'] + 2, ty + 2), (0,0,0), -1)
        cv2.putText(image, b['label'], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)
    return image

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Frame", "Animal", "Belt Number", "Feeding", "Confidence"])
    if not os.path.exists(FEEDING_EVENTS_CSV):
        with open(FEEDING_EVENTS_CSV, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Animal_ID", "Animal_Type", "Belt_Number", "Start_Frame", "End_Frame",
                             "Start_Time_s", "End_Time_s", "Duration_s"])

def log_animal_id(frame_num, animal_type, belt_number, feeding, avg_conf):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        belt_str = f"{int(belt_number):03d}"
    except:
        belt_str = str(belt_number) if belt_number else "None"
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, frame_num, animal_type, belt_str,
                         "Feeding" if feeding else "Not Feeding",
                         f"{avg_conf:.2f}"])
    print(f"📝 Assigned belt {belt_str} to {animal_type} (feeding={feeding}) at frame {frame_num}")

def log_feeding_event(animal_id, animal_type, belt_number, start_frame, end_frame, start_time_s, end_time_s, duration_s):
    try:
        belt_str = f"{int(belt_number):03d}" if belt_number and belt_number != "None" else "None"
    except:
        belt_str = str(belt_number) if belt_number else "None"
    with open(FEEDING_EVENTS_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([animal_id, animal_type, belt_str,
                         start_frame, end_frame, f"{start_time_s:.2f}", f"{end_time_s:.2f}", f"{duration_s:.2f}"])
    print(f"🌾 Feeding bout logged: Animal {animal_id} ({animal_type}) fed for {duration_s:.2f} seconds")

def generate_feeding_summary():
    if not os.path.exists(FEEDING_EVENTS_CSV):
        print("No feeding events file found. Skipping summary.")
        return
    summary_by_id = defaultdict(lambda: {"total_seconds": 0.0, "num_bouts": 0, "type": "", "belt": ""})
    summary_by_belt = defaultdict(lambda: {"total_seconds": 0.0, "num_bouts": 0, "type": "", "animal_id": ""})
    with open(FEEDING_EVENTS_CSV, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                duration = float(row["Duration_s"])
            except:
                continue
            aid = row["Animal_ID"]
            atype = row["Animal_Type"]
            belt = row["Belt_Number"]
            summary_by_id[aid]["total_seconds"] += duration
            summary_by_id[aid]["num_bouts"] += 1
            summary_by_id[aid]["type"] = atype
            if belt != "None":
                summary_by_id[aid]["belt"] = belt
            if belt != "None":
                summary_by_belt[belt]["total_seconds"] += duration
                summary_by_belt[belt]["num_bouts"] += 1
                summary_by_belt[belt]["type"] = atype
                summary_by_belt[belt]["animal_id"] = aid
    with open(FEEDING_SUMMARY_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Animal_ID", "Animal_Type", "Belt_Number", "Total_Feeding_Seconds", "Number_of_Bouts"])
        for aid, data in sorted(summary_by_id.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            writer.writerow([aid, data["type"], data["belt"], f"{data['total_seconds']:.2f}", data["num_bouts"]])
        writer.writerow([])
        writer.writerow(["=== SUMMARY BY BELT NUMBER ==="])
        writer.writerow(["Belt_Number", "Animal_Type", "Animal_ID", "Total_Feeding_Seconds", "Number_of_Bouts"])
        for belt, data in sorted(summary_by_belt.items(), key=lambda x: x[0]):
            writer.writerow([belt, data["type"], data["animal_id"], f"{data['total_seconds']:.2f}", data["num_bouts"]])
    print(f"\n✅ Feeding summary saved to {FEEDING_SUMMARY_CSV}")
    print("\n=== Feeding Summary (by Animal ID) ===")
    for aid, data in summary_by_id.items():
        print(f"Animal {aid} ({data['type']}) - Belt {data['belt']}: {data['total_seconds']:.2f} seconds in {data['num_bouts']} bouts")
    print("\n=== Feeding Summary (by Belt Number) ===")
    for belt, data in summary_by_belt.items():
        print(f"Belt {belt} ({data['type']} ID {data['animal_id']}): {data['total_seconds']:.2f} seconds in {data['num_bouts']} bouts")

def select_crop_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    orig_h, orig_w = frame.shape[:2]
    scale = min(CROP_DISPLAY_MAX / orig_w, CROP_DISPLAY_MAX / orig_h, 1.0)
    if scale < 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display_frame = frame.copy()
        scale = 1.0
    print(f"Original: {orig_w}x{orig_h}, Display: {display_frame.shape[1]}x{display_frame.shape[0]}")
    roi = cv2.selectROI("Select Crop", display_frame, showCrosshair=True)
    cv2.destroyWindow("Select Crop")
    if roi == (0,0,0,0):
        return None
    x, y, w, h = roi
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)
    x = max(0, min(x, orig_w-1))
    y = max(0, min(y, orig_h-1))
    w = min(w, orig_w - x)
    h = min(h, orig_h - y)
    return (x, y, w, h)

# ===========================
#  MAIN
# ===========================
def main():
    global cap, out, frame_count, fps
    global used_belt_numbers

    if len(sys.argv) < 2:
        print("Usage: python full_pipeline.py <input_path> [output_name] [conf_animal] [conf_digit] [conf_feed] [mode] [use_tta] [scale] [crop]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else "output_full.mp4"
    conf_animal = float(sys.argv[3]) if len(sys.argv) > 3 else CONF_ANIMAL
    conf_digit = float(sys.argv[4]) if len(sys.argv) > 4 else CONF_DIGIT
    conf_feed = float(sys.argv[5]) if len(sys.argv) > 5 else CONF_FEED
    mode = sys.argv[6] if len(sys.argv) > 6 else DIGIT_MODE
    use_tta = bool(int(sys.argv[7])) if len(sys.argv) > 7 else DIGIT_USE_TTA
    scale = float(sys.argv[8]) if len(sys.argv) > 8 else SCALE_FACTOR
    do_crop = bool(int(sys.argv[9])) if len(sys.argv) > 9 else False

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    # Load models
    if not os.path.exists(ANIMAL_MODEL_PATH):
        print(f"❌ Animal model not found: {ANIMAL_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(DIGIT_MODEL_PATH):
        print(f"❌ Digit model not found: {DIGIT_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(FEEDING_MODEL_PATH):
        print(f"❌ Feeding model not found: {FEEDING_MODEL_PATH}")
        sys.exit(1)

    model_animal = YOLO(ANIMAL_MODEL_PATH)
    model_digit = YOLO(DIGIT_MODEL_PATH)
    model_feed = YOLO(FEEDING_MODEL_PATH)

    print(f"Animal classes: {model_animal.names}")
    print(f"Digit classes: {model_digit.names}")
    print(f"Feeding classes: {model_feed.names}")

    is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        crop_roi = None
        if do_crop:
            crop_roi = select_crop_roi(input_path)
            if crop_roi:
                print(f"Crop: {crop_roi}")
            else:
                print("No crop selected")
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        if crop_roi:
            _, _, out_w, out_h = crop_roi
        else:
            out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
        frame_count = 0
        init_csv()

        persistent_animals = {}
        next_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if crop_roi:
                x, y, w, h = crop_roi
                frame = frame[y:y+h, x:x+w]

            clean_frame_digit = frame.copy()
            clean_frame_feed = frame.copy()

            # Animal + belt detection (belt boxes only for drawing, not used for assignment)
            results_animal = model_animal(frame, conf=conf_animal, iou=IOU_ANIMAL)[0]
            belt_boxes = []
            animal_boxes = []
            for box in results_animal.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    belt_boxes.append(box)
                elif cls == 1:
                    animal_boxes.append((box, "buffalo"))
                elif cls == 2:
                    animal_boxes.append((box, "cow"))

            # Feeding detection
            feed_results = model_feed(clean_frame_feed, conf=conf_feed)[0]
            feeding_detections = []
            if feed_results.boxes is not None and len(feed_results.boxes) > 0:
                boxes = feed_results.boxes.xyxy.cpu().numpy()
                classes = feed_results.boxes.cls.cpu().numpy()
                confs = feed_results.boxes.conf.cpu().numpy()
                for box, cls, conf in zip(boxes, classes, confs):
                    class_name = model_feed.names[int(cls)].lower()
                    is_feeding = 'down' in class_name
                    x1, y1, x2, y2 = map(int, box[:4])
                    feeding_detections.append((x1, y1, x2, y2, is_feeding, conf))

            # Draw feeding boxes on main frame
            for (x1, y1, x2, y2, is_feeding, conf) in feeding_detections:
                if is_feeding:
                    color = (255, 0, 0)
                    label = f"Feeding {conf:.2f}"
                else:
                    color = (0, 255, 255)
                    label = f"Not Feeding {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Digit detection
            digit_frame = clean_frame_digit.copy()
            if scale != 1.0:
                new_w = int(out_w * scale)
                new_h = int(out_h * scale)
                digit_frame = cv2.resize(digit_frame, (new_w, new_h))
            enhanced = preprocess(digit_frame, mode)
            if use_tta:
                digit_detections = tta_predict(model_digit, enhanced, conf_digit)
            else:
                res = model_digit(enhanced, conf=conf_digit)[0]
                digit_detections = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]),
                                     int(b.cls[0]), float(b.conf[0])) for b in res.boxes]
            digit_detections = nms_merge(digit_detections)
            if scale != 1.0:
                mapped = []
                for (x1, y1, x2, y2, cls, conf) in digit_detections:
                    ox1 = int(x1 / scale)
                    oy1 = int(y1 / scale)
                    ox2 = int(x2 / scale)
                    oy2 = int(y2 / scale)
                    mapped.append((ox1, oy1, ox2, oy2, cls, conf))
                digit_detections = mapped

            # Group digits
            used = [False] * len(digit_detections)
            groups = []
            for i, d1 in enumerate(digit_detections):
                if used[i]:
                    continue
                group = [d1]
                used[i] = True
                cx1 = (d1[0] + d1[2]) // 2
                cy1 = (d1[1] + d1[3]) // 2
                for j, d2 in enumerate(digit_detections):
                    if used[j]:
                        continue
                    cx2 = (d2[0] + d2[2]) // 2
                    cy2 = (d2[1] + d2[3]) // 2
                    dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                    if dist < 50:
                        group.append(d2)
                        used[j] = True
                if len(group) == 3:
                    groups.append(group)

            # ========== NEW: Direct assignment of digit groups to animals ==========
            current_assignments = {}  # animal_idx -> (belt_number, avg_conf)

            for group in groups:
                # overall bounding box of the digit group
                gx1 = min(d[0] for d in group)
                gy1 = min(d[1] for d in group)
                gx2 = max(d[2] for d in group)
                gy2 = max(d[3] for d in group)
                group_center = ((gx1+gx2)//2, (gy1+gy2)//2)

                # sort digits to form correct order (by x or by y based on orientation)
                xs = [(d[0]+d[2])//2 for d in group]
                ys = [(d[1]+d[3])//2 for d in group]
                if max(ys)-min(ys) > max(xs)-min(xs):
                    group.sort(key=lambda d: (d[1]+d[3])//2)   # vertical arrangement
                else:
                    group.sort(key=lambda d: (d[0]+d[2])//2)   # horizontal arrangement

                number_str = "".join(str(d[4]) for d in group)
                avg_conf = sum(d[5] for d in group) / 3

                # Find which animal contains this digit group's center (or overlaps)
                best_animal_idx = None
                best_overlap = 0.0
                for ani_idx, (ani_box, ani_type) in enumerate(animal_boxes):
                    ax1, ay1, ax2, ay2 = map(int, ani_box.xyxy[0])
                    # Check if center inside animal box
                    if ax1 <= group_center[0] <= ax2 and ay1 <= group_center[1] <= ay2:
                        best_animal_idx = ani_idx
                        break   # first animal that contains the center wins
                    else:
                        # calculate overlap area between digit group and animal box
                        inter_x1 = max(gx1, ax1)
                        inter_y1 = max(gy1, ay1)
                        inter_x2 = min(gx2, ax2)
                        inter_y2 = min(gy2, ay2)
                        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                            group_area = (gx2 - gx1) * (gy2 - gy1)
                            overlap_ratio = inter_area / group_area
                            if overlap_ratio > best_overlap:
                                best_overlap = overlap_ratio
                                best_animal_idx = ani_idx

                if best_animal_idx is not None:
                    # Check if belt number already used globally
                    if number_str in used_belt_numbers:
                        if DEBUG:
                            print(f"Frame {frame_count}: belt {number_str} already assigned to another animal. Skipping.")
                    else:
                        current_assignments[best_animal_idx] = (number_str, avg_conf)
                        if DEBUG:
                            print(f"Frame {frame_count}: digit group {number_str} assigned directly to animal {best_animal_idx} (containment/overlap {best_overlap:.2f})")
                else:
                    if DEBUG:
                        print(f"Frame {frame_count}: digit group {number_str} not inside any animal box.")

            # ========== END of direct assignment ==========

            # Associate feeding status to animals (unchanged)
            animal_feeding = {}
            for ani_idx, (ani_box, ani_type) in enumerate(animal_boxes):
                ax1, ay1, ax2, ay2 = map(int, ani_box.xyxy[0])
                ani_center = ((ax1+ax2)//2, (ay1+ay2)//2)
                min_dist = FEEDING_TO_ANIMAL_MAX_DIST
                feeding_status = False
                for (fx1, fy1, fx2, fy2, is_feeding, _) in feeding_detections:
                    f_center = ((fx1+fx2)//2, (fy1+fy2)//2)
                    dist = ((ani_center[0]-f_center[0])**2 + (ani_center[1]-f_center[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        feeding_status = is_feeding
                if min_dist < FEEDING_TO_ANIMAL_MAX_DIST:
                    animal_feeding[ani_idx] = feeding_status
                else:
                    animal_feeding[ani_idx] = False

            # Track animals across frames (unchanged)
            current_animals = []
            for ani_idx, (ani_box, ani_type) in enumerate(animal_boxes):
                ax1, ay1, ax2, ay2 = map(int, ani_box.xyxy[0])
                center = ((ax1+ax2)//2, (ay1+ay2)//2)
                current_animals.append((ani_idx, center, ani_type))

            # Remove old persistent animals
            to_delete = []
            for p_id, p_info in persistent_animals.items():
                if frame_count - p_info['last_seen'] > MAX_MISSED_FRAMES:
                    if p_info.get('feeding_status', False):
                        end_time_s = frame_count / fps
                        start_frame = p_info['feeding_start_frame']
                        start_time_s = p_info['feeding_start_time']
                        duration = end_time_s - start_time_s
                        log_feeding_event(p_id, p_info['type'],
                                          p_info.get('current_belt'),
                                          start_frame, frame_count,
                                          start_time_s, end_time_s, duration)
                    to_delete.append(p_id)
            for p_id in to_delete:
                del persistent_animals[p_id]

            # Match current animals to persistent ones
            matched = {}
            used_persistent = set()
            for cur_idx, (_, cur_center, cur_type) in enumerate(current_animals):
                best_dist = 100
                best_id = None
                for p_id, p_info in persistent_animals.items():
                    if p_info['type'] != cur_type:
                        continue
                    if p_id in used_persistent:
                        continue
                    p_center = p_info['center']
                    dist = ((cur_center[0]-p_center[0])**2 + (cur_center[1]-p_center[1])**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_id = p_id
                if best_id is not None and best_dist < 80:
                    matched[cur_idx] = best_id
                    used_persistent.add(best_id)
                    persistent_animals[best_id]['center'] = cur_center
                    persistent_animals[best_id]['last_seen'] = frame_count
                else:
                    new_id = next_id
                    next_id += 1
                    matched[cur_idx] = new_id
                    persistent_animals[new_id] = {
                        'center': cur_center,
                        'type': cur_type,
                        'buffer': deque(maxlen=STABLE_FRAMES),
                        'current_belt': None,
                        'last_seen': frame_count,
                        'feeding_status': False,
                        'feeding_start_frame': None,
                        'feeding_start_time': None,
                    }

            # Update buffers and log stable IDs
            for cur_idx, p_id in matched.items():
                if cur_idx in current_assignments:
                    belt_number, avg_conf = current_assignments[cur_idx]
                    if belt_number in used_belt_numbers:
                        continue
                    persistent_animals[p_id]['buffer'].append(belt_number)
                    buf = persistent_animals[p_id]['buffer']
                    if len(buf) == STABLE_FRAMES:
                        counter = Counter(buf)
                        most_common = counter.most_common(1)[0]
                        if most_common[1] >= STABILITY_THRESHOLD * STABLE_FRAMES:
                            stable_number = most_common[0]
                            if persistent_animals[p_id]['current_belt'] != stable_number:
                                feeding = animal_feeding.get(cur_idx, False)
                                log_animal_id(frame_count, persistent_animals[p_id]['type'], stable_number, feeding, avg_conf)
                                persistent_animals[p_id]['current_belt'] = stable_number
                                used_belt_numbers.add(stable_number)

            # Feeding bout tracking
            for p_id, p_info in persistent_animals.items():
                if p_id in matched.values():
                    cur_idx = None
                    for cidx, pid in matched.items():
                        if pid == p_id:
                            cur_idx = cidx
                            break
                    if cur_idx is not None:
                        current_feeding = animal_feeding.get(cur_idx, False)
                    else:
                        current_feeding = False
                else:
                    current_feeding = False
                prev_feeding = p_info.get('feeding_status', False)
                if not prev_feeding and current_feeding:
                    p_info['feeding_start_frame'] = frame_count
                    p_info['feeding_start_time'] = frame_count / fps
                    p_info['feeding_status'] = True
                    if DEBUG:
                        print(f"🌾 Animal {p_id} ({p_info['type']}) started feeding at frame {frame_count}")
                elif prev_feeding and not current_feeding:
                    end_time_s = frame_count / fps
                    start_frame = p_info['feeding_start_frame']
                    start_time_s = p_info['feeding_start_time']
                    duration = end_time_s - start_time_s
                    belt_num = p_info.get('current_belt')
                    log_feeding_event(p_id, p_info['type'], belt_num,
                                      start_frame, frame_count,
                                      start_time_s, end_time_s, duration)
                    p_info['feeding_status'] = False
                    p_info['feeding_start_frame'] = None
                    p_info['feeding_start_time'] = None

            # Draw digit detections on frame
            frame = draw_digit_detections(frame, digit_detections)

            # Draw belt boxes (optional, for visualization)
            for belt in belt_boxes:
                x1, y1, x2, y2 = map(int, belt.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, "Belt", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # Draw animals with assigned ID and feeding status
            for cur_idx, (ani_box, ani_type) in enumerate(animal_boxes):
                x1, y1, x2, y2 = map(int, ani_box.xyxy[0])
                conf = float(ani_box.conf[0])
                if ani_type == "cow":
                    color = (0,255,0)
                    base_label = f"Cow {conf:.2f}"
                else:
                    color = (255,0,0)
                    base_label = f"Buffalo {conf:.2f}"
                p_id = matched.get(cur_idx)
                if p_id is not None and persistent_animals[p_id]['current_belt'] is not None:
                    belt_number = persistent_animals[p_id]['current_belt']
                    base_label = f"{base_label} ID:{belt_number}"
                feeding = animal_feeding.get(cur_idx, False)
                feeding_text = " Feeding" if feeding else ""
                label = f"{base_label}{feeding_text}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, ANIMAL_FONT_SCALE, color, 2)

            cv2.putText(frame, f"Frame: {frame_count}", (out_w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            out.write(frame)
            cv2.imshow('Full Pipeline', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        # End of video: finalize ongoing feeding bouts
        for p_id, p_info in persistent_animals.items():
            if p_info.get('feeding_status', False):
                end_time_s = frame_count / fps
                start_frame = p_info['feeding_start_frame']
                start_time_s = p_info['feeding_start_time']
                duration = end_time_s - start_time_s
                belt_num = p_info.get('current_belt')
                log_feeding_event(p_id, p_info['type'], belt_num,
                                  start_frame, frame_count,
                                  start_time_s, end_time_s, duration)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Video saved to {output_name} ({frame_count} frames)")

        generate_feeding_summary()

    else:
        # Image mode – no bout tracking, but use same direct assignment logic for IDs
        img = cv2.imread(input_path)
        if img is None:
            print("Could not read image")
            sys.exit(1)

        results_animal = model_animal(img, conf=conf_animal, iou=IOU_ANIMAL)[0]
        belt_boxes = []
        animal_boxes = []
        for box in results_animal.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                belt_boxes.append(box)
            elif cls == 1:
                animal_boxes.append((box, "buffalo"))
            elif cls == 2:
                animal_boxes.append((box, "cow"))

        # Feeding detection
        clean_img_feed = img.copy()
        feed_results = model_feed(clean_img_feed, conf=conf_feed)[0]
        feeding_detections = []
        if feed_results.boxes is not None and len(feed_results.boxes) > 0:
            boxes = feed_results.boxes.xyxy.cpu().numpy()
            classes = feed_results.boxes.cls.cpu().numpy()
            confs = feed_results.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                class_name = model_feed.names[int(cls)].lower()
                is_feeding = 'down' in class_name
                x1, y1, x2, y2 = map(int, box[:4])
                feeding_detections.append((x1, y1, x2, y2, is_feeding, conf))

        for (x1, y1, x2, y2, is_feeding, conf) in feeding_detections:
            if is_feeding:
                color = (255, 0, 0)
                label = f"Feeding {conf:.2f}"
            else:
                color = (0, 255, 255)
                label = f"Not Feeding {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Digit detection
        clean_img_digit = cv2.imread(input_path)
        digit_img = clean_img_digit.copy()
        if scale != 1.0:
            h, w = digit_img.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            digit_img = cv2.resize(digit_img, (new_w, new_h))
        enhanced = preprocess(digit_img, mode)
        if use_tta:
            digit_detections = tta_predict(model_digit, enhanced, conf_digit)
        else:
            res = model_digit(enhanced, conf=conf_digit)[0]
            digit_detections = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]),
                                 int(b.cls[0]), float(b.conf[0])) for b in res.boxes]
        digit_detections = nms_merge(digit_detections)
        if scale != 1.0:
            mapped = []
            for (x1, y1, x2, y2, cls, conf) in digit_detections:
                ox1 = int(x1 / scale)
                oy1 = int(y1 / scale)
                ox2 = int(x2 / scale)
                oy2 = int(y2 / scale)
                mapped.append((ox1, oy1, ox2, oy2, cls, conf))
            digit_detections = mapped

        # Group digits
        used = [False] * len(digit_detections)
        groups = []
        for i, d1 in enumerate(digit_detections):
            if used[i]:
                continue
            group = [d1]
            used[i] = True
            cx1 = (d1[0] + d1[2]) // 2
            cy1 = (d1[1] + d1[3]) // 2
            for j, d2 in enumerate(digit_detections):
                if used[j]:
                    continue
                cx2 = (d2[0] + d2[2]) // 2
                cy2 = (d2[1] + d2[3]) // 2
                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                if dist < 50:
                    group.append(d2)
                    used[j] = True
            if len(group) == 3:
                groups.append(group)

        # Direct assignment for image mode
        assignments = {}
        for group in groups:
            gx1 = min(d[0] for d in group)
            gy1 = min(d[1] for d in group)
            gx2 = max(d[2] for d in group)
            gy2 = max(d[3] for d in group)
            group_center = ((gx1+gx2)//2, (gy1+gy2)//2)
            xs = [(d[0]+d[2])//2 for d in group]
            ys = [(d[1]+d[3])//2 for d in group]
            if max(ys)-min(ys) > max(xs)-min(xs):
                group.sort(key=lambda d: (d[1]+d[3])//2)
            else:
                group.sort(key=lambda d: (d[0]+d[2])//2)
            number_str = "".join(str(d[4]) for d in group)
            best_animal_idx = None
            best_overlap = 0.0
            for ani_idx, (ani_box, ani_type) in enumerate(animal_boxes):
                ax1, ay1, ax2, ay2 = map(int, ani_box.xyxy[0])
                if ax1 <= group_center[0] <= ax2 and ay1 <= group_center[1] <= ay2:
                    best_animal_idx = ani_idx
                    break
                else:
                    inter_x1 = max(gx1, ax1)
                    inter_y1 = max(gy1, ay1)
                    inter_x2 = min(gx2, ax2)
                    inter_y2 = min(gy2, ay2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        group_area = (gx2 - gx1) * (gy2 - gy1)
                        overlap_ratio = inter_area / group_area
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_animal_idx = ani_idx
            if best_animal_idx is not None:
                assignments[best_animal_idx] = number_str

        # Associate feeding to animals
        animal_feeding = {}
        for ani_idx, (ani_box, ani_type) in enumerate(animal_boxes):
            ax1, ay1, ax2, ay2 = map(int, ani_box.xyxy[0])
            ani_center = ((ax1+ax2)//2, (ay1+ay2)//2)
            min_dist = FEEDING_TO_ANIMAL_MAX_DIST
            feeding_status = False
            for (fx1, fy1, fx2, fy2, is_feeding, _) in feeding_detections:
                f_center = ((fx1+fx2)//2, (fy1+fy2)//2)
                dist = ((ani_center[0]-f_center[0])**2 + (ani_center[1]-f_center[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    feeding_status = is_feeding
            animal_feeding[ani_idx] = feeding_status if min_dist < FEEDING_TO_ANIMAL_MAX_DIST else False

        # Draw
        img = draw_digit_detections(img, digit_detections)
        for idx, (ani_box, ani_type) in enumerate(animal_boxes):
            x1, y1, x2, y2 = map(int, ani_box.xyxy[0])
            if ani_type == "cow":
                color = (0,255,0)
                base_label = f"Cow"
            else:
                color = (255,0,0)
                base_label = f"Buffalo"
            if idx in assignments:
                belt_number = assignments[idx]
                base_label = f"{base_label} ID:{belt_number}"
            feeding = animal_feeding.get(idx, False)
            feeding_text = " Feeding" if feeding else ""
            label = f"{base_label}{feeding_text}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, ANIMAL_FONT_SCALE, color, 2)

        for belt in belt_boxes:
            x1, y1, x2, y2 = map(int, belt.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(img, "Belt", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        out_name = "output_" + os.path.basename(input_path)
        cv2.imwrite(out_name, img)
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"✅ Saved {out_name}")

if __name__ == "__main__":
    main()