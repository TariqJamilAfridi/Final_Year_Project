#!/usr/bin/env python3
"""
Test script for standing/lying segmentation model.
Usage: python standing_lying_V1.py <video_path> [output_video_path] [confidence_threshold]
"""

import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# 1. Load model
# -------------------------------
MODEL_PATH = "standing_lying.pt"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    sys.exit(1)

print(f"Loading model {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print(f"✅ Model loaded. Classes: {model.names}")

# -------------------------------
# 2. Parse arguments
# -------------------------------
if len(sys.argv) < 2:
    print("\nUsage:")
    print("  python standing_lying_V1.py <video_path> [output_video] [confidence]")
    print("  output_video: optional, e.g., output.mp4")
    print("  confidence: detection threshold (default 0.5)")
    sys.exit(1)

video_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else None
conf_thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

if not os.path.exists(video_path):
    print(f"❌ Video not found: {video_path}")
    sys.exit(1)

# -------------------------------
# 3. Open video and prepare output
# -------------------------------
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = None
if output_path:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"📁 Output video will be saved to {output_path}")

frame_count = 0
print("Processing video. Press 'q' to quit, 's' to save current frame.")

# -------------------------------
# 4. Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Run inference
    results = model(frame, conf=conf_thresh)[0]

    # Create an overlay for masks
    overlay = frame.copy()
    mask_combined = np.zeros((height, width), dtype=np.uint8)

    # Check if there are any detections
    if results.boxes is not None and len(results.boxes) > 0:
        # Get boxes, masks, classes, confidences
        boxes = results.boxes.xyxy.cpu().numpy()
        masks = results.masks.data.cpu().numpy() if results.masks is not None else None
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
            class_id = int(cls)
            class_name = model.names[class_id]
            confidence = float(conf)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            # Choose colour for class
            if class_name.lower() == 'standing':
                color = (0, 255, 0)   # green
            elif class_name.lower() == 'lying':
                color = (0, 0, 255)   # red
            else:
                color = (255, 0, 0)   # blue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # If masks exist, draw mask overlay
            if masks is not None and i < len(masks):
                mask_np = masks[i] * 255
                mask_np = mask_np.astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
                overlay[mask_resized > 0] = color

    # Blend overlay with original frame (semi‑transparent)
    blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    # Add frame counter
    cv2.putText(blended, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Show the result
    cv2.imshow("Standing / Lying Segmentation", blended)

    # Write to output if needed
    if out:
        out.write(blended)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('s'):
        cv2.imwrite(f"frame_{frame_count}.jpg", blended)
        print(f"Saved frame {frame_count} as frame_{frame_count}.jpg")

# -------------------------------
# 5. Cleanup
# -------------------------------
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
print(f"✅ Processing complete. {frame_count} frames processed.")