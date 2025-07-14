import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from reid_utils import draw_tracks
import os

YOLO_MODEL_PATH = os.path.join("yolov11", "best.pt")
VIDEO_PATH = "assets/15sec_input_720p.mp4"
OUTPUT_PATH = "output_tracked.mp4"
PLAYER_CLASS_ID = 0  # assuming class 0 is player
CONF_THRESHOLD = 0.3

#loading
model = YOLO(YOLO_MODEL_PATH)

#initialize deepsort tracker
tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_iou_distance=0.7,
    embedder="mobilenet",  #for lightweight appearance
)

#setup - video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_idx = 0
print(f"\n Started tracking on: '{VIDEO_PATH}'\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    #extract players from detections
    for box in results.boxes:
        if int(box.cls[0]) == PLAYER_CLASS_ID:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())

            if conf < CONF_THRESHOLD:
                continue

            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            detections.append(([xc, yc, bw, bh], conf))

    #tracking players
    if detections:
        tracks = tracker.update_tracks(detections, frame=frame)
    else:
        tracks = []

    #draw and show logs
    frame = draw_tracks(frame, tracks)
    active_ids = [t.track_id for t in tracks if t.is_confirmed()]
    print(f"[Frame {frame_idx}] Players Detected: {len(detections)} | Active Tracks: {active_ids}")
    frame_idx += 1

    #live tracking
    cv2.putText(frame, f"Tracked: {len(active_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Tracking Output", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped early by user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n Successfully tracked and saved video to: '{OUTPUT_PATH}'")
