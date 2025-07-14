# Player Re-Identification & Tracking 📹⚽

This project performs **player detection, re-identification (Re-ID), and tracking** in football match videos using **YOLOv11** for object detection and **Deep SORT** for tracking with appearance-based embeddings.

---

## 📁 Project Structure
```
player-reid-project/
├── main.py # Main tracking pipeline
├── reid_utils.py # Drawing utility for track boxes
├── yolov11/
│ └── best.pt # Pre-trained YOLOv11 model weights
├── assets/
│ └── 15sec_input_720p.mp4 # Input video clip
├── output_tracked.mp4 # Output video (auto-generated)
├── README.md # Project documentation
└── requirements.txt # Dependencies
```

---

## 🛠️ Setup Instructions

1. **Clone or Download this Repository**
```
git clone https://github.com/yourname/player-reid-project.git
cd player-reid-project
```
2. Install Python Dependencies
Ensure Python ≥ 3.9 is installed. Then run:
```
pip install -r requirements.txt
```
## ▶️ How to Run the Tracker

`python main.py`
This will:

- Read the input video (assets/15sec_input_720p.mp4)
- Detect and track players using YOLOv11 + Deep SORT
- Display a live tracking preview (with bounding boxes and track IDs)
- Save the result to output_tracked.mp4
- Log detections and track IDs frame-by-frame to the terminal
- To stop early, press q during live preview.

---
