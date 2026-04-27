# Setup & Installation

## System Requirements

- Python 3.8+
- OpenCV 4.x
- CUDA 11.8+ (for GPU inference, optional but recommended)

## Installation

1. **Clone/setup project**
   ```bash
   cd /home/ashish-ratna/PMJ/Two-Man\ Rule
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv11-Pose model**
   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov11-pose.pt')"
   ```
   This downloads the model to ~/.yolo/

5. **Prepare reference image**
   Place a baseline image of the closed door at `assets/closed_ref.jpg`

## Configuration

Edit `config.py` and provide ROI coordinates:

```python
LOCK_A_ROI = [x, y, w, h]  # Rectangle for Lock A
LOCK_B_ROI = [x, y, w, h]  # Rectangle for Lock B
DOOR_ROI = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  # Polygon
INTERACTION_ZONE = [[x1, y1], [x2, y2], ...]  # Floor-level polygon
CLOSED_DOOR_REFERENCE = "assets/closed_ref.jpg"
```

## Running Tests

```bash
pytest tests/ -v
```

## Running the System

```bash
python main.py <video_file>
# or
python main.py 0  # For webcam
```

Press `q` to exit.

## Troubleshooting

- **Model not found**: Ensure YOLOv11-Pose is downloaded
- **No detections**: Check lighting and camera angle; ensure full body is visible
- **ROI not working**: Verify coordinates match frame dimensions (use visualization to debug)
