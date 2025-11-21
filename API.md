# API Documentation

## Core Modules

### Detector (`src/detector.py`)

YOLOv8-based people detector with batched inference support.

#### Class: `Detector`

```python
from src.detector import Detector

detector = Detector(
    model_path="yolov8n.pt",
    conf_threshold=0.5,
    nms_threshold=0.45,
    device="cuda"
)
```

**Methods:**

- `detect(frame, return_format='xyxy')` - Detect people in a single frame
  - Returns: `(boxes, scores, class_ids)`
  - `boxes`: numpy array of shape (N, 4) in format [x, y, w, h] or [x1, y1, x2, y2]
  - `scores`: confidence scores
  - `class_ids`: class identifiers

- `detect_batch(frames)` - Batch detection for multiple frames
  - Returns: List of (boxes, scores, class_ids) tuples

- `export_onnx(output_path)` - Export model to ONNX format

---

### Tracker (`src/tracker.py`)

Multi-object tracker with StrongSORT and ByteTrack support.

#### Class: `Tracker`

```python
from src.tracker import Tracker

tracker = Tracker(
    tracker_type="strongsort",  # or "bytetrack"
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)
```

**Methods:**

- `update(detections, scores, frame=None)` - Update tracks with new detections
  - `detections`: List of bboxes [x, y, w, h]
  - `scores`: List of confidence scores
  - `frame`: Optional frame for appearance features
  - Returns: List of `Track` objects

#### Class: `Track`

**Attributes:**
- `track_id`: Unique track identifier
- `bbox`: Bounding box [x, y, w, h]
- `confidence`: Detection confidence
- `state`: Track state ('tentative', 'confirmed', 'deleted')
- `age`: Frames since last update
- `history`: List of previous positions

---

### Counter (`src/counter.py`)

Line-crossing counter with anti-double-count mechanisms.

#### Class: `LineCounter`

```python
from src.counter import LineCounter

counter = LineCounter(
    line=((x1, y1), (x2, y2)),
    min_track_length=10,
    debounce_frames=30
)
```

**Methods:**

- `update(tracks)` - Update counts based on current tracks
  - Returns: List of new `CountEvent` objects

- `get_events(since=None)` - Get counting events
  - `since`: Optional timestamp to filter events

- `reset()` - Reset all counts and states

**Attributes:**
- `count_in`: Total IN count
- `count_out`: Total OUT count
- `events`: List of all counting events

#### Class: `CountEvent`

**Attributes:**
- `timestamp`: Event timestamp
- `track_id`: Associated track ID
- `direction`: 'IN' or 'OUT'
- `crossing_point`: (x, y) coordinates
- `confidence`: Detection confidence

---

### Pipeline (`src/pipeline.py`)

Complete detection → tracking → counting pipeline.

#### Class: `PeopleCountingPipeline`

```python
from src.pipeline import PeopleCountingPipeline

pipeline = PeopleCountingPipeline(
    detector_model="yolov8n.pt",
    tracker_type="strongsort",
    conf_threshold=0.5
)
```

**Methods:**

- `run(source, counting_line, display=True, save_output=None)`
  - `source`: Video file path, RTSP URL, or webcam index
  - `counting_line`: ((x1, y1), (x2, y2)) coordinates
  - `display`: Show live preview
  - `save_output`: Optional output video path
  - Returns: Dict with counts and statistics

- `set_counting_line(line, min_track_length=10, debounce_frames=30)`
  - Configure counting parameters

**CLI Usage:**

```bash
python src/pipeline.py \
    --source video.mp4 \
    --line 300,400,900,400 \
    --model yolov8n.pt \
    --tracker strongsort \
    --display \
    --output result.mp4
```

---

## Utilities

### Geometry (`src/utils/geometry.py`)

Geometric utility functions.

**Functions:**

- `line_intersection(line1, line2)` - Find intersection point
- `point_side_of_line(point, line)` - Determine which side of line
- `trajectory_crosses_line(trajectory, line)` - Check if path crosses line
- `point_to_line_distance(point, line)` - Calculate distance

---

### Database (`src/utils/database.py`)

SQLite database for event logging.

#### Class: `Database`

```python
from src.utils.database import Database

db = Database(db_path="database/events.db")
```

**Methods:**

- `insert_event(event)` - Log a counting event
- `get_events(start_time, end_time, direction)` - Query events
- `get_count_by_time(interval='hour')` - Get aggregated counts
- `export_csv(output_path)` - Export to CSV

---

### Video Reader (`src/utils/video_reader.py`)

Asynchronous video reader with buffering.

#### Class: `VideoReader`

```python
from src.utils.video_reader import VideoReader

reader = VideoReader(
    source="video.mp4",
    buffer_size=30,
    resize=(640, 480)
)

reader.start()
ret, frame, frame_number = reader.read(timeout=1.0)
reader.stop()
```

---

## Training

### Train YOLOv8

```bash
python train/train_yolov8.py \
    --data train/config.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 16 \
    --device 0
```

### Export Models

```bash
# Export to ONNX
python src/utils/export_onnx.py \
    --model models/best.pt \
    --format onnx \
    --dynamic

# Export to TensorRT
python src/utils/export_onnx.py \
    --model models/best.pt \
    --format tensorrt \
    --half
```

---

## Evaluation

### Counting Accuracy

```bash
python eval/eval_counting.py \
    --dataset datasets/PETS2009 \
    --model yolov8n.pt \
    --output eval/counting_results.json
```

### Tracking Metrics

```bash
python eval/eval_tracking.py \
    --dataset datasets/MOT17 \
    --model yolov8n.pt \
    --tracker strongsort \
    --output eval/tracking_results.json
```

---

## Streamlit Dashboard

### Launch App

```bash
streamlit run app.py
```

### Features

- **Video Upload**: Upload MP4/AVI/MOV files
- **RTSP Streams**: Connect to live camera feeds
- **Line Drawing**: Interactive counting line configuration
- **Live Monitoring**: Real-time counts and visualization
- **Analytics**: Historical charts and statistics
- **Export**: Download events as CSV/JSON

---

## Docker Deployment

### Build Image

```bash
docker-compose build
```

### Run Container

```bash
docker-compose up -d
```

### Access Dashboard

```
http://localhost:8501
```

---

## Configuration

### Detector Settings

- `conf_threshold`: Confidence threshold (0.0-1.0)
- `nms_threshold`: Non-maximum suppression threshold
- `device`: 'cuda' or 'cpu'

### Tracker Settings

- `tracker_type`: 'strongsort' or 'bytetrack'
- `max_age`: Maximum frames to keep lost tracks
- `min_hits`: Minimum detections before confirmation
- `iou_threshold`: IoU threshold for matching

### Counter Settings

- `min_track_length`: Minimum track length to count
- `debounce_frames`: Frames to wait before re-counting
- `direction_threshold`: Pixels to determine direction

---

## Examples

### Basic Usage

```python
from src.pipeline import PeopleCountingPipeline

# Initialize pipeline
pipeline = PeopleCountingPipeline(
    detector_model="yolov8n.pt",
    tracker_type="strongsort"
)

# Define counting line
counting_line = ((300, 400), (900, 400))

# Process video
results = pipeline.run(
    source="video.mp4",
    counting_line=counting_line,
    display=True
)

print(f"IN: {results['count_in']}")
print(f"OUT: {results['count_out']}")
```

### Custom Integration

```python
from src.detector import Detector
from src.tracker import Tracker
from src.counter import LineCounter
import cv2

# Initialize components
detector = Detector("yolov8n.pt")
tracker = Tracker("strongsort")
counter = LineCounter(line=((300, 400), (900, 400)))

# Process video
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect
    boxes, scores, _ = detector.detect(frame)
    
    # Track
    tracks = tracker.update(boxes.tolist(), scores.tolist(), frame)
    
    # Count
    events = counter.update(tracks)
    
    # Handle new events
    for event in events:
        print(f"New {event.direction}: Track {event.track_id}")

cap.release()

print(f"Total IN: {counter.count_in}")
print(f"Total OUT: {counter.count_out}")
```
