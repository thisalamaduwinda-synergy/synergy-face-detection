# Employee Face Recognition System

Real-time employee identification from CCTV streams using deep learning face embeddings, FAISS similarity search, and a FastAPI backend.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CCTV / IP Camera Fleet                          │
│          RTSP Stream 1          RTSP Stream 2          ...          │
└───────────────┬─────────────────────┬───────────────────────────────┘
                │                     │
       ┌────────▼─────────────────────▼────────┐
       │         MultiCameraManager             │
       │   CameraStream threads (buffered)      │
       └────────────────────┬──────────────────┘
                            │ Frame objects
       ┌────────────────────▼──────────────────┐
       │          RecognitionPipeline           │
       │    ThreadPoolExecutor (per camera)     │
       │                                        │
       │  ┌──────────────┐  ┌───────────────┐  │
       │  │ FaceDetector │  │  FaceTracker  │  │
       │  │ (InsightFace │  │  (debounce    │  │
       │  │  RetinaFace) │  │   duplicates) │  │
       │  └──────┬───────┘  └───────────────┘  │
       │         │ DetectedFace + embedding      │
       │  ┌──────▼───────┐                      │
       │  │FaceRecognizer│                      │
       │  │  FAISS index │                      │
       │  │ (cosine sim) │                      │
       │  └──────┬───────┘                      │
       └─────────┼────────────────────────────── ┘
                 │ RecognitionResult
       ┌─────────▼──────────────────────────────┐
       │          LoggingService                 │
       │  Async queue → SQLite/PostgreSQL log    │
       │  Frame save  │  Webhook alert           │
       └─────────┬────────────────────────────── ┘
                 │
       ┌─────────▼──────────────────────────────┐
       │         FastAPI Server (port 8000)      │
       │  REST API  │  MJPEG stream  │ WebSocket │
       │  /api/*    │  /video/{id}   │  /ws/*    │
       └─────────┬────────────────────────────── ┘
                 │ HTTP / WebSocket
       ┌─────────▼──────────────────────────────┐
       │         Web Dashboard (port 8000/)      │
       │  Live camera grid  │  Detection feed    │
       │  Stat cards        │  Employee register │
       └────────────────────────────────────────┘
```

### Folder structure

```
Face Detection/
├── config/
│   └── config.yaml              ← system configuration
├── modules/
│   ├── __init__.py
│   ├── camera_stream.py         ← RTSP / IP camera manager
│   ├── face_detector.py         ← InsightFace detection + embedding
│   ├── face_embedding.py        ← standalone embedding utilities
│   ├── face_recognizer.py       ← FAISS recognition + face tracker
│   ├── employee_database.py     ← SQLAlchemy ORM (SQLite / PostgreSQL)
│   ├── logging_service.py       ← async event logger + webhook alerts
│   └── api_server.py            ← FastAPI application factory
├── scripts/
│   ├── register_employee.py     ← register new employees (CLI / CSV)
│   ├── create_face_dataset.py   ← build aligned face image dataset
│   ├── generate_embeddings.py   ← batch-generate & store embeddings
│   └── tune_threshold.py        ← find optimal similarity threshold
├── dashboard/
│   ├── templates/index.html     ← single-page dashboard
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── data/
│   ├── employees/               ← employee photos
│   └── faces/                   ← aligned 112×112 face crops
├── logs/                        ← log files + optional frame captures
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   └── test_recognition.py
├── main.py                      ← entry point
├── requirements.txt
└── .env.example
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or 3.11 |
| CUDA (optional) | 11.8+ for GPU acceleration |
| Docker (optional) | 24+ |

---

## Installation

### 1 – Clone / copy the project

```bash
cd "Face Detection"
```

### 2 – Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3 – Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU acceleration (optional):** Replace `onnxruntime` with
> `onnxruntime-gpu` and `faiss-cpu` with `faiss-gpu` in
> `requirements.txt`, then set `use_gpu: true` in `config.yaml`.

### 4 – Copy the environment file

```bash
cp .env.example .env
# Edit .env with your actual values
```

---

## Quick Start (webcam test)

```bash
# Register yourself with the local webcam
python scripts/register_employee.py \
    --id EMP001 --name "Your Name" --dept "Engineering" \
    --capture

# Start the system using your webcam (camera index 0)
python main.py --camera-source 0
```

Open `http://localhost:8000` in a browser to see the live dashboard.

---

## Connecting RTSP CCTV Cameras

### 1 – Find your camera's RTSP URL

Most IP cameras follow one of these patterns:

| Vendor | URL pattern |
|--------|-------------|
| Hikvision | `rtsp://user:pass@<IP>:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@<IP>:554/cam/realmonitor?channel=1&subtype=0` |
| Axis | `rtsp://user:pass@<IP>/axis-media/media.amp` |
| Generic ONVIF | `rtsp://user:pass@<IP>:554/stream1` |

### 2 – Test the stream with VLC or ffplay

```bash
ffplay "rtsp://admin:password@192.168.1.100:554/stream1"
```

### 3 – Add cameras to `config/config.yaml`

```yaml
cameras:
  - id: "cam_entrance"
    name: "Main Entrance"
    source: "rtsp://admin:P@ssw0rd@192.168.1.100:554/stream1"
    fps: 25
  - id: "cam_lobby"
    name: "Lobby"
    source: "rtsp://admin:P@ssw0rd@192.168.1.101:554/stream1"
    fps: 25
```

### Network requirements

- The machine running this software must have network access to the cameras.
- If cameras use RTSP over UDP and experience packet loss, the system
  automatically requests TCP transport via `CAP_PROP_OPEN_TIMEOUT_MSEC`.

---

## Registering Employees

### Option A – Single employee with 1–5 photo files

```bash
# Repeat --photo up to 5 times
python scripts/register_employee.py \
    --id EMP042 \
    --name "Jane Doe" \
    --dept "Finance" \
  --photo data/employees/EMP042_front.jpg \
  --photo data/employees/EMP042_left.jpg \
  --photo data/employees/EMP042_right.jpg
```

### Option B – Webcam capture

```bash
python scripts/register_employee.py \
    --id EMP042 --name "Jane Doe" --dept "Finance" \
    --capture
```

Press **SPACE** to take the photo, **Q** to quit.

### Option C – Batch CSV import

```bash
# CSV format: employee_id,name,department,photo_path
python scripts/register_employee.py --csv employees.csv --has-header
```

### Option D – Via the web dashboard

Click the **＋** button in the bottom-right of the dashboard, fill in
the form, and upload **1 to 5** face photos.

### Option E – Via the REST API

```bash
curl -X POST http://localhost:8000/api/employees \
  -F "employee_id=EMP050" \
  -F "name=John Smith" \
  -F "department=IT" \
  -F "photos=@/path/to/john_front.jpg" \
  -F "photos=@/path/to/john_left.jpg" \
  -F "photos=@/path/to/john_right.jpg"
```

Registration requires at least **1** photo and accepts at most **5** photos.
When multiple photos are provided, their embeddings are averaged to improve
recognition stability.

---

## Building the Face Dataset (optional but recommended)

For higher recognition accuracy, provide multiple photos per employee
and let the system average their embeddings.

```
data/raw_photos/
  EMP001/
    front.jpg
    slight_left.jpg
    slight_right.jpg
  EMP002/
    ...
```

```bash
# Step 1: Detect faces, align, and save 112×112 crops
python scripts/create_face_dataset.py \
    --source data/raw_photos \
    --output data/faces \
    --augment          # optional: flip / brightness variants

# Step 2: Generate and store averaged embeddings
python scripts/generate_embeddings.py \
    --faces-dir data/faces \
    --average
```

---

## Starting the System

```bash
# Production – headless (server / Docker)
python main.py --no-display

# Development – with local OpenCV display
python main.py

# Custom config
python main.py --config /path/to/custom.yaml
```

---

## API Reference

### Employees

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/employees` | Register employee (multipart/form-data) |
| `GET` | `/api/employees` | List all active employees |
| `GET` | `/api/employees/{id}` | Get one employee |
| `DELETE` | `/api/employees/{id}` | Deactivate employee |

### Logs & stats

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/logs?limit=100` | Recent detection logs |
| `GET` | `/api/stats` | System stats (DB + cameras + recognizer) |
| `GET` | `/api/cameras` | Camera status & FPS |

### Streaming

| Protocol | Endpoint | Description |
|----------|----------|-------------|
| HTTP MJPEG | `/video/{camera_id}` | Live camera feed |
| WebSocket | `/ws/events` | Real-time detection events (JSON) |
| WebSocket | `/ws/stream/{camera_id}` | Annotated JPEG frames |

### Interactive docs

Visit `http://localhost:8000/docs` for the Swagger UI.

---

## Tuning the Similarity Threshold

A bad threshold causes false acceptances (too low) or false rejections
(too high). Run the tuner on a labelled test set:

```bash
# Prepare test images in data/test_faces/<employee_id>/*.jpg
python scripts/tune_threshold.py --test-dir data/test_faces

# Output example:
# Threshold   Accuracy  Precision    Recall      F1     FAR     FRR
# ──────────────────────────────────────────────────────────────────
#      0.35     0.9120     0.8800    0.9600  0.9183  0.1200  0.0400
#      0.40     0.9280     0.9100    0.9400  0.9247  0.0900  0.0600
#      0.45     0.9440     0.9350    0.9400  0.9375  0.0650  0.0600  ← best F1
#      0.50     0.9360     0.9600    0.9000  0.9290  0.0400  0.1000
```

Update `recognition.threshold` in `config.yaml` accordingly.

---

## Docker Deployment

```bash
cd docker

# Build and start all services
docker compose up --build -d

# Follow logs
docker compose logs -f face-recognition

# Stop
docker compose down
```

The `data/` and `logs/` directories are mounted as volumes so your
database and captured frames persist across container restarts.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Embedding serialisation round-trip
- Cosine similarity correctness
- FAISS index build / search / add / remove
- Face tracker cooldown logic
- Database CRUD and detection logging

---

## Performance Notes

| Configuration | Expected throughput |
|---------------|---------------------|
| CPU only, 1 camera, buffalo_s | ~8–12 FPS |
| CPU only, 1 camera, buffalo_l | ~5–8 FPS |
| GPU (RTX 3060), 1 camera, buffalo_l | ~25–30 FPS |
| GPU (RTX 3060), 4 cameras, buffalo_l | ~20–25 FPS per camera |

**Tuning tips:**
- Increase `frame_skip` in `config.yaml` to reduce CPU load
  (2 = process every other frame, gives 2× headroom).
- Use `buffalo_s` model for faster but slightly less accurate detection.
- Set `use_gpu: true` and install `onnxruntime-gpu` + `faiss-gpu` for
  significant speedups.
- FAISS is O(n) with IndexFlatIP but n ≤ 1000 employees is trivially
  fast (< 1 ms per query). No approximation index needed.

---

## Environment Variables

All values in `.env` override `config.yaml` settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_TYPE` | `sqlite` | `sqlite` or `postgresql` |
| `SQLITE_PATH` | `data/employees.db` | SQLite file path |
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PASSWORD` | — | PostgreSQL password |
| `REDIS_ENABLED` | `false` | Enable Redis caching |
| `USE_GPU` | `false` | Enable CUDA acceleration |
| `RECOGNITION_THRESHOLD` | `0.45` | Cosine similarity threshold |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ALERT_WEBHOOK_URL` | — | Webhook URL for unknown-person alerts |

---

## Security Considerations

- All file uploads are validated (type + decodability) before processing.
- Employee IDs are sanitised to prevent path traversal attacks.
- Database queries use SQLAlchemy's parameterised ORM (no raw SQL
  string interpolation).
- The API does not expose raw face embeddings through any endpoint.
- For production deployments, add HTTPS (TLS termination via nginx or
  a reverse proxy) and API key / JWT authentication.

---

## License

MIT – see LICENSE file.
