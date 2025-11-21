# CCTV People Counting System - Example Placeholder

This directory contains example videos for testing the people counting system.

## Sample Videos

Due to size limitations, sample videos are not included in the repository.

### Download Sample Videos

You can download sample test videos from:
- **PETS2009**: http://www.cvg.reading.ac.uk/PETS2009/
- **MOT17**: https://motchallenge.net/data/MOT17/
- **Custom**: Record your own CCTV footage

### Expected Format

- **Video formats**: MP4, AVI, MOV, MKV
- **Resolution**: Any (will be automatically processed)
- **Frame rate**: Recommended 15-30 FPS
- **Content**: People walking in view of camera

### Sample Video Naming

Place your test videos in this directory with descriptive names:
- `sample1.mp4` - Sparse crowd scene
- `sample2.mp4` - Dense crowd scene  
- `outdoor_scene.mp4` - Outdoor environment
- `indoor_mall.mp4` - Indoor shopping mall

## Quick Test

Once you have a sample video, run:

```bash
# Using Python pipeline
python src/pipeline.py --source examples/sample.mp4 --line 300,400,900,400

# Using Streamlit dashboard
streamlit run app.py
```

## RTSP Stream Testing

For RTSP stream testing, use the format:
```
rtsp://username:password@camera_ip:554/stream_path
```

Example:
```bash
python src/pipeline.py --source "rtsp://admin:password@192.168.1.100:554/stream1"
```

## Creating Your Own Test Video

Tips for creating good test videos:
1. Fixed camera position (no panning)
2. Clear view of walking path
3. Good lighting conditions
4. Minimal occlusions if possible
5. At least 30 seconds duration for testing
