#!/bin/bash
# Quick demo launcher for CCTV People Counting System

echo "==================================="
echo "CCTV People Counting System - Demo"
echo "==================================="
echo ""

# Check if example video exists
if [ ! -f "examples/sample.mp4" ]; then
    echo "Downloading sample video..."
    # For now, we'll note that user needs to provide a video
    echo "⚠️  Please place a sample video at examples/sample.mp4"
    echo "   Or provide your own video through the Streamlit interface"
    echo ""
fi

# Check if dependencies are installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting Streamlit dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the demo"
echo ""

# Launch Streamlit app
streamlit run app.py --server.port=8501
