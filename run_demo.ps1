# Quick demo launcher for CCTV People Counting System (Windows)

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "CCTV People Counting System - Demo" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Check if example video exists
if (-not (Test-Path "examples\sample.mp4")) {
    Write-Host "⚠️  Sample video not found at examples\sample.mp4" -ForegroundColor Yellow
    Write-Host "   You can provide your own video through the Streamlit interface" -ForegroundColor Yellow
    Write-Host ""
}

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "venv\Scripts\Activate.ps1"

# Check if dependencies are installed
$streamlitInstalled = python -c "import streamlit" 2>$null
if (-not $streamlitInstalled) {
    Write-Host "Installing dependencies..." -ForegroundColor Green
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "Starting Streamlit dashboard..." -ForegroundColor Green
Write-Host "Dashboard will be available at: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the demo" -ForegroundColor Yellow
Write-Host ""

# Launch Streamlit app
streamlit run app.py --server.port=8501
