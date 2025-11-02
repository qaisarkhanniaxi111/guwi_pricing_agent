#!/bin/bash
# Startup script for deployment

echo "Starting Gutter Pricing API..."

# Install dependencies if not installed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting application..."
# Try gunicorn first, fallback to Flask dev server
if command -v gunicorn &> /dev/null; then
    echo "Using Gunicorn..."
    gunicorn app:app --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120
else
    echo "Gunicorn not found, using Flask development server..."
    python app.py
fi