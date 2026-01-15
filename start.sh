#!/bin/bash
# Startup script for Render

echo "Starting Gutter Pricing API..."
echo "PORT: $PORT"

# Run gunicorn with environment PORT and preload
exec gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --timeout 120 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    --preload