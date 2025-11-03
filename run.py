#!/usr/bin/env python
"""
Startup wrapper script for deployment
Handles PORT environment variable properly
"""
import os
import sys

# Get the port from environment or use default
port = os.environ.get('PORT', '8000')

# Validate port
try:
    port = int(port)
    if port < 1 or port > 65535:
        port = 8000
except (ValueError, TypeError):
    port = 8000

print(f"Starting server on port {port}...")

# Set PORT back to environment for app.py
os.environ['PORT'] = str(port)

# Now run the app
if __name__ == '__main__':
    # Import and run the Flask app
    from app import app, load_model
    
    if load_model():
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("ERROR: Failed to load model")
        sys.exit(1)
