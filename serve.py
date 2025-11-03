#!/usr/bin/env python
"""
Production server using Waitress (works on all platforms)
"""
import os
import sys

# Import the app
from app import app, load_model

if __name__ == '__main__':
    # Load the model
    if not load_model():
        print("ERROR: Failed to load model. Exiting.")
        sys.exit(1)
    
    # Get port from environment
    port = int(os.environ.get('PORT', 8000))
    
    print(f"=" * 60)
    print(f"Starting Gutter Pricing API on port {port}")
    print(f"=" * 60)
    
    # Try to use waitress (production server)
    try:
        from waitress import serve
        print("Using Waitress production server...")
        serve(app, host='0.0.0.0', port=port, threads=4)
    except ImportError:
        # Fallback to Flask development server
        print("Waitress not found, using Flask development server...")
        app.run(host='0.0.0.0', port=port, debug=False)
