#!/usr/bin/env python3
"""
Simple HTTP server to serve the research content locally
This allows the interactive viewer to load markdown files properly
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path

# Set the directory to serve (the resume-2025-novabright directory)
SERVE_DIR = Path(__file__).parent
PORT = 8080

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    """Start the local HTTP server"""
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"🚀 Starting local server...")
            print(f"📁 Serving directory: {SERVE_DIR}")
            print(f"🌐 Server running at: http://localhost:{PORT}")
            print(f"🔗 Interactive viewer: http://localhost:{PORT}/Emergent_Consciousness/html_only_verisons_current/emergent-consciousness-visualization-enhanced.html")
            print(f"\n💡 Press Ctrl+C to stop the server")
            # Automatically open the browser
            viewer_url = f"http://localhost:{PORT}/Emergent_Consciousness/html_only_verisons_current/emergent-consciousness-visualization-enhanced.html"
            print(f"\n🌍 Opening browser to: {viewer_url}")
            webbrowser.open(viewer_url)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use. Try a different port or stop the existing server.")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    start_server()
# End of Selection
