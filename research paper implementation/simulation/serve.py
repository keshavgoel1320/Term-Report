"""Tiny HTTP server to serve the simulation UI.

Usage:
    python serve.py
    
Opens http://localhost:8080 in your default browser.
Serves files from the project root so the simulation can access
results.json, data/products.json, and data/queries.json.
"""

import http.server
import os
import sys
import threading
import webbrowser

PORT = 8080
# Serve from the project root (parent of simulation/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def log_message(self, format, *args):
        # Quieter logging
        if "200" in str(args):
            return
        super().log_message(format, *args)


def main():
    server = http.server.HTTPServer(("", PORT), Handler)
    url = f"http://localhost:{PORT}/simulation/index.html"
    print(f"\n  >> Serving at {url}\n")
    print(f"  Press Ctrl+C to stop.\n")

    # Open browser after a short delay
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
