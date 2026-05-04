"""Saamid real-time backend.

`server.app` exposes a FastAPI app that serves the dashboard and pushes live
detection events over a WebSocket. `server.pipeline` runs the audio -> model
-> triangulation loop in a background thread.
"""
