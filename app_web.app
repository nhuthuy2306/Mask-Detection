"""Compatibility launcher for the laptop webcam Flask application.

Run with: python app_web.app
"""
import os
from app_web import app

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
