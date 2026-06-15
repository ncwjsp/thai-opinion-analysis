"""
Entry point — start the FastAPI server.

Usage:
    python run.py

Or directly with uvicorn:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

This module configures application-wide logging before launching the ASGI
server so that start-up messages from the database and model layers are
captured with the standard format defined in ``config.setup_logging``.
"""

import uvicorn

from config import setup_logging

if __name__ == "__main__":
    # Initialise logging first so uvicorn's own startup is logged consistently.
    setup_logging()
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
