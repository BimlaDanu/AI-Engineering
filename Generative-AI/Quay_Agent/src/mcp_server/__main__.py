"""Entry point, so that ``python -m src.mcp_server`` starts the server.

A client is configured with a command and nothing else, and the shortest command
that works is the one people get right. The module exists only to make that
command valid; everything it does is in :mod:`src.mcp_server.server`.
"""

from __future__ import annotations

import sys

from src.mcp_server.server import main

if __name__ == "__main__":
    sys.exit(main())
