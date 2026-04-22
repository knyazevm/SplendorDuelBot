"""
play_web.py — Launch the web interface for human-vs-AI play.

Usage:
    python scripts/play_web.py
    python scripts/play_web.py --port 8080
    python scripts/play_web.py --host 0.0.0.0 --port 8000
"""
import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="Splendor Duel Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    import uvicorn
    print(f"\n  🎮 Splendor Duel — open http://{args.host}:{args.port}\n")
    uvicorn.run("splendor_duel.web.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
