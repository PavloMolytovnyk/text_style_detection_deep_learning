"""Start the UI, API, or both, on Windows/Linux/macOS without Docker."""

import argparse
import os
import subprocess
import sys

from textstyle.settings import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ui", "api", "both"], default="ui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--api-port", type=int, default=8000)
    args = parser.parse_args()
    children = []
    try:
        if args.mode in ["api", "both"]:
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "app.main:app",
                        "--host",
                        args.host,
                        "--port",
                        str(args.api_port),
                    ],
                    cwd=ROOT,
                )
            )
        if args.mode in ["ui", "both"]:
            env = os.environ.copy()
            if args.mode == "both":
                env["API_URL"] = f"http://127.0.0.1:{args.api_port}"
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        "streamlit_app.py",
                        "--server.address",
                        args.host,
                        "--server.port",
                        str(args.ui_port),
                    ],
                    cwd=ROOT,
                    env=env,
                )
            )
        while children:
            try:
                code = children[-1].wait(timeout=1)
                if code:
                    raise SystemExit(code)
                break
            except subprocess.TimeoutExpired:
                if any(child.poll() is not None for child in children):
                    raise SystemExit("A service stopped. Check the console output above.") from None
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    main()
