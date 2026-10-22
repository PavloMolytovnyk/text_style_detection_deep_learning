#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
if [[ ! -f .venv/textscope-ready ]]; then
    if [[ "$(uname -s)" != "Darwin" ]]; then
        .venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
    fi
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/textscope-ready
fi
exec .venv/bin/python run.py "$@"
