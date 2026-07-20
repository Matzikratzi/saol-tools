#!/bin/zsh
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -r requirements.txt
open http://127.0.0.1:8001
exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
