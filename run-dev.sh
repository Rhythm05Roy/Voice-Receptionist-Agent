#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
