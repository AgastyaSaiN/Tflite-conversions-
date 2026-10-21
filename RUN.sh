#!/bin/bash
# Starts the conversion of every .pt file in put_pt_files_here (Linux).   Usage:  ./RUN.sh            or   ./RUN.sh --quick
cd "$(dirname "$0")" || exit 1
source ./activate_env.sh
PY=python
command -v python >/dev/null 2>&1 || PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python was not found. Run ./SETUP_LINUX.sh once first (see README.md, part 'Linux')."
  exit 1
fi
"$PY" run.py "$@"
