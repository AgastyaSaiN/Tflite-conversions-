#!/bin/bash
# Checks that this computer has everything needed (Linux).   Usage:  ./CHECK_SETUP.sh
cd "$(dirname "$0")" || exit 1
source ./activate_env.sh
PY=python
command -v python >/dev/null 2>&1 || PY=python3
"$PY" check_setup.py
