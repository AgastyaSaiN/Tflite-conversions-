#!/bin/bash
# Sourced by RUN.sh / CHECK_SETUP.sh: if the current Python lacks the packages, switch to the conda environment "basept" created by SETUP_LINUX.sh.
if ! python -c "import torch, tensorflow" >/dev/null 2>&1; then
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/miniconda3" "/opt/conda" "/opt/miniconda3" "$(dirname "$(dirname "$(command -v conda 2>/dev/null)")" 2>/dev/null)"; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "$base/etc/profile.d/conda.sh"
      conda activate basept 2>/dev/null && { echo "(using conda environment basept)"; break; }
    fi
  done
fi
