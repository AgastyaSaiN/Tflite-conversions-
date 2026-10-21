#!/bin/bash
# ONE-TIME setup on the Linux computer. Needs NO internet: everything it installs is in the folder linux_offline_packages.
# It (1) installs Miniconda (Python 3.11) if conda is not there yet, (2) creates the conda environment "basept",
# (3) installs all packages into it from the local wheelhouse, (4) runs the setup check.
# Usage:   bash SETUP_LINUX.sh
set -e
cd "$(dirname "$0")"
PKG=linux_offline_packages
say() { echo; echo "== $*"; }
fail() { echo; echo "ERROR: $*"; exit 1; }

say "1/5 checking the computer"
[ "$(uname -m)" = "x86_64" ] || fail "These packages are for Linux on x86_64 (Intel/AMD); this computer is $(uname -m)."
GLIBC=$(ldd --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)
echo "glibc $GLIBC (needs 2.28 or newer)"
awk -v v="$GLIBC" 'BEGIN{exit !(v+0 >= 2.28)}' || fail "glibc $GLIBC is too old for these packages (need 2.28 or newer, e.g. Ubuntu 20.04+, Debian 10+, RHEL/Rocky 8+)."
[ -d "$PKG/wheelhouse" ] || fail "The folder $PKG/wheelhouse is missing. Copy the whole folder 'Base pt conversions and testing' again."

say "2/5 checking that the copied files are complete"
if [ -f "$PKG/SHA256SUMS" ]; then
  (cd "$PKG" && sha256sum --quiet -c SHA256SUMS) || fail "Some package files are damaged or missing (the copy was interrupted?). Copy the folder again."
  echo "all package files are intact"
else
  echo "(no checksum list found, skipping)"
fi

say "3/5 conda"
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /opt/miniconda3; do [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break; done
fi
if ! command -v conda >/dev/null 2>&1; then
  INSTALLER=$(ls "$PKG"/Miniconda3-py311_*-Linux-x86_64.sh 2>/dev/null | head -1)
  [ -n "$INSTALLER" ] || fail "conda is not installed and the Miniconda installer is not in $PKG."
  echo "installing Miniconda into $HOME/miniconda3 (about a minute)"
  bash "$INSTALLER" -b -p "$HOME/miniconda3" >/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda found: $(command -v conda)"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

say "4/5 creating the environment 'basept' (Python 3.11) and installing the packages (several minutes, no internet)"
if conda env list | grep -q '^basept '; then
  echo "environment basept already exists: reusing it (to start from scratch run: conda env remove -n basept)"
else
  conda create -y --offline -n basept python=3.11 >/dev/null 2>&1 || \
    conda create -y --offline -n basept python=3.11 || fail "Could not create the environment from the local conda packages. Install Miniconda from $PKG (the py311 installer) and run this again."
fi
conda activate basept
python -c "import sys; assert sys.version_info[:2]==(3,11), sys.version" || fail "The environment does not have Python 3.11."
grep -v '^--' "$PKG/requirements_linux.lock.txt" > /tmp/basept_requirements.txt
python -m pip install --no-index --find-links "$PKG/wheelhouse" -r /tmp/basept_requirements.txt || fail "Package installation failed. The messages above name the package."

say "5/5 setup check"
python check_setup.py || true
echo
echo "Setup finished. From now on:"
echo "   1. copy the model file(s) into  put_pt_files_here/   (and pictures into put_test_pictures_here/ and put_calibration_pictures_here/)"
echo "   2. run   ./RUN.sh      (you do not need to activate anything by hand)"
