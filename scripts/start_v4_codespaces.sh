#!/usr/bin/env bash
# Start the approved v4 interface and Python analysis engine on one origin.
set -euo pipefail
cd "$(dirname "$0")/.."

python_cmd=python3
if command -v python3.12 >/dev/null 2>&1; then
  python_cmd=python3.12
fi
if ! "$python_cmd" -c "import sys; assert sys.version_info[:2] == (3, 12)" >/dev/null 2>&1; then
  echo "Python 3.12 is required." >&2
  exit 1
fi
if [ -x ".venv/bin/python" ] && ! .venv/bin/python -c "import sys; assert sys.version_info[:2] == (3, 12)" >/dev/null 2>&1; then
  echo "The existing .venv was created with a different Python version." >&2
  exit 1
fi
if [ ! -x ".venv/bin/python" ]; then
  "$python_cmd" -m venv .venv
fi

# Some Codespaces Python images omit both pip and ensurepip. Bootstrap pip
# into this virtual environment only; never modify the system interpreter.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  if ! .venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
      echo "pip and ensurepip are missing, and curl is required to bootstrap pip." >&2
      exit 1
    fi
    bootstrap_script="$(mktemp "${TMPDIR:-/tmp}/castle-get-pip.XXXXXXXX")"
    trap 'rm -f "$bootstrap_script"' EXIT
    curl --fail --silent --show-error --location \
      "https://bootstrap.pypa.io/get-pip.py" --output "$bootstrap_script"
    .venv/bin/python "$bootstrap_script"
    rm -f "$bootstrap_script"
    trap - EXIT
  fi
fi

.venv/bin/python -m pip install --disable-pip-version-check -r requirements-lock.txt

# Called once during Codespaces creation before the editor is attached.
if [ "${1:-}" = "--setup-only" ]; then
  echo "Portfolio Impact dependencies are ready."
  exit 0
fi

exec .venv/bin/python -m castle_risk.codespaces_server "$@"
