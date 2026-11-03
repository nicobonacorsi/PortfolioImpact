#!/usr/bin/env bash
# Start Portfolio Impact automatically when a Codespace browser attaches.
# The server stays available while the private Codespace is running.
set -euo pipefail
cd "$(dirname "$0")/.."

runtime_dir="${TMPDIR:-/tmp}"
pid_file="$runtime_dir/portfolio-impact-v4-${UID}.pid"
log_file="$runtime_dir/portfolio-impact-v4-${UID}.log"

if [ -r "$pid_file" ]; then
  existing_pid=""
  IFS= read -r existing_pid < "$pid_file" || true
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Portfolio Impact is already running (PID $existing_pid)."
    exit 0
  fi
fi

nohup bash scripts/start_v4_codespaces.sh > "$log_file" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$pid_file"

echo "Portfolio Impact is starting automatically on private port 8765."
echo "If startup fails, inspect $log_file."

# Fail the attach hook visibly if the process terminates immediately.
sleep 2
if ! kill -0 "$server_pid" 2>/dev/null; then
  echo "Portfolio Impact exited during startup:" >&2
  tail -n 40 "$log_file" >&2
  exit 1
fi
