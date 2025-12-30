#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Stop repo-local Overpass dispatcher + local HTTP wrapper.

Usage:
  geodata/amensity/overpass_stop.sh
EOF
}

log() { echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] $*" >&2; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi

OVERPASS_DIR="${REPO_ROOT}/geodata/overpass"
DISPATCHER_PIDFILE="${OVERPASS_DIR}/dispatcher.pid"
HTTP_PIDFILE="${OVERPASS_DIR}/http.pid"
SOCKET_DIR="${OVERPASS_DIR}/socket"
DB_DIR="${OVERPASS_DIR}/db"
LOGS_DIR="${OVERPASS_DIR}/logs"

stop_pidfile() {
  local name="$1"
  local pidfile="$2"

  if [[ ! -f "${pidfile}" ]]; then
    log "${name}: not running (no pidfile)"
    return 0
  fi

  local pid
  pid="$(cat "${pidfile}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    log "${name}: removing empty pidfile: ${pidfile}"
    rm -f -- "${pidfile}"
    return 0
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    log "${name}: stale pidfile (pid ${pid}); removing: ${pidfile}"
    rm -f -- "${pidfile}"
    return 0
  fi

  log "${name}: stopping (pid ${pid})"
  kill -TERM "${pid}" 2>/dev/null || true

  local i
  for i in {1..50}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f -- "${pidfile}"
      log "${name}: stopped"
      return 0
    fi
    sleep 0.2
  done

  log "${name}: did not stop in time; sending SIGKILL"
  kill -KILL "${pid}" 2>/dev/null || true
  rm -f -- "${pidfile}"
  return 0
}

stop_pidfile "HTTP wrapper" "${HTTP_PIDFILE}"
stop_pidfile "Dispatcher" "${DISPATCHER_PIDFILE}"

kill_stray() {
  local name="$1"
  shift
  local pattern="$1"
  shift
  if ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids="$(pgrep -f "${pattern}" || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  log "${name}: found stray processes (pgrep -f '${pattern}'): ${pids}"
  kill -TERM ${pids} 2>/dev/null || true
  sleep 0.5
  for pid in ${pids}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}

# If pidfiles were stale/missing after a crash, try best-effort cleanup by pattern.
kill_stray "Dispatcher" "dispatcher.*--db-dir=${DB_DIR}"
kill_stray "HTTP wrapper" "overpass_http\\.py"

compute_socket_link() {
  local id=""
  if command -v sha1sum >/dev/null 2>&1; then
    id="$(printf '%s' "${REPO_ROOT}" | sha1sum | awk '{print substr($1,1,10)}')"
  elif command -v shasum >/dev/null 2>&1; then
    id="$(printf '%s' "${REPO_ROOT}" | shasum | awk '{print substr($1,1,10)}')"
  else
    id="$(printf '%s' "${REPO_ROOT}" | cksum | awk '{print $1}')"
  fi
  printf '%s' "/tmp/overpass_socket_${id}"
}

SOCKET_LINK="$(compute_socket_link)"
if [[ -L "${SOCKET_LINK}" ]]; then
  target="$(readlink "${SOCKET_LINK}" || true)"
  if [[ "${target}" == "${SOCKET_DIR}" ]]; then
    rm -f -- "${SOCKET_LINK}"
    log "Removed socket symlink: ${SOCKET_LINK}"
  fi
fi

# Remove stale unix socket files that can block dispatcher start (EADDRINUSE).
if [[ -d "${SOCKET_DIR}" ]]; then
  rm -f -- "${SOCKET_DIR}"/osm3s_* 2>/dev/null || true
fi
