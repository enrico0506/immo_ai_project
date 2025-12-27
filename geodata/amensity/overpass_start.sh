#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start repo-local Overpass dispatcher + local HTTP wrapper.

Usage:
  geodata/amensity/overpass_start.sh
EOF
}

log() { echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi

AMENSITY_DIR="${REPO_ROOT}/geodata/amensity"
OVERPASS_DIR="${REPO_ROOT}/geodata/overpass"
DB_DIR="${OVERPASS_DIR}/db"
LOGS_DIR="${OVERPASS_DIR}/logs"
SOCKET_DIR="${OVERPASS_DIR}/socket"

OVERPASS_PREFIX="${AMENSITY_DIR}/vendor/overpass/install"
DISPATCHER_BIN="${OVERPASS_PREFIX}/bin/dispatcher"
INTERPRETER_BIN="${OVERPASS_PREFIX}/cgi-bin/interpreter"

IMPORT_MARKER="${DB_DIR}/.import_complete"

DISPATCHER_PIDFILE="${OVERPASS_DIR}/dispatcher.pid"
HTTP_PIDFILE="${OVERPASS_DIR}/http.pid"

DISPATCHER_LOG="${LOGS_DIR}/dispatcher.log"
HTTP_LOG="${LOGS_DIR}/http.log"

mkdir -p "${DB_DIR}" "${LOGS_DIR}"
mkdir -p "${SOCKET_DIR}"

cd "${REPO_ROOT}"

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
  if [[ "${target}" != "${SOCKET_DIR}" ]]; then
    log "Updating socket symlink: ${SOCKET_LINK} -> ${SOCKET_DIR}"
    ln -sfn "${SOCKET_DIR}" "${SOCKET_LINK}"
  fi
elif [[ -e "${SOCKET_LINK}" ]]; then
  die "Socket path exists and is not a symlink: ${SOCKET_LINK}"
else
  log "Creating socket symlink: ${SOCKET_LINK} -> ${SOCKET_DIR}"
  ln -s "${SOCKET_DIR}" "${SOCKET_LINK}"
fi

export OVERPASS_SOCKET_DIR="${SOCKET_LINK}/"

if [[ ! -x "${DISPATCHER_BIN}" || ! -x "${INTERPRETER_BIN}" ]]; then
  log "Overpass not built yet; building now..."
  "${AMENSITY_DIR}/overpass_build.sh"
fi
[[ -x "${DISPATCHER_BIN}" ]] || die "Missing dispatcher: ${DISPATCHER_BIN}"
[[ -x "${INTERPRETER_BIN}" ]] || die "Missing interpreter: ${INTERPRETER_BIN}"

[[ -f "${IMPORT_MARKER}" ]] || die "No imported DB found (missing ${IMPORT_MARKER}). Run: geodata/amensity/overpass_setup.sh"
meta="$(grep -E '^meta=' "${IMPORT_MARKER}" | head -n1 | cut -d= -f2- || true)"
meta="${meta:-no}"
case "${meta}" in yes|no) ;; *) meta="no" ;; esac

META_FLAG=()
if [[ "${meta}" == "yes" ]]; then
  META_FLAG=(--meta)
fi

is_running_from_pidfile() {
  local pidfile="$1"
  [[ -f "${pidfile}" ]] || return 1
  local pid
  pid="$(cat "${pidfile}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

start_bg() {
  local pidfile="$1"
  local logfile="$2"
  shift 2
  rm -f -- "${pidfile}"
  : >> "${logfile}"
  if command -v nohup >/dev/null 2>&1; then
    nohup "$@" >>"${logfile}" 2>&1 < /dev/null &
  else
    "$@" >>"${logfile}" 2>&1 < /dev/null &
  fi
  echo $! > "${pidfile}"
}

if is_running_from_pidfile "${DISPATCHER_PIDFILE}"; then
  log "Dispatcher already running (pid $(cat "${DISPATCHER_PIDFILE}"))"
else
  if [[ -f "${DISPATCHER_PIDFILE}" ]]; then
    log "Removing stale pidfile: ${DISPATCHER_PIDFILE}"
    rm -f -- "${DISPATCHER_PIDFILE}"
  fi
  log "Starting dispatcher..."
  start_bg "${DISPATCHER_PIDFILE}" "${DISPATCHER_LOG}" \
    "${DISPATCHER_BIN}" --osm-base --db-dir="${DB_DIR}" --socket-dir="${SOCKET_LINK}" "${META_FLAG[@]}"
  sleep 0.5
  if ! is_running_from_pidfile "${DISPATCHER_PIDFILE}"; then
    log "Dispatcher failed to start. Last 200 lines of ${DISPATCHER_LOG}:"
    tail -n 200 "${DISPATCHER_LOG}" >&2 || true
    exit 1
  fi
  log "Dispatcher started (pid $(cat "${DISPATCHER_PIDFILE}"))"
fi

if ! command -v python3 >/dev/null 2>&1; then
  die "Missing python3. Install python3 and retry."
fi

if is_running_from_pidfile "${HTTP_PIDFILE}"; then
  log "HTTP wrapper already running (pid $(cat "${HTTP_PIDFILE}"))"
else
  if [[ -f "${HTTP_PIDFILE}" ]]; then
    log "Removing stale pidfile: ${HTTP_PIDFILE}"
    rm -f -- "${HTTP_PIDFILE}"
  fi
  log "Starting HTTP wrapper on 127.0.0.1:8080 ..."
  start_bg "${HTTP_PIDFILE}" "${HTTP_LOG}" \
    python3 "${AMENSITY_DIR}/overpass_http.py"
  sleep 0.5
  if ! is_running_from_pidfile "${HTTP_PIDFILE}"; then
    log "HTTP wrapper failed to start. Last 200 lines of ${HTTP_LOG}:"
    tail -n 200 "${HTTP_LOG}" >&2 || true
    exit 1
  fi
  log "HTTP wrapper started (pid $(cat "${HTTP_PIDFILE}"))"
fi

log "Ready:"
log "  POST http://127.0.0.1:8080/api/interpreter"
