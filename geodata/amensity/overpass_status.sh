#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Show status for repo-local Overpass services and run a health check.

Usage:
  geodata/amensity/overpass_status.sh [--no-health]
EOF
}

log() { echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] $*" >&2; }

NO_HEALTH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --no-health) NO_HEALTH=1; shift ;;
    *) log "ERROR: Unknown argument: $1"; usage; exit 2 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi

AMENSITY_DIR="${REPO_ROOT}/geodata/amensity"
OVERPASS_PREFIX="${AMENSITY_DIR}/vendor/overpass/install"
OVERPASS_DIR="${REPO_ROOT}/geodata/overpass"
DB_DIR="${OVERPASS_DIR}/db"
LOGS_DIR="${OVERPASS_DIR}/logs"

DISPATCHER_PIDFILE="${OVERPASS_DIR}/dispatcher.pid"
HTTP_PIDFILE="${OVERPASS_DIR}/http.pid"

DISPATCHER_LOG="${LOGS_DIR}/dispatcher.log"
HTTP_LOG="${LOGS_DIR}/http.log"

IMPORT_MARKER="${DB_DIR}/.import_complete"

is_running_from_pidfile() {
  local pidfile="$1"
  [[ -f "${pidfile}" ]] || return 1
  local pid
  pid="$(cat "${pidfile}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

build_ok=0
if [[ -x "${OVERPASS_PREFIX}/bin/dispatcher" && -x "${OVERPASS_PREFIX}/bin/init_osm3s.sh" && -x "${OVERPASS_PREFIX}/cgi-bin/interpreter" ]]; then
  build_ok=1
fi

log "Overpass build: $([[ ${build_ok} -eq 1 ]] && echo OK || echo MISSING) (${OVERPASS_PREFIX})"
log "DB import: $([[ -f ${IMPORT_MARKER} ]] && echo OK || echo MISSING) (${IMPORT_MARKER})"

if is_running_from_pidfile "${DISPATCHER_PIDFILE}"; then
  log "Dispatcher: RUNNING (pid $(cat "${DISPATCHER_PIDFILE}")) log=${DISPATCHER_LOG}"
else
  log "Dispatcher: STOPPED (pidfile=${DISPATCHER_PIDFILE})"
fi

if is_running_from_pidfile "${HTTP_PIDFILE}"; then
  log "HTTP wrapper: RUNNING (pid $(cat "${HTTP_PIDFILE}")) log=${HTTP_LOG}"
else
  log "HTTP wrapper: STOPPED (pidfile=${HTTP_PIDFILE})"
fi

if [[ "${NO_HEALTH}" -eq 1 ]]; then
  exit 0
fi

if ! is_running_from_pidfile "${DISPATCHER_PIDFILE}"; then
  log "Health check: FAIL (dispatcher not running)"
  exit 1
fi
if ! is_running_from_pidfile "${HTTP_PIDFILE}"; then
  log "Health check: FAIL (http wrapper not running)"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  log "Health check: FAIL (missing python3)"
  exit 1
fi

python3 - <<'PY'
import json
import sys
import urllib.parse
import urllib.request

query = """[out:json][timeout:25];
node(around:300,52.5200,13.4050)["amenity"="cafe"];
out 5;
"""

data = urllib.parse.urlencode({"data": query}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8080/api/interpreter",
    data=data,
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read()
        status = getattr(resp, "status", None)
except Exception as e:
    print(f"[health] request failed: {e}", file=sys.stderr)
    sys.exit(1)

try:
    json.loads(body)
except Exception as e:
    snippet = body[:400]
    print(f"[health] JSON parse failed: {e}", file=sys.stderr)
    print(f"[health] Response snippet (first 400 bytes): {snippet!r}", file=sys.stderr)
    sys.exit(1)

print(f"[health] OK (http_status={status}, bytes={len(body)})")
PY
