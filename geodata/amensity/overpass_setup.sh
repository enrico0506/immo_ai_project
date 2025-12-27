#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare repo-local Overpass DB: create folders, fetch/convert an extract, and import it.

Usage:
  geodata/amensity/overpass_setup.sh [--extract-url URL | --extract-file PATH | --interactive]
                                    [--meta=yes|no]
                                    [--force-import]

Defaults:
  - Extract: first entry in ./geodata/amensity/extracts.txt
  - Meta: no
EOF
}

log() { echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

EXTRACT_URL=""
EXTRACT_FILE=""
INTERACTIVE=0
META="no"
FORCE_IMPORT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --extract-url)
      [[ $# -ge 2 ]] || die "--extract-url requires an argument"
      EXTRACT_URL="$2"
      shift 2
      ;;
    --extract-file)
      [[ $# -ge 2 ]] || die "--extract-file requires an argument"
      EXTRACT_FILE="$2"
      shift 2
      ;;
    --interactive)
      INTERACTIVE=1
      shift
      ;;
    --meta=*)
      META="${1#*=}"
      shift
      ;;
    --meta)
      [[ $# -ge 2 ]] || die "--meta requires yes|no (or use --meta=yes|no)"
      META="$2"
      shift 2
      ;;
    --force-import)
      FORCE_IMPORT=1
      shift
      ;;
    *)
      die "Unknown argument: $1 (use --help)"
      ;;
  esac
done

case "${META}" in
  yes|no) ;;
  *) die "--meta must be yes or no" ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi

AMENSITY_DIR="${REPO_ROOT}/geodata/amensity"
EXTRACTS_FILE="${AMENSITY_DIR}/extracts.txt"
OSM_DIR="${REPO_ROOT}/geodata/osm"
OVERPASS_DIR="${REPO_ROOT}/geodata/overpass"
DB_DIR="${OVERPASS_DIR}/db"
DIFFS_DIR="${OVERPASS_DIR}/diffs"
LOGS_DIR="${OVERPASS_DIR}/logs"
DISPATCHER_PIDFILE="${OVERPASS_DIR}/dispatcher.pid"

OVERPASS_PREFIX="${AMENSITY_DIR}/vendor/overpass/install"
INIT_OSM3S="${OVERPASS_PREFIX}/bin/init_osm3s.sh"

mkdir -p "${AMENSITY_DIR}" "${OSM_DIR}" "${DB_DIR}" "${DIFFS_DIR}" "${LOGS_DIR}"

if [[ -f "${DISPATCHER_PIDFILE}" ]]; then
  pid="$(cat "${DISPATCHER_PIDFILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    die "Dispatcher is running (pid ${pid}). Stop it before importing: geodata/amensity/overpass_stop.sh"
  fi
fi

if [[ ! -x "${INIT_OSM3S}" ]]; then
  log "Overpass not built yet; building now..."
  "${AMENSITY_DIR}/overpass_build.sh"
fi
[[ -x "${INIT_OSM3S}" ]] || die "Missing init script after build: ${INIT_OSM3S}"

require_cmd() { command -v "$1" >/dev/null 2>&1; }

print_install_help_osmium() {
  cat >&2 <<'EOF'
Need osmium-tool to convert .pbf -> .osm.bz2.

Debian/Ubuntu:
  sudo apt-get update
  sudo apt-get install -y osmium-tool bzip2

Fedora:
  sudo dnf install -y osmium-tool bzip2

Arch:
  sudo pacman -S --needed osmium-tool bzip2
EOF
}

download() {
  local url="$1"
  local dest="$2"

  if [[ -s "${dest}" ]]; then
    log "Extract already present: ${dest}"
    return 0
  fi

  local tmp="${dest}.partial"
  rm -f -- "${tmp}"

  if require_cmd wget; then
    log "Downloading (wget): ${url}"
    if [[ -t 2 ]]; then
      wget --progress=bar:force:noscroll -O "${tmp}" "${url}" || die "Download failed: ${url}"
    else
      wget -q -O "${tmp}" "${url}" || die "Download failed: ${url}"
    fi
  elif require_cmd curl; then
    log "Downloading (curl): ${url}"
    if [[ -t 2 ]]; then
      curl -fSL -# -o "${tmp}" "${url}" || die "Download failed: ${url}"
    else
      curl -fsSL -o "${tmp}" "${url}" || die "Download failed: ${url}"
    fi
  else
    die "Need wget or curl to download extracts"
  fi

  mv -f -- "${tmp}" "${dest}"
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

parse_extracts() {
  [[ -f "${EXTRACTS_FILE}" ]] || die "Missing extracts list: ${EXTRACTS_FILE}"
  mapfile -t lines < <(grep -vE '^[[:space:]]*(#|$)' "${EXTRACTS_FILE}" || true)
  if (( ${#lines[@]} == 0 )); then
    die "No extracts found in ${EXTRACTS_FILE}"
  fi

  labels=()
  urls=()
  for line in "${lines[@]}"; do
    line="$(trim "${line}")"
    local label url
    if [[ "${line}" == *$'\t'* ]]; then
      label="${line%%$'\t'*}"
      url="${line#*$'\t'}"
    elif [[ "${line}" == *"|"* ]]; then
      label="${line%%|*}"
      url="${line#*|}"
    else
      label="${line%% *}"
      url="${line#* }"
    fi
    label="$(trim "${label}")"
    url="$(trim "${url}")"
    [[ -n "${label}" && -n "${url}" ]] || continue
    labels+=("${label}")
    urls+=("${url}")
  done
  if (( ${#labels[@]} == 0 )); then
    die "No valid extracts found in ${EXTRACTS_FILE}"
  fi
}

select_extract_url() {
  if [[ -n "${EXTRACT_URL}" && -n "${EXTRACT_FILE}" ]]; then
    die "Use only one of --extract-url or --extract-file"
  fi

  if [[ -n "${EXTRACT_URL}" ]]; then
    printf '%s' "${EXTRACT_URL}"
    return 0
  fi

  if [[ -n "${EXTRACT_FILE}" ]]; then
    printf '%s' ""
    return 0
  fi

  parse_extracts
  if [[ "${INTERACTIVE}" -eq 1 ]]; then
    log "Available extracts from ${EXTRACTS_FILE}:"
    for i in "${!labels[@]}"; do
      printf '%2d) %s\n    %s\n' "$((i+1))" "${labels[$i]}" "${urls[$i]}" >&2
    done
    local choice
    read -r -p "Select extract [1-${#labels[@]}] (default 1): " choice
    choice="${choice:-1}"
    [[ "${choice}" =~ ^[0-9]+$ ]] || die "Invalid choice: ${choice}"
    (( choice >= 1 && choice <= ${#labels[@]} )) || die "Choice out of range: ${choice}"
    printf '%s' "${urls[$((choice-1))]}"
    return 0
  fi

  printf '%s' "${urls[0]}"
}

EXTRACT_SRC_TYPE=""
EXTRACT_INPUT=""
EXTRACT_LABEL=""

if [[ -n "${EXTRACT_FILE}" ]]; then
  EXTRACT_SRC_TYPE="file"
  [[ -f "${EXTRACT_FILE}" ]] || die "Extract file not found: ${EXTRACT_FILE}"
  src="$(cd -- "$(dirname -- "${EXTRACT_FILE}")" && pwd)/$(basename -- "${EXTRACT_FILE}")"
  base="$(basename -- "${src}")"
  if [[ "${base}" == *.pbf ]]; then
    require_cmd osmium || { print_install_help_osmium; exit 1; }
    require_cmd bzip2 || { print_install_help_osmium; exit 1; }
  fi
  dest="${OSM_DIR}/${base}"
  if [[ "${src}" != "${dest}" ]]; then
    if [[ -s "${dest}" ]]; then
      log "Using existing copy in repo: ${dest}"
    else
      log "Copying extract into repo: ${dest}"
      cp -f -- "${src}" "${dest}"
    fi
  fi
  EXTRACT_INPUT="${dest}"
  EXTRACT_LABEL="${base}"
else
  EXTRACT_SRC_TYPE="url"
  url="$(select_extract_url)"
  [[ -n "${url}" ]] || die "Could not determine extract URL"
  base="$(basename -- "${url}")"
  if [[ "${base}" == *.pbf ]]; then
    require_cmd osmium || { print_install_help_osmium; exit 1; }
    require_cmd bzip2 || { print_install_help_osmium; exit 1; }
  fi
  dest="${OSM_DIR}/${base}"
  download "${url}" "${dest}"
  EXTRACT_INPUT="${dest}"
  EXTRACT_LABEL="${base}"
fi

EXTRACT_OSM_BZ2="${EXTRACT_INPUT}"
if [[ "${EXTRACT_INPUT}" == *.pbf ]]; then
  require_cmd osmium || { print_install_help_osmium; exit 1; }
  require_cmd bzip2 || { print_install_help_osmium; exit 1; }
  out="${OSM_DIR}/$(basename -- "${EXTRACT_INPUT%.pbf}").osm.bz2"
  if [[ -s "${out}" ]]; then
    log "Converted extract already present: ${out}"
  else
    log "Converting .pbf -> .osm.bz2 (this can take a while): ${out}"
    tmp="${out}.partial"
    rm -f -- "${tmp}"
    if require_cmd pv; then
      if ! osmium cat "${EXTRACT_INPUT}" -f osm | pv -ptebar | bzip2 -c > "${tmp}"; then
        rm -f -- "${tmp}" || true
        die "Conversion failed"
      fi
    else
      log "Tip: install 'pv' for a progress bar during conversion."
      if ! osmium cat "${EXTRACT_INPUT}" -f osm | bzip2 -c > "${tmp}"; then
        rm -f -- "${tmp}" || true
        die "Conversion failed"
      fi
    fi
    mv -f -- "${tmp}" "${out}"
  fi
  EXTRACT_OSM_BZ2="${out}"
  EXTRACT_LABEL="$(basename -- "${out}")"
fi

if [[ "${EXTRACT_OSM_BZ2}" != *.osm.bz2 ]]; then
  die "Unsupported extract format: ${EXTRACT_INPUT}\nProvide a .osm.bz2 (preferred) or a .pbf (requires osmium-tool)."
fi

require_cmd bunzip2 || die "Missing bunzip2 (from bzip2). Install bzip2 and retry."

IMPORT_MARKER="${DB_DIR}/.import_complete"
if [[ -f "${IMPORT_MARKER}" && "${FORCE_IMPORT}" -eq 0 ]]; then
  prev_extract="$(grep -E '^extract=' "${IMPORT_MARKER}" | head -n1 | cut -d= -f2- || true)"
  prev_meta="$(grep -E '^meta=' "${IMPORT_MARKER}" | head -n1 | cut -d= -f2- || true)"
  if [[ "${prev_extract}" == "$(basename -- "${EXTRACT_LABEL}")" && "${prev_meta}" == "${META}" ]]; then
    log "Import already complete (marker present): ${IMPORT_MARKER}"
    log "extract=${prev_extract} meta=${prev_meta}"
    exit 0
  fi
  die "DB already imported with different settings (marker: ${IMPORT_MARKER}). Use --force-import to reimport."
fi

if [[ "${META}" == "yes" ]]; then
  log "WARNING: --meta=yes can fail if the extract lacks metadata; default is --meta=no."
fi

if [[ "${FORCE_IMPORT}" -eq 1 ]]; then
  log "--force-import set: clearing DB dir: ${DB_DIR}"
  find "${DB_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
fi

IMPORT_LOG="${LOGS_DIR}/import.log"
META_FLAG=()
if [[ "${META}" == "yes" ]]; then
  META_FLAG=(--meta)
fi

log "Importing into Overpass DB: ${DB_DIR}"
log "Import log: ${IMPORT_LOG}"

UPDATE_DB="${OVERPASS_PREFIX}/bin/update_database"
if [[ -x "${UPDATE_DB}" ]]; then
  if ! (
    set -euo pipefail
    if require_cmd pv; then
      bunzip2 < "${EXTRACT_OSM_BZ2}" | pv -ptebar | "${UPDATE_DB}" --db-dir="${DB_DIR}" "${META_FLAG[@]}"
    else
      bunzip2 < "${EXTRACT_OSM_BZ2}" | "${UPDATE_DB}" --db-dir="${DB_DIR}" "${META_FLAG[@]}"
    fi
  ) >"${IMPORT_LOG}" 2>&1; then
    log "Import failed. Last 200 lines of ${IMPORT_LOG}:"
    tail -n 200 "${IMPORT_LOG}" >&2 || true
    exit 1
  fi
else
  if ! (
    set -euo pipefail
    "${INIT_OSM3S}" "${EXTRACT_OSM_BZ2}" "${DB_DIR}" "${OVERPASS_PREFIX}" "${META_FLAG[@]}"
  ) >"${IMPORT_LOG}" 2>&1; then
    log "Import failed. Last 200 lines of ${IMPORT_LOG}:"
    tail -n 200 "${IMPORT_LOG}" >&2 || true
    exit 1
  fi
fi

if [[ ! -f "${DB_DIR}/osm_base_version" ]]; then
  log "Import failed. Last 200 lines of ${IMPORT_LOG}:"
  tail -n 200 "${IMPORT_LOG}" >&2 || true
  die "Import did not create expected DB files (missing ${DB_DIR}/osm_base_version)"
fi

cat > "${IMPORT_MARKER}" <<EOF
extract=$(basename -- "${EXTRACT_LABEL}")
source_type=${EXTRACT_SRC_TYPE}
source_path=${EXTRACT_INPUT}
meta=${META}
imported_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
EOF

log "Import OK. Marker written: ${IMPORT_MARKER}"
