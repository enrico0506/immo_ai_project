#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build Overpass (osm-3s) locally inside this repo.

Usage:
  geodata/amensity/overpass_build.sh [--force-build] [--version X.Y.Z]

Notes:
  - Installs into: ./geodata/amensity/vendor/overpass/install/
  - Writes build log to: ./geodata/overpass/logs/overpass_build.log
EOF
}

log() { echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

FORCE_BUILD=0
OVERPASS_VERSION="${OVERPASS_VERSION:-0.7.62}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --force-build) FORCE_BUILD=1; shift ;;
    --version)
      [[ $# -ge 2 ]] || die "--version requires an argument"
      OVERPASS_VERSION="$2"
      shift 2
      ;;
    *)
      die "Unknown argument: $1 (use --help)"
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi

AMENSITY_DIR="${REPO_ROOT}/geodata/amensity"
VENDOR_DIR="${AMENSITY_DIR}/vendor/overpass"
SRC_BASE="${VENDOR_DIR}/src"
SRC_DIR="${SRC_BASE}/osm-3s_v${OVERPASS_VERSION}"
PREFIX_DIR="${VENDOR_DIR}/install"
TARBALL="${VENDOR_DIR}/osm-3s_v${OVERPASS_VERSION}.tar.gz"
URL="https://dev.overpass-api.de/releases/osm-3s_v${OVERPASS_VERSION}.tar.gz"
BUILD_LOG="${REPO_ROOT}/geodata/overpass/logs/overpass_build.log"

mkdir -p "${AMENSITY_DIR}" "${VENDOR_DIR}" "${SRC_BASE}"
mkdir -p "$(dirname -- "${BUILD_LOG}")"

verify_outputs() {
  [[ -x "${PREFIX_DIR}/bin/dispatcher" ]] || return 1
  [[ -x "${PREFIX_DIR}/bin/init_osm3s.sh" ]] || return 1
  [[ -x "${PREFIX_DIR}/cgi-bin/interpreter" ]] || return 1
  return 0
}

if verify_outputs && [[ "${FORCE_BUILD}" -eq 0 ]]; then
  log "Overpass already built: ${PREFIX_DIR}"
  log "OK: ${PREFIX_DIR}/bin/dispatcher"
  log "OK: ${PREFIX_DIR}/bin/init_osm3s.sh"
  log "OK: ${PREFIX_DIR}/cgi-bin/interpreter"
  exit 0
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || return 1
  return 0
}

print_install_help() {
  cat >&2 <<'EOF'
Missing build dependencies.

Debian/Ubuntu:
  sudo apt-get update
  sudo apt-get install -y build-essential make wget tar bzip2 zlib1g-dev libexpat1-dev libbz2-dev liblz4-dev

Fedora:
  sudo dnf install -y gcc-c++ make wget tar bzip2 zlib-devel expat-devel bzip2-devel lz4-devel

Arch:
  sudo pacman -S --needed base-devel wget tar bzip2 zlib expat bzip2 lz4
EOF
}

missing=()
for c in tar make; do
  require_cmd "${c}" || missing+=("${c}")
done
if require_cmd wget; then
  :
elif require_cmd curl; then
  :
else
  missing+=("wget-or-curl")
fi
require_cmd g++ || missing+=("g++")

if (( ${#missing[@]} > 0 )); then
  log "Missing commands: ${missing[*]}"
  print_install_help
  exit 1
fi

compile_test() {
  local code="$1"
  shift
  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN
  printf '%b' "${code}" | g++ -x c++ - "$@" -o "${tmpdir}/a.out" >/dev/null 2>&1
}

if ! compile_test '#include <expat.h>\nint main(){return 0;}\n' -lexpat; then
  log "Missing libexpat development files."
  print_install_help
  exit 1
fi
if ! compile_test '#include <zlib.h>\nint main(){return 0;}\n' -lz; then
  log "Missing zlib development files."
  print_install_help
  exit 1
fi
if ! compile_test '#include <bzlib.h>\nint main(){return 0;}\n' -lbz2; then
  log "Missing libbz2 development files."
  print_install_help
  exit 1
fi

ENABLE_LZ4=0
if compile_test '#include <lz4.h>\nint main(){return 0;}\n' -llz4; then
  ENABLE_LZ4=1
else
  log "liblz4 not found; building without lz4 (optional, faster compression)."
fi

download() {
  local url="$1"
  local dest="$2"

  if [[ -s "${dest}" ]]; then
    if tar -tzf "${dest}" >/dev/null 2>&1; then
      log "Tarball already present: ${dest}"
      return 0
    fi
    log "Tarball exists but looks corrupt; re-downloading: ${dest}"
    rm -f -- "${dest}"
  fi

  local tmp="${dest}.partial"
  rm -f -- "${tmp}"
  log "Downloading: ${url}"
  if require_cmd wget; then
    wget -q -O "${tmp}" "${url}" || die "Download failed (wget): ${url}"
  else
    curl -fsSL -o "${tmp}" "${url}" || die "Download failed (curl): ${url}"
  fi
  mv -f -- "${tmp}" "${dest}"
}

download "${URL}" "${TARBALL}"

if [[ ! -d "${SRC_DIR}" ]]; then
  log "Extracting: ${TARBALL}"
  tar -xzf "${TARBALL}" -C "${SRC_BASE}"
fi

if [[ "${FORCE_BUILD}" -eq 1 ]]; then
  log "--force-build set: removing existing install prefix: ${PREFIX_DIR}"
  rm -rf -- "${PREFIX_DIR}"
fi

log "Building Overpass ${OVERPASS_VERSION} (log: ${BUILD_LOG})"

CONFIGURE_ARGS=(--prefix="${PREFIX_DIR}")
if [[ "${ENABLE_LZ4}" -eq 1 ]]; then
  CONFIGURE_ARGS+=(--enable-lz4)
fi

JOBS=1
if require_cmd nproc; then
  JOBS="$(nproc)"
fi

if ! (
  set -euo pipefail
  cd "${SRC_DIR}"
  if [[ -x ./configure ]]; then
    :
  elif require_cmd autoreconf; then
    autoreconf -i
  else
    die "Missing ./configure and autoreconf; install autoconf/automake/libtool and retry."
  fi
  if [[ -f Makefile ]]; then
    make distclean >/dev/null 2>&1 || true
  fi
  ./configure "${CONFIGURE_ARGS[@]}"
  make -j"${JOBS}"
  make install
) >"${BUILD_LOG}" 2>&1; then
  log "Build failed. Last 200 lines of ${BUILD_LOG}:"
  tail -n 200 "${BUILD_LOG}" >&2 || true
  exit 1
fi

if ! verify_outputs; then
  log "Build finished but expected outputs are missing under: ${PREFIX_DIR}"
  log "Expected:"
  log "  ${PREFIX_DIR}/bin/dispatcher"
  log "  ${PREFIX_DIR}/bin/init_osm3s.sh"
  log "  ${PREFIX_DIR}/cgi-bin/interpreter"
  log "Last 200 lines of ${BUILD_LOG}:"
  tail -n 200 "${BUILD_LOG}" >&2 || true
  exit 1
fi

cat > "${PREFIX_DIR}/.build_info" <<EOF
version=${OVERPASS_VERSION}
url=${URL}
built_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
lz4=${ENABLE_LZ4}
EOF

log "Build OK: ${PREFIX_DIR}"
log "OK: ${PREFIX_DIR}/bin/dispatcher"
log "OK: ${PREFIX_DIR}/bin/init_osm3s.sh"
log "OK: ${PREFIX_DIR}/cgi-bin/interpreter"
