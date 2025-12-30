#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Compress ImmoScout24 raw *.jsonl files to *.jsonl.zst to save disk space.

Usage:
  get_data/past/immoscout24/scripts/compress_raw_jsonl.sh [options]

Options:
  --dir PATH     Directory to scan (default: get_data/past/immoscout24/data/raw)
  --level N      zstd compression level (default: 19; try 10-15 for speed)
  --keep         Keep original .jsonl files (default: delete after verified)
  --force        Re-compress even if .zst exists (overwrites)
  --dry-run      Print actions without writing
  -h, --help     Show this help

Decompress examples:
  zstd -d file.jsonl.zst -o file.jsonl
  unzstd file.jsonl.zst
  zstdcat file.jsonl.zst | head
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
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
fi

RAW_DIR="${REPO_ROOT}/get_data/past/immoscout24/data/raw"
LEVEL="19"
KEEP="0"
FORCE="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      RAW_DIR="$2"
      shift 2
      ;;
    --level)
      LEVEL="$2"
      shift 2
      ;;
    --keep)
      KEEP="1"
      shift
      ;;
    --force)
      FORCE="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1 (try --help)"
      ;;
  esac
done

command -v zstd >/dev/null 2>&1 || die "zstd not found in PATH"
[[ -d "${RAW_DIR}" ]] || die "Directory not found: ${RAW_DIR}"

log "Scanning: ${RAW_DIR}"

mapfile -d '' files < <(find "${RAW_DIR}" -type f -name '*.jsonl' -print0)
if [[ "${#files[@]}" -eq 0 ]]; then
  log "No .jsonl files found."
  exit 0
fi

log "Found ${#files[@]} .jsonl files."

for src in "${files[@]}"; do
  dst="${src}.zst"
  tmp="${dst}.tmp.$$"

  if [[ -e "${dst}" && "${FORCE}" != "1" ]]; then
    log "Skip (already exists): ${dst}"
    continue
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${KEEP}" == "1" ]]; then
      log "DRY RUN: zstd -${LEVEL} '${src}' -> '${dst}' (keep original)"
    else
      log "DRY RUN: zstd -${LEVEL} '${src}' -> '${dst}' (remove original after verify)"
    fi
    continue
  fi

rm -f -- "${tmp}"
trap 'rm -f -- "${tmp}"' RETURN

log "Compress: ${src} -> ${dst} (level ${LEVEL})"
zstd "-${LEVEL}" -T0 -q --no-progress -f -o "${tmp}" "${src}"
zstd -t -q "${tmp}"
mv -f -- "${tmp}" "${dst}"

if [[ "${KEEP}" != "1" ]]; then
  rm -f -- "${src}"
fi

trap - RETURN
done

log "Done."
