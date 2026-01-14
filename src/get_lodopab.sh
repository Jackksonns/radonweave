#!/usr/bin/env bash
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
# Fast & robust LoDoPaB-CT downloader
# - Downloads all 9 official files (6 zips + 3 csvs)
# - Uses aria2c (multi-connection) with resume & overwrite
# - Verifies MD5 checksums
# - Optional auto-unzip (default: ON)

set -euo pipefail

DATA_DIR="${1:-/root/autodl-tmp/datasets/lodopab}"   # target dir (can pass a custom path as $1)
AUTO_UNZIP="${AUTO_UNZIP:-1}"                        # 1=unzip after download, 0=skip unzip
THREADS="${THREADS:-16}"                             # aria2 connections per file
SPLIT="${SPLIT:-16}"                                 # aria2 split per file
BASE_URL="https://zenodo.org/records/3384092/files"  # official source (Zenodo)  

echo "Target directory: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "[INFO] aria2c not found. Installing..."
  if command -v apt >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt update && apt install -y aria2
  else
    echo "[WARN] apt not available; fallback to wget (slower)."
  fi
fi

FILES=(
  "observation_train.zip"
  "observation_validation.zip"
  "observation_test.zip"
  "ground_truth_train.zip"
  "ground_truth_validation.zip"
  "ground_truth_test.zip"
  "patient_ids_rand_train.csv"
  "patient_ids_rand_validation.csv"
  "patient_ids_rand_test.csv"
)

download_with_aria2() {
  local f="$1"
  aria2c \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --continue=true \
    --file-allocation=none \
    -x "${THREADS}" -s "${SPLIT}" -k 1M \
    -d "${DATA_DIR}" "${BASE_URL}/${f}"
}

download_with_wget() {
  local f="$1"
  wget -O "${DATA_DIR}/${f}" -c "${BASE_URL}/${f}"
}

echo "[INFO] Starting downloads..."
for f in "${FILES[@]}"; do
  echo "  -> ${f}"
  if command -v aria2c >/dev/null 2>&1; then
    download_with_aria2 "${f}"
  else
    download_with_wget "${f}"
  fi
done

echo "[INFO] Downloads finished."

# MD5s for 6 ZIPs (CSV MD5s can vary across mirrors; they are tiny so rarely corrupt)
# Source: Zenodo LoDoPaB-CT record (files table).
# observation_* and ground_truth_* MD5 verified as of the official record page.
cat > lodopab.md5 << 'EOF'
ecc655767fbe3d40908ca823921f4c7f  ground_truth_test.zip
f06829ccf2b9bb817abd093ce490b2c7  ground_truth_train.zip
666c36f403734842f14ca8811a63b8f7  ground_truth_validation.zip
9ae6b053bb1faa94d573311af8ec67b2  observation_test.zip
c4fde721ac5862469812de49f98fb0a3  observation_train.zip
3cff406e09c59774912655eb7a72cfcf  observation_validation.zip
EOF

echo "[INFO] Verifying MD5 for the 6 ZIP archives..."
md5sum -c lodopab.md5

if [[ "${AUTO_UNZIP}" == "1" ]]; then
  echo "[INFO] Unzipping (this will take a while and ~115 GB disk space)..."
  # Install unzip if needed
  if ! command -v unzip >/dev/null 2>&1; then
    if command -v apt >/dev/null 2>&1; then
      apt update && apt install -y unzip
    else
      echo "[WARN] unzip not available and apt missing; skipping unzip."
      AUTO_UNZIP="0"
    fi
  fi

  if [[ "${AUTO_UNZIP}" == "1" ]]; then
    # -o: overwrite silently; -q: quiet
    for z in observation_*.zip ground_truth_*.zip; do
      echo "  -> extracting ${z}"
      unzip -o -q "${z}"
    done
    echo "[INFO] Unzip completed."
  fi
fi

echo
echo "========================================================"
echo "LoDoPaB files are in: ${DATA_DIR}"
echo "If you use DIVαℓ, you can export DIVAL_DATA_PATH to make it discoverable:"
echo "  export DIVAL_DATA_PATH=\"$(dirname "${DATA_DIR}")\""
echo "Then re-run your training, e.g.:"
echo "  python trainer.py"
echo "========================================================"
