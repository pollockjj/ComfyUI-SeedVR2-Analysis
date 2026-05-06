#!/usr/bin/env bash
# Bootstrap the vendored DOVER tree (repo + pretrained weight) inside the pack.
# Run once after cloning the pack into a ComfyUI custom_nodes/ directory.
#
# Vendoring is operator-driven and not tracked in the pack repo because:
#   - DOVER is a separate upstream (VQAssessment/DOVER) — pin to a SHA the
#     pack's nodes.py was tested against.
#   - DOVER.pth weights are 239 MB; not a normal-git artifact.
#
# Pinned SHA + weight digest are recorded in nodes.py's tool_provenance
# block at runtime; verify post-bootstrap that the digests match.

set -euo pipefail

PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${PACK_ROOT}/vendor"
DOVER_DIR="${VENDOR_DIR}/DOVER"
DOVER_PIN="f1ddc96215bc7fbcf8f315c65d47905f339c3419"
DOVER_WEIGHT_URL="https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth"
DOVER_WEIGHT_SHA256="f4a42c0bbc94c94dd7409e7f40887d44c5c30314d1d09e7edf03cc35813b4838"

mkdir -p "${VENDOR_DIR}"

if [ ! -d "${DOVER_DIR}/.git" ]; then
    echo "[bootstrap] cloning VQAssessment/DOVER → ${DOVER_DIR}"
    git clone https://github.com/VQAssessment/DOVER.git "${DOVER_DIR}"
fi

echo "[bootstrap] checking out pinned DOVER SHA ${DOVER_PIN}"
git -C "${DOVER_DIR}" fetch origin
git -C "${DOVER_DIR}" checkout "${DOVER_PIN}"

WEIGHT_PATH="${DOVER_DIR}/pretrained_weights/DOVER.pth"
mkdir -p "$(dirname "${WEIGHT_PATH}")"
if [ ! -f "${WEIGHT_PATH}" ]; then
    echo "[bootstrap] downloading DOVER.pth → ${WEIGHT_PATH}"
    curl -fL "${DOVER_WEIGHT_URL}" -o "${WEIGHT_PATH}"
fi

OBSERVED_SHA="$(sha256sum "${WEIGHT_PATH}" | awk '{print $1}')"
if [ "${OBSERVED_SHA}" != "${DOVER_WEIGHT_SHA256}" ]; then
    echo "[bootstrap] FATAL: DOVER.pth sha256 mismatch" >&2
    echo "  expected: ${DOVER_WEIGHT_SHA256}" >&2
    echo "  observed: ${OBSERVED_SHA}" >&2
    exit 1
fi

echo "[bootstrap] OK — DOVER pinned at ${DOVER_PIN}, weights sha256 ${DOVER_WEIGHT_SHA256}"
