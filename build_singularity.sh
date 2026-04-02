#!/bin/bash
# Build the cosmos-policy Singularity image.
# Run this as a Slurm job:
#   sbatch build_singularity.sh
# Or interactively on a compute node:
#   bash build_singularity.sh
#
# Requires: singularity with --fakeroot support (ask your admin if unsure).
# If --fakeroot is unavailable, see the FALLBACK section at the bottom.

#SBATCH --job-name=build-cosmos-policy
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.log

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DEF_FILE="${REPO_DIR}/cosmos-policy.def"
SIF_FILE="${REPO_DIR}/cosmos-policy.sif"

echo "=== Building cosmos-policy Singularity image ==="
echo "Definition : ${DEF_FILE}"
echo "Output     : ${SIF_FILE}"
echo "Started    : $(date)"
echo

# Remove a stale/incomplete image if present
if [[ -f "${SIF_FILE}" ]]; then
    echo "Removing existing image: ${SIF_FILE}"
    rm -f "${SIF_FILE}"
fi

singularity build --fakeroot "${SIF_FILE}" "${DEF_FILE}"

echo
echo "=== Build complete: $(date) ==="
echo "Image size : $(du -sh "${SIF_FILE}" | cut -f1)"
echo
echo "To launch an interactive session run:"
echo "  ./run_cosmos.sh"

# ---------------------------------------------------------------------------
# FALLBACK: if --fakeroot is not enabled on your cluster, build via podman
# by forcing image storage onto node-local /tmp (avoids the NFS xattr error).
# Uncomment the block below and comment out the singularity build line above.
# ---------------------------------------------------------------------------
#
# LOCAL_STORE="/tmp/${USER}-podman-build"
# mkdir -p "${LOCAL_STORE}"
# STORAGE_CONF="${LOCAL_STORE}/storage.conf"
# cat > "${STORAGE_CONF}" <<EOF
# [storage]
# driver = "overlay"
# graphRoot = "${LOCAL_STORE}/storage"
# runRoot  = "${LOCAL_STORE}/run"
# [storage.options]
# mount_program = "/usr/bin/fuse-overlayfs"
# EOF
#
# export CONTAINERS_STORAGE_CONF="${STORAGE_CONF}"
#
# echo "Building Docker image with podman (local /tmp storage)..."
# podman build -t cosmos-policy "${REPO_DIR}/docker"
#
# echo "Exporting OCI archive..."
# OCI_TAR="${REPO_DIR}/cosmos-policy-oci.tar"
# podman save cosmos-policy -o "${OCI_TAR}"
#
# echo "Converting to Singularity .sif..."
# singularity build "${SIF_FILE}" "docker-archive:${OCI_TAR}"
#
# echo "Cleaning up..."
# rm -f "${OCI_TAR}"
# rm -rf "${LOCAL_STORE}"
#
# echo "Done. Image at: ${SIF_FILE}"
