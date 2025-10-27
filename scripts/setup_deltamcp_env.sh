#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-deltamcp-star}
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${ENV_FILE:-"$SCRIPT_DIR/../env/deltamcp-star-environment.yaml"}
MICROMAMBA_ROOT=${MICROMAMBA_ROOT:-"$HOME/micromamba"}
MICROMAMBA_BIN=${MICROMAMBA_BIN:-"$MICROMAMBA_ROOT/micromamba"}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-"$HOME/.local/share/mamba"}

if [ ! -x "$MICROMAMBA_BIN" ]; then
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64|amd64)
      MICROMAMBA_PLATFORM="linux-64"
      ;;
    aarch64|arm64)
      MICROMAMBA_PLATFORM="linux-aarch64"
      ;;
    *)
      echo "Unsupported architecture: $ARCH" >&2
      exit 1
      ;;
  esac

  mkdir -p "$MICROMAMBA_ROOT"
  curl -Ls "https://micro.mamba.pm/api/micromamba/${MICROMAMBA_PLATFORM}/latest" | tar -xvj -C "$MICROMAMBA_ROOT" --strip-components=1 bin/micromamba
fi

"$MICROMAMBA_BIN" shell init -s bash -p "$MAMBA_ROOT_PREFIX" >/dev/null 2>&1 || true
"$MICROMAMBA_BIN" create -y -n "$ENV_NAME" -f "$ENV_FILE"
