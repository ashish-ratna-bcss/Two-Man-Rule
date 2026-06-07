#!/usr/bin/env bash
# Fetch the OSNet appearance Re-ID model (ONNX) used by models/reid_extractor.py.
#
# Run once on the server (needs network). Until the file exists, the tracker falls
# back to its pose-keypoint Re-ID, so this is non-blocking.
#
# Default target matches config.REID_MODEL_PATH:
#     models/weights/osnet_x0_25_msmt17.onnx
#
# Override the source with REID_MODEL_URL if your mirror differs:
#     REID_MODEL_URL="https://your.mirror/osnet_x0_25_msmt17.onnx" ./scripts/get_reid_model.sh
set -euo pipefail

cd "$(dirname "$0")/.."
DEST="models/weights/osnet_x0_25_msmt17.onnx"
mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
  echo "[get_reid_model] Already present: $DEST"
  exit 0
fi

# Primary path: export from boxmot/torchreid if available (most reliable, no fragile URL).
if python3 -c "import boxmot" 2>/dev/null; then
  echo "[get_reid_model] boxmot found — exporting osnet_x0_25_msmt17 to ONNX..."
  python3 - "$DEST" <<'PY'
import sys, shutil, glob, os
from pathlib import Path
from boxmot.appearance.reid_auto_backend import ReidAutoBackend  # noqa
from boxmot.utils import WEIGHTS
# Trigger boxmot's own download + ONNX export for the named model.
from boxmot.appearance.backends.onnx_backend import ONNXBackend  # noqa
import torch
from boxmot.appearance.reid_model_factory import build_model, get_model_name, load_pretrained_weights
name = "osnet_x0_25_msmt17"
pt = Path(WEIGHTS) / f"{name}.pt"
if not pt.exists():
    from boxmot.appearance.reid_auto_backend import ReidAutoBackend
    ReidAutoBackend(weights=pt, device=torch.device("cpu"), half=False)  # downloads .pt
model = build_model(get_model_name(pt), num_classes=1, pretrained=False, use_gpu=False)
load_pretrained_weights(model, str(pt)); model.eval()
dummy = torch.randn(1, 3, 256, 128)
onnx_tmp = str(Path(WEIGHTS) / f"{name}.onnx")
torch.onnx.export(model, dummy, onnx_tmp, input_names=["images"], output_names=["features"],
                  dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}}, opset_version=12)
shutil.copy(onnx_tmp, sys.argv[1])
print("exported:", sys.argv[1])
PY
  echo "[get_reid_model] Done: $DEST"
  exit 0
fi

# Fallback path: direct download from a mirror.
URL="${REID_MODEL_URL:-}"
if [[ -z "$URL" ]]; then
  cat >&2 <<EOF
[get_reid_model] No boxmot install and no REID_MODEL_URL set.
Provide a mirror that hosts osnet_x0_25_msmt17.onnx, e.g.:
    REID_MODEL_URL="https://<mirror>/osnet_x0_25_msmt17.onnx" $0
Or:  pip install boxmot   then re-run this script (exports ONNX automatically).
The system runs fine without it (pose-keypoint Re-ID fallback) until placed.
EOF
  exit 1
fi

echo "[get_reid_model] Downloading $URL -> $DEST"
if command -v curl >/dev/null; then
  curl -fL "$URL" -o "$DEST"
else
  wget -O "$DEST" "$URL"
fi
echo "[get_reid_model] Done: $DEST"
