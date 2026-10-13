#!/bin/bash
# One-time install of CallAI on an AWS g4dn.xlarge (NVIDIA T4) with plain Ubuntu 22.04 or 24.04.
# These are the steps that were run on the first server (13 Oct 2026). Run from the package folder, as a user with sudo:
#     bash deploy/install_ubuntu.sh            # first run installs the NVIDIA driver and asks for a reboot
#     sudo reboot                              # only if it says so, then run the same command again
# Model files that are not in git (MODELS.md) must already be in place: models/whisper_otr_ft/, code/models/gemini_cls.pkl.
set -e
APP=$(pwd)
[ -f "$APP/otr_pipeline.py" ] || { echo "run this from the package folder (where otr_pipeline.py is)"; exit 1; }

# 1. Use the whole root disk (AWS images often leave the root partition at ~8 GB on a larger volume).
ROOTDEV=$(findmnt -n -o SOURCE /)
DISK=/dev/$(lsblk -no PKNAME "$ROOTDEV")
PART=$(echo "$ROOTDEV" | grep -o '[0-9]*$')
sudo growpart "$DISK" "$PART" 2>/dev/null && sudo resize2fs "$ROOTDEV" >/dev/null || true
df -h / | tail -1

# 2. System packages and the NVIDIA driver (R570 or newer; skipped if one is already working).
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git unzip curl libsndfile1 patchelf
if ! nvidia-smi >/dev/null 2>&1; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver-570-server
    echo "NVIDIA driver installed: run 'sudo reboot', then run this script again."
    exit 0
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 3. Python 3.12 environment (Ubuntu 22.04 ships 3.10, too old for the pinned packages), via uv.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.12 .venv
. .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'torch does not see the GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# 4. llama.cpp, Qwen3-4B, bge-m3. The prebuilt llama-server needs glibc 2.38 (Ubuntu 24.04); on 22.04 build it here.
if ! python setup.py; then
    if ls llama/*/llama-server >/dev/null 2>&1 && ! llama/*/llama-server --version >/dev/null 2>&1; then
        echo "prebuilt llama-server cannot run on this Ubuntu: building it from source (about 25 minutes)"
        bash deploy/build_llama_from_source.sh
        python setup.py
    else
        exit 1
    fi
fi

# 5. Quick check of the classifier on the CPU.
python test_classifier.py | tail -4
echo "install complete: start serve.py (API) or worker.py (queue mode); see README.md and deploy/README.md"
