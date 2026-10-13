#!/bin/bash
# Build llama.cpp b11064 with CUDA from source, for Linux systems older than Ubuntu 24.04 (glibc < 2.38), where the
# prebuilt llama-server from setup.py fails with "GLIBC_2.38 not found". Installs the CUDA 12.8 toolkit (Ubuntu 22.04
# repo), builds llama-server for the T4 (compute 7.5), and installs it into ./llama/ of the package. ~25 minutes.
#     bash deploy/build_llama_from_source.sh        (run from the package folder)
set -e
APP=$(pwd)
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-8 cmake build-essential patchelf
export PATH=/usr/local/cuda-12.8/bin:$PATH
rm -rf /tmp/llama.cpp && git clone -q --depth 1 --branch b11064 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(nproc)" --target llama-server
D="$APP/llama/llama-b11064"
rm -rf "$APP/llama" && mkdir -p "$D"
cp build/bin/llama-server "$D/"
find build -name "*.so*" -exec cp -P {} "$D/" \;
for f in "$D"/llama-server "$D"/*.so*; do [ -L "$f" ] || patchelf --set-rpath '$ORIGIN' "$f"; done
"$D/llama-server" --version
echo "llama-server built into $D"
