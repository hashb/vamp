# Dockerfile for building VAMP WebAssembly targets with Emscripten
#
# Build:
#   docker build -f docker/wasm.dockerfile -t vamp-wasm .
#
# Extract build artifacts:
#   docker run --rm vamp-wasm cat /vamp/build-wasm/vamp_planning.mjs > build-wasm/vamp_planning.mjs
#   docker run --rm vamp-wasm cat /vamp/build-wasm/vamp_planning.wasm > build-wasm/vamp_planning.wasm
#   docker run --rm vamp-wasm cat /vamp/build-wasm/vamp_smoke.mjs > build-wasm/vamp_smoke.mjs
#   docker run --rm vamp-wasm cat /vamp/build-wasm/vamp_smoke.wasm > build-wasm/vamp_smoke.wasm
#
# Or copy everything out:
#   docker create --name vamp-wasm-tmp vamp-wasm
#   docker cp vamp-wasm-tmp:/vamp/build-wasm ./build-wasm
#   docker rm vamp-wasm-tmp
#
# Run smoke test inside container:
#   docker run --rm vamp-wasm node scripts/wasm_smoke.js
#   docker run --rm vamp-wasm node scripts/wasm_planning.js

FROM emscripten/emsdk:3.1.74

ENV DEBIAN_FRONTEND=noninteractive

# Install Eigen3 headers (header-only, works for cross-compilation)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . /vamp
WORKDIR /vamp

# Configure with Emscripten toolchain
# Point Eigen3_DIR to the system-installed Eigen3 CMake config
RUN emcmake cmake -S . -B build-wasm \
    -DCMAKE_BUILD_TYPE=Release \
    -DVAMP_BUILD_PYTHON_BINDINGS=OFF \
    -DEigen3_DIR=/usr/lib/cmake/eigen3

# Build all WASM targets
RUN cmake --build build-wasm -j$(nproc)

# Verify builds succeeded
RUN ls -la build-wasm/vamp_smoke.mjs build-wasm/vamp_smoke.wasm \
          build-wasm/vamp_planning.mjs build-wasm/vamp_planning.wasm

# Run smoke test
RUN node scripts/wasm_smoke.js

CMD ["node", "scripts/wasm_planning.js"]
