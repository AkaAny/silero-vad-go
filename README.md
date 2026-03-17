<h1 align="center">
  <br>
  silero-vad-go
  <br>
</h1>
<h4 align="center">A simple Golang (purego + ONNX Runtime) speech detector powered by Silero VAD</h4>
<p align="center">
  <a href="https://pkg.go.dev/github.com/streamer45/silero-vad-go"><img src="https://pkg.go.dev/badge/github.com/streamer45/silero-vad-go.svg" alt="Go Reference"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>
<br>

### Requirements

- [Golang](https://go.dev/doc/install) >= v1.21
- A C compiler (e.g. Clang/GCC)
- [CMake](https://cmake.org/) >= 3.16
- ONNX Runtime (v1.18.1)
- A [Silero VAD](https://github.com/snakers4/silero-vad) model (v5)

### Development

This project now builds a dedicated native dynamic library with CMake and loads it from Go through `purego`.

#### Build native bridge library

```sh
cmake -S native -B native/build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build native/build --config Release
```

After build:

- Darwin: `native/build/libsilero_vad_bridge.dylib`
- Linux: `native/build/libsilero_vad_bridge.so`

You can point Go runtime to this library with:

```sh
export SILERO_VAD_BRIDGE_PATH=/absolute/path/to/libsilero_vad_bridge.{dylib|so}
```

If `SILERO_VAD_BRIDGE_PATH` is not set, the runtime will try default loader paths and `./native/build/`.

#### Install ONNX Runtime (required for native build)

You need both:

- headers: `onnxruntime_c_api.h` (under `include/`)
- shared library: `libonnxruntime.so` (Linux) or `libonnxruntime.dylib` (Darwin)

Use official prebuilt packages from ONNX Runtime releases.

##### Linux (x64 example)

```sh
mkdir -p /tmp/onnxruntime && cd /tmp/onnxruntime
curl -L -o onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz
tar -xzf onnxruntime.tgz

# point CMake to extracted root
export ONNXRUNTIME_ROOT=/tmp/onnxruntime/onnxruntime-linux-x64-1.18.1

# optional: runtime loader path, if libonnxruntime.so is not in system path
export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH"
```

##### Darwin (Apple Silicon / Intel)

```sh
mkdir -p /tmp/onnxruntime && cd /tmp/onnxruntime

# Apple Silicon (arm64)
curl -L -o onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-osx-arm64-1.18.1.tgz

# Intel (x86_64), use this instead:
# curl -L -o onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-osx-x86_64-1.18.1.tgz

tar -xzf onnxruntime.tgz
export ONNXRUNTIME_ROOT=$(PWD)/onnxruntime/onnxruntime-osx-arm64-1.18.1

# optional: runtime loader path, if libonnxruntime.dylib is not in system path
export DYLD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:$DYLD_LIBRARY_PATH"
```

Then build native bridge:

```sh
cmake -S native -B native/build -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build native/build --config Release
```

Tip: if runtime still cannot find ONNX Runtime, copy `libonnxruntime.so` / `libonnxruntime.dylib`
to the same directory as `libsilero_vad_bridge.so` / `libsilero_vad_bridge.dylib`.

### License

MIT License - see [LICENSE](LICENSE) for full text

