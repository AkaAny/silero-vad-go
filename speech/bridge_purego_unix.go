//go:build darwin || linux

package speech

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	bridgeErrBufLen = 1024
)

var (
	loadBridgeOnce sync.Once
	loadBridgeErr  error

	bridgeCreateFn  func(modelPath *byte, sampleRate int32, logLevel int32, outHandle *unsafe.Pointer, errBuf *byte, errBufLen uintptr) int32
	bridgeInferFn   func(handle unsafe.Pointer, pcm *float32, pcmLen uintptr, stateInOut *float32, probOut *float32, errBuf *byte, errBufLen uintptr) int32
	bridgeDestroyFn func(handle unsafe.Pointer)
)

type ortBridge struct {
	handle unsafe.Pointer
}

func newOrtBridge(modelPath string, sampleRate int, logLevel int32) (*ortBridge, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("invalid sample rate")
	}

	if err := loadBridge(); err != nil {
		return nil, err
	}

	modelPathBytes := append([]byte(modelPath), 0)
	errBuf := make([]byte, bridgeErrBufLen)
	var handle unsafe.Pointer

	rc := bridgeCreateFn(
		&modelPathBytes[0],
		int32(sampleRate),
		logLevel,
		&handle,
		&errBuf[0],
		uintptr(len(errBuf)),
	)
	if rc != 0 {
		return nil, fmt.Errorf("%s", cStringBytes(errBuf))
	}
	if handle == nil {
		return nil, fmt.Errorf("native bridge returned nil handle")
	}

	return &ortBridge{handle: handle}, nil
}

func (b *ortBridge) Infer(pcm []float32, state []float32) (float32, error) {
	if b == nil || b.handle == nil {
		return 0, fmt.Errorf("native bridge is not initialized")
	}
	if len(pcm) == 0 {
		return 0, fmt.Errorf("pcm should not be empty")
	}
	if len(state) != stateLen {
		return 0, fmt.Errorf("invalid state length: got %d, want %d", len(state), stateLen)
	}

	errBuf := make([]byte, bridgeErrBufLen)
	var prob float32

	rc := bridgeInferFn(
		b.handle,
		&pcm[0],
		uintptr(len(pcm)),
		&state[0],
		&prob,
		&errBuf[0],
		uintptr(len(errBuf)),
	)
	if rc != 0 {
		return 0, fmt.Errorf("%s", cStringBytes(errBuf))
	}

	return prob, nil
}

func (b *ortBridge) Destroy() {
	if b == nil || b.handle == nil {
		return
	}
	bridgeDestroyFn(b.handle)
	b.handle = nil
}

func loadBridge() error {
	loadBridgeOnce.Do(func() {
		var libHandle uintptr

		var err error
		candidates := bridgeCandidates()
		for _, candidate := range candidates {
			libHandle, err = purego.Dlopen(candidate, purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if err == nil {
				break
			}
		}
		if libHandle == 0 {
			loadBridgeErr = fmt.Errorf("failed to load silero vad bridge library: %w", err)
			return
		}

		purego.RegisterLibFunc(&bridgeCreateFn, libHandle, "VadCreate")
		purego.RegisterLibFunc(&bridgeInferFn, libHandle, "VadInfer")
		purego.RegisterLibFunc(&bridgeDestroyFn, libHandle, "VadDestroy")
	})

	return loadBridgeErr
}

func bridgeCandidates() []string {
	ldLibraryPath := os.Getenv("LD_LIBRARY_PATH")
	fmt.Println("LD_LIBRARY_PATH:", ldLibraryPath)
	override := os.Getenv("SILERO_VAD_BRIDGE_PATH")
	if override != "" {
		return []string{override}
	}

	fileName := "libsilero_vad_bridge.so"
	if runtime.GOOS == "darwin" {
		fileName = "libsilero_vad_bridge.dylib"
	}

	candidates := []string{
		fileName,
		filepath.Join(".", fileName),
		filepath.Join(".", "native", "build", fileName),
	}

	execPath, err := os.Executable()
	if err == nil {
		execDir := filepath.Dir(execPath)
		candidates = append(candidates, filepath.Join(execDir, fileName))
	}

	return candidates
}

func cStringBytes(buf []byte) string {
	for i := range buf {
		if buf[i] == 0 {
			return string(buf[:i])
		}
	}
	return string(buf)
}
