//go:build !darwin && !linux

package speech

import "fmt"

type ortBridge struct{}

func newOrtBridge(modelPath string, sampleRate int, logLevel int32) (*ortBridge, error) {
	return nil, fmt.Errorf("unsupported platform: purego bridge currently supports darwin/linux only")
}

func (b *ortBridge) Infer(pcm []float32, state []float32) (float32, error) {
	return 0, fmt.Errorf("unsupported platform")
}

func (b *ortBridge) Destroy() {}
