//go:build darwin

package speech

import (
	"fmt"
)

func (sd *Detector) Infer(pcm []float32) (float32, error) {
	if len(pcm) == 0 {
		return 0, fmt.Errorf("failed to infer: empty pcm")
	}
	return sd.bridge.Infer(pcm, sd.state[:])
}
