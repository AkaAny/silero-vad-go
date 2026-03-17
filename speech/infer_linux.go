//go:build !darwin

package speech

import (
	"fmt"
)

func (sd *Detector) Infer(samples []float32) (float32, error) {
	if len(samples) == 0 {
		return 0, fmt.Errorf("failed to infer: empty pcm")
	}

	pcm := samples
	if sd.currSample > 0 {
		// Append context from previous iteration.
		pcm = append(sd.ctx[:], samples...)
	}
	// Save the last contextLen samples as context for the next iteration.
	copy(sd.ctx[:], samples[len(samples)-contextLen:])

	return sd.bridge.Infer(pcm, sd.state[:])
}
