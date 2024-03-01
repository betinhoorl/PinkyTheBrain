package mathematics

import "math"

// ITanh ...
type ITanh interface {
	Tanh(x float64) float64
}

type tanh struct{}

// NewTanh ...
func NewTanh() ITanh {
	return &tanh{}
}

func (t *tanh) Tanh(x float64) float64 {
	return math.Tanh(x)
}
