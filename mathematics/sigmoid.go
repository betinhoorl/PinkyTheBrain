package mathematics

import (
	"math"
)

type activation func(float64) float64

// ISigmoid ...
type ISigmoid interface {
	Sigmoid(x float64) float64
	DerivativeSigmoid(y float64) float64
	Activate(x, w []float64, fn activation) float64
}

// Sigmoid neurônio tem uma função de ativação sigmóide
type sigmoid struct{}

// NewSigmoid ...
func NewSigmoid() ISigmoid {
	return &sigmoid{}
}

// Sigmoid ...
func (s *sigmoid) Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// DerivativeSigmoid
func (s *sigmoid) DerivativeSigmoid(y float64) float64 {
	return y * (1 - y)
}

// Activate função de ativação
func (s *sigmoid) Activate(x, w []float64, fn activation) float64 {
	raw := 0.0
	for j := 0; j < len(x); j++ {
		raw += x[j] * w[j]
	}
	return fn(raw)
}
