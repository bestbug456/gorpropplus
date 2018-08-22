package gorpropplus

import (
	"math"
)

// SSE is the "sum of squared errors" error function
func SSE(nnResult float64, expected float64) float64 {
	return 0.5 * math.Pow(expected-nnResult, 2.0)
}

func DerivateSSE(nnResult float64, expected float64) float64 {
	return nnResult - expected
}

// CE is the "cross-entropy" error function
func CE(nnResult float64, expected float64) float64 {
	v := expected * math.Log(nnResult)
	v2 := (1.0 - expected) * math.Log(1.0-nnResult)
	return -(v + v2)
}

func DerivateCE(nnResult float64, expected float64) float64 {
	v := (1.0 - expected) / (1.0 - nnResult)
	v2 := expected / nnResult
	return v - v2
}