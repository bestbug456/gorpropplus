package gorpropplus

import (
	"math"
)

// IperbolicTangent return the tanh value
func IperbolicTangent(neuronValue float64) float64 {
	return math.Tanh(neuronValue)
}

// DerivateIperbolicTangent return the derivate of tanh
func DerivateIperbolicTangent(neuronValue float64) float64 {
	return 1.0 - math.Pow(neuronValue, 2.0)
}

// Logistic is the classic logistic function
func Logistic(neuronValue float64) float64 {
	return 1.0 / (1.0 + math.Exp(-neuronValue))
}

// DerivateLogistic is the derivate of logistic
func DerivateLogistic(neuronValue float64) float64 {
	return neuronValue * (1.0 - neuronValue)
}
