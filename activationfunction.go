package gorpropplus

import (
	"math"
)

func IperbolicTangent(neuronValue float64) float64 {
	return math.Tanh(neuronValue)
}

func DerivateIperbolicTangent(neuronValue float64) float64 {
	return 1.0 - math.Pow(neuronValue, 2.0)
}

func Logistic(neuronValue float64) float64 {
	return 1.0 / (1.0 + math.Exp(-neuronValue))
}

func DerivateLogistic(neuronValue float64) float64 {
	return neuronValue * (1.0 - neuronValue)
}
