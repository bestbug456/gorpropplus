package gorpropplus

import (
	"testing"
	"math"
)

func round(num float64) int {
    return int(num + math.Copysign(0.5, num))
}

func toFixed(num float64, precision int) float64 {
    output := math.Pow(10, float64(precision))
    return float64(round(num * output)) / output
}

func TestIperbolicTangent(t *testing.T) {

	neuronValue := -0.3531139100
	activatedNeuronOuput := IperbolicTangent(neuronValue)
	correctActivatedNeuronOuput := -0.3391342216
	if toFixed(activatedNeuronOuput, 6) != toFixed(correctActivatedNeuronOuput, 6) {
		t.Fatalf("Error: the activatedNeuronOuput is %f instead of %f", activatedNeuronOuput, correctActivatedNeuronOuput)
	}
}

func TestDerivateIperbolicTangent(t *testing.T) {
	
	neuronValue := -0.7606964211
	activatedNeuronOuput := DerivateIperbolicTangent(neuronValue)
	correctActivatedNeuronOuput := 0.4213409550
	if toFixed(activatedNeuronOuput, 6) != toFixed(correctActivatedNeuronOuput, 6) {
		t.Fatalf("Error: the derivate activatedNeuronOuput is %f instead of %f", activatedNeuronOuput, correctActivatedNeuronOuput)
	}
}

func TestLogistic(t *testing.T) {
	
	neuronValue := 1.8052738837
	activatedNeuronOuput := Logistic(neuronValue)
	correctActivatedNeuronOuput := 0.8587897097
	if toFixed(activatedNeuronOuput, 6)!= toFixed(correctActivatedNeuronOuput, 6) {
		t.Fatalf("Error: the activatedNeuronOuput is %f instead of %f", activatedNeuronOuput, correctActivatedNeuronOuput)
	}
}

func TestDerivateLogistic(t *testing.T) {
	
	neuronValue := 0.4640093329
	activatedNeuronOuput := DerivateLogistic(neuronValue)
	correctActivatedNeuronOuput := 0.2487046719
	if toFixed(activatedNeuronOuput, 6) != toFixed(correctActivatedNeuronOuput, 6) {
		t.Fatalf("Error: the derivate activatedNeuronOuput is %f instead of %f", activatedNeuronOuput, correctActivatedNeuronOuput)
	}
}
