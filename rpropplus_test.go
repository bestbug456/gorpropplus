package gorpropplus

import (
	"testing"
)

func TestNewNeuralNetworkAndSetup(t *testing.T) {
	args := NeuralNetworkArguments{
		HiddenLayer:        []int{2},
		InputSize:          3,
		OutputSize:         2,
		Threshold:          0.01,
		StepMax:            100,
		LifeSignStep:       100,
		LinearOutput:       false,
		Minus:              10,
		Plus:               100,
		ActivationFunction: Logistic,
		DerivateActivation: DerivateLogistic,
		ErrorFunction:      SSE,
		DerivateError:      DerivateSSE,
	}

	NN, err := NewNeuralNetworkAndSetup(args)
	if err != nil {
		t.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	if NN == nil {
		t.Fatalf("Error: neural network is nil")
	}
	if NN.ActivationFunction == nil {
		t.Fatalf("Error: Activation function shoul be set but is not.")
	}
	if len(NN.Weights) != 2 {
		t.Fatalf("Error: weights should be 2 but is %d", len(NN.Weights))
	}
	for i := 0; i < len(NN.Weights); i++ {
		if len(NN.Weights[i]) != 4 && len(NN.Weights[i]) != 3 {
			t.Fatalf("Exspected %d of dimension 4 or 3 but is %d", i, len(NN.Weights[i]))
		}
	}
}

func TestNewNeuralNetworkAndSetupNoHiddenLayer(t *testing.T) {
	NN, err := createNewStandardNNNoHiddenlayer()
	if err != nil {
		t.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	if NN == nil {
		t.Fatalf("Error: neural network is nil")
	}
	if NN.ActivationFunction == nil {
		t.Fatalf("Error: Activation function shoul be set but is not.")
	}
	if len(NN.Weights) != 1 {
		t.Fatalf("Error: weights should be 1 but is %d", len(NN.Weights))
	}
	for i := 0; i < len(NN.Weights); i++ {
		if len(NN.Weights[i]) != 5 {
			t.Fatalf("Exspected %d of dimension 5 but is %d", i, len(NN.Weights[i]))
		}
	}
}

func TestActivationNeuronAndDerivateNoHiddenLayer(t *testing.T) {
	nn, err := createNewStandardNNNoHiddenlayer()
	if err != nil {
		t.Fatalf("Error while creating a new neural network: %s", err.Error())
	}

	_, derivate, err := nn.activationNeuronAndDerivate(
		[][]float64{
			[]float64{1, 0, 1, 0, 0},
			[]float64{1, 1, 0, 0, 0},
			[]float64{1, 0, 0, 0, 1},
			[]float64{1, 1, 0, 0, 0},
			[]float64{1, 1, 0, 0, 0},
		},
		[][]float64{
			[]float64{0.65520215243},
			[]float64{1.20945613399},
			[]float64{-0.01284607794},
			[]float64{-1.73709863158},
			[]float64{-0.24877700455},
		})
	if err != nil {
		t.Fatalf("Error while activate: %s", err.Error())
	}

	if derivate[0][0] != 0.22588630165319862 {
		t.Fatalf("Error: exspected 0.22588630165319862 but having %v", derivate[0][0])
	}
	if derivate[1][0] != 0.1161618363924593 {
		t.Fatalf("Error: exspected 0.1161618363924593 but having %v", derivate[1][0])
	}
	if derivate[2][0] != 0.23995386943975403 {
		t.Fatalf("Error: exspected 0.23995386943975403 but having %v", derivate[2][0])
	}
	if derivate[3][0] != 0.1161618363924593 {
		t.Fatalf("Error: exspected 0.1161618363924593 but having %v", derivate[3][0])
	}
	if derivate[4][0] != 0.1161618363924593 {
		t.Fatalf("Error: exspected 0.1161618363924593 but having %v", derivate[4][0])
	}

}

func TestActivationNeuronAndDerivateHiddenLayer(t *testing.T) {
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		t.Fatalf("Error while creating a new neural network: %s", err.Error())
	}

	_, derivate, err := nn.activationNeuronAndDerivate(
		[][]float64{
			[]float64{1, 0, 1, 0, 0},
			[]float64{1, 1, 0, 0, 0},
			[]float64{1, 0, 0, 0, 1},
			[]float64{1, 1, 0, 0, 0},
			[]float64{1, 1, 0, 0, 0},
		},
		[][]float64{
			[]float64{-0.1788923462, 1.1447291138},
			[]float64{-0.9036827427, 1.9007222418},
			[]float64{2.2724162157, -0.1313923743},
			[]float64{0.5534474491, -0.2396527720},
			[]float64{-3.0938322741, 1.9692725150},
		})
	if err != nil {
		t.Fatalf("Error while activate: %s", err.Error())
	}

	if derivate[0][0] != 0.09768765836926072 {
		t.Fatalf("Error: exspected 0.09768765836926072 but having %v", derivate[0][0])
	}
	if derivate[0][1] != 0.19539709423503504 {
		t.Fatalf("Error: exspected 0.19539709423503504 but having %v", derivate[0][1])
	}

	if derivate[1][0] != 0.18900039274490302 {
		t.Fatalf("Error: exspected 0.18900039274490302 but having %v", derivate[1][0])
	}
	if derivate[1][1] != 0.04335180333407164 {
		t.Fatalf("Error: exspected 0.04335180333407164 but having %v", derivate[1][0])
	}

	if derivate[2][0] != 0.03518521503762909 {
		t.Fatalf("Error: exspected 0.23995386943975403 but having %v", derivate[2][0])
	}
	if derivate[2][1] != 0.040724293471417546 {
		t.Fatalf("Error: exspected 0.23995386943975403 but having %v", derivate[2][0])
	}

	if derivate[3][0] != 0.18900039274490302 {
		t.Fatalf("Error: exspected 0.18900039274490302 but having %v", derivate[3][0])
	}
	if derivate[3][1] != 0.04335180333407164 {
		t.Fatalf("Error: exspected 0.1161618363924593 but having %v", derivate[3][0])
	}

	if derivate[4][0] != 0.18900039274490302 {
		t.Fatalf("Error: exspected 0.18900039274490302 but having %v", derivate[4][0])
	}

	if derivate[4][1] != 0.04335180333407164 {
		t.Fatalf("Error: exspected 0.04335180333407164 but having %v", derivate[4][0])
	}
}

func TestComputeNet(t *testing.T) {
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		t.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	nn.Weights = [][][]float64{
		[][]float64{
			[]float64{-0.1788923462, 1.1447291138},
			[]float64{-0.9036827427, 1.9007222418},
			[]float64{2.2724162157, -0.1313923743},
			[]float64{0.5534474491, -0.2396527720},
			[]float64{-3.0938322741, 1.9692725150},
		},
		[][]float64{
			[]float64{0.6960654309},
			[]float64{-2.3317657244},
			[]float64{0.5703955869},
		},
	}
	deriv, _, outputNet, err := nn.computeNet(
		[][]float64{
			[]float64{0, 1, 0, 0},
			[]float64{1, 0, 0, 0},
			[]float64{0, 0, 0, 1},
			[]float64{1, 0, 0, 0},
			[]float64{1, 0, 0, 0},
		},
		[][]float64{
			[]float64{0},
			[]float64{0},
			[]float64{0},
			[]float64{0},
			[]float64{0},
		},
	)
	if err != nil {
		t.Fatalf("Error: %s\n", err.Error())
	}

	if deriv[0][0][0] != 0.09768765836926072 {
		t.Fatalf("Error: exspected 0.09768765836926072 but having %v", deriv[0][0][0])
	}
	if deriv[0][0][1] != 0.19539709423503504 {
		t.Fatalf("Error: exspected 0.19539709423503504 but having %v", deriv[0][0][1])
	}
	if deriv[0][1][0] != 0.18900039274490302 {
		t.Fatalf("Error: exspected 0.18900039274490302 but having %v", deriv[0][1][0])
	}
	if deriv[0][1][1] != 0.04335180333407164 {
		t.Fatalf("Error: exspected 0.04335180333407164 but having %v", deriv[0][1][0])
	}
	if deriv[0][2][0] != 0.03518521503762909 {
		t.Fatalf("Error: exspected 0.23995386943975403 but having %v", deriv[0][2][0])
	}
	if deriv[0][2][1] != 0.040724293471417546 {
		t.Fatalf("Error: exspected 0.23995386943975403 but having %v", deriv[0][2][0])
	}
	if deriv[0][3][0] != 0.18900039274490302 {
		t.Fatalf("Error: exspected 0.18900039274490302 but having %v", deriv[0][3][0])
	}
	if deriv[0][3][1] != 0.04335180333407164 {
		t.Fatalf("Error: exspected 0.1161618363924593 but having %v", deriv[0][3][0])
	}
	if deriv[0][4][0] != 0.18900039274490302 {
		t.Fatalf("Error: exspected 0.18900039274490302 but having %v", deriv[0][4][0])
	}
	if deriv[0][4][1] != 0.04335180333407164 {
		t.Fatalf("Error: exspected 0.04335180333407164 but having %v", deriv[0][4][0])
	}
	if deriv[1][0][0] != 0.20009538257382214 {
		t.Fatalf("Error: exspected 0.20009538257382214 but having %v", deriv[1][0][0])
	}
	if deriv[1][1][0] != 0.22530695136219045 {
		t.Fatalf("Error: exspected 0.22530695136219045 but having %v", deriv[1][1][0])
	}
	if deriv[1][2][0] != 0.18198561022644041 {
		t.Fatalf("Error: exspected 0.18198561022644041 but having %v", deriv[1][2][0])
	}
	if deriv[1][3][0] != 0.22530695136219045 {
		t.Fatalf("Error: exspected 0.22530695136219045 but having %v", deriv[1][3][0])
	}
	if deriv[1][4][0] != 0.22530695136219045 {
		t.Fatalf("Error: exspected 0.22530695136219045 but having %v", deriv[1][4][0])
	}
	// for this and only this value we have 2 possible
	// value. This is inconsistency is caused by the go
	// 1.10.x which cause a different result.
	if outputNet[0][0] != 0.27660658598298415 && outputNet[0][0] != 0.2766065859829842 {
		t.Fatalf("Error: exspected 0.27660658598298415 or 0.2766065859829842 but having %v", outputNet[0][0])
	}
	if outputNet[1][0] != 0.6571402196695981 {
		t.Fatalf("Error: exspected 0.6571402196695981 but having %v", outputNet[1][0])
	}
	if outputNet[2][0] != 0.7607956858798849 {
		t.Fatalf("Error: exspected 0.7607956858798849 but having %v", outputNet[2][0])
	}
	if outputNet[3][0] != 0.6571402196695981 {
		t.Fatalf("Error: exspected 0.6571402196695981 but having %v", outputNet[3][0])
	}
	if outputNet[4][0] != 0.6571402196695981 {
		t.Fatalf("Error: exspected 0.6571402196695981 but having %v", outputNet[4][0])
	}

	nn.LinearOutput = true
	deriv, _, outputNet, err = nn.computeNet(
		[][]float64{
			[]float64{0, 1, 0, 0},
			[]float64{1, 0, 0, 0},
			[]float64{0, 0, 0, 1},
			[]float64{1, 0, 0, 0},
			[]float64{1, 0, 0, 0},
		},
		[][]float64{
			[]float64{0},
			[]float64{0},
			[]float64{0},
			[]float64{0},
			[]float64{0},
		},
	)
	if err != nil {
		t.Fatalf("Error: %s\n", err.Error())
	}

	if deriv[0][0][0] != 0.09768765836926072 {
		t.Fatalf("Error: exspected 0.09768765836926072 but having %v", deriv[0][0][0])
	}
	if deriv[0][0][1] != 0.19539709423503504 {
		t.Fatalf("Error: exspected 0.19539709423503504 but having %v", deriv[0][0][1])
	}
	if deriv[0][1][0] != 0.18900039274490302 {
		t.Fatalf("Error: exspected 0.18900039274490302 but having %v", deriv[0][1][0])
	}
	if deriv[0][1][1] != 0.04335180333407164 {
		t.Fatalf("Error: exspected 0.04335180333407164 but having %v", deriv[0][1][0])
	}
	if deriv[0][2][0] != 0.03518521503762909 {
		t.Fatalf("Error: exspected 0.23995386943975403 but having %v", deriv[0][2][0])
	}
	if deriv[0][2][1] != 0.040724293471417546 {
		t.Fatalf("Error: exspected 0.23995386943975403 but having %v", deriv[0][2][0])
	}
	if deriv[0][3][0] != 0.18900039274490302 {
		t.Fatalf("Error: exspected 0.18900039274490302 but having %v", deriv[0][3][0])
	}
	if deriv[0][3][1] != 0.04335180333407164 {
		t.Fatalf("Error: exspected 0.1161618363924593 but having %v", deriv[0][3][0])
	}
	if deriv[0][4][0] != 0.18900039274490302 {
		t.Fatalf("Error: exspected 0.18900039274490302 but having %v", deriv[0][4][0])
	}
	if deriv[0][4][1] != 0.04335180333407164 {
		t.Fatalf("Error: exspected 0.04335180333407164 but having %v", deriv[0][4][0])
	}
	if deriv[1][0][0] != 1.0 {
		t.Fatalf("Error: exspected 1.0 but having %v", deriv[1][0][0])
	}
	if deriv[1][1][0] != 1.0 {
		t.Fatalf("Error: exspected 1.0 but having %v", deriv[1][1][0])
	}
	if deriv[1][2][0] != 1.0 {
		t.Fatalf("Error: exspected 1.0 but having %v", deriv[1][2][0])
	}
	if deriv[1][3][0] != 1.0 {
		t.Fatalf("Error: exspected 1.0 but having %v", deriv[1][3][0])
	}
	if deriv[1][4][0] != 1.0 {
		t.Fatalf("Error: exspected 1.0 but having %v", deriv[1][4][0])
	}

	if outputNet[0][0] != -0.9613569858276159 {
		t.Fatalf("Error: exspected -0.9613569858276159 but having %v", outputNet[0][0])
	}
	if outputNet[1][0] != 0.6505758599860808 {
		t.Fatalf("Error: exspected 0.6505758599860808 but having %v", outputNet[1][0])
	}
	if outputNet[2][0] != 1.157046783177173 {
		t.Fatalf("Error: exspected 1.157046783177173 but having %v", outputNet[2][0])
	}
	if outputNet[3][0] != 0.6505758599860808 {
		t.Fatalf("Error: exspected 0.6505758599860808 but having %v", outputNet[3][0])
	}
	if outputNet[4][0] != 0.6505758599860808 {
		t.Fatalf("Error: exspected 0.6505758599860808 but having %v", outputNet[4][0])
	}
}

func TestCalculateDelatMatrix(t *testing.T) {
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		t.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	nn.LinearOutput = true
	nn.Weights = [][][]float64{
		[][]float64{
			[]float64{1.28506694649, 1.3474934673},
			[]float64{-0.05348510772, 1.2787610798},
			[]float64{-0.44637920150, 0.3111029354},
			[]float64{0.59364180026, -0.0679702674},
			[]float64{0.85790068032, -0.6079485337},
		},
		[][]float64{
			[]float64{0.3784489620},
			[]float64{0.9666980021},
			[]float64{0.1872564559},
		},
	}
	deriv, _, outputNet, err := nn.computeNet(
		[][]float64{
			[]float64{0, 1, 0, 0},
			[]float64{1, 0, 0, 0},
			[]float64{0, 0, 0, 1},
			[]float64{1, 0, 0, 0},
			[]float64{1, 0, 0, 0},
		},
		[][]float64{
			[]float64{0},
			[]float64{1},
			[]float64{1},
			[]float64{1},
			[]float64{0},
		},
	)
	output := nn.calculateDelatMatrix(
		deriv[len(deriv)-1],
		outputNet,
		[][]float64{
			[]float64{0},
			[]float64{1},
			[]float64{1},
			[]float64{1},
			[]float64{0},
		},
	)
	if output[0][0] != 1.2106913406445328 {
		t.Fatalf("Error: exspected 0.20009538257382214 but having %v", output[0][0])
	}
	if output[1][0] != 0.30138804488138193 {
		t.Fatalf("Error: exspected 0.22530695136219045 but having %v", output[1][0])
	}

	if output[2][0] != 0.37040635413218936 {
		t.Fatalf("Error: exspected 0.18198561022644041 but having %v", output[2][0])
	}
	if output[3][0] != 0.30138804488138193 {
		t.Fatalf("Error: exspected 0.22530695136219045 but having %v", output[3][0])
	}

	if output[4][0] != 1.301388044881382 {
		t.Fatalf("Error: exspected 0.22530695136219045 but having %v", output[4][0])
	}

	nn.LinearOutput = false
	nn.Weights = [][][]float64{
		[][]float64{
			[]float64{-0.2021119892, -0.1625534721},
			[]float64{0.6345485891, -0.1496952018},
			[]float64{1.3177581717, -0.8658362465},
			[]float64{0.7358875502, -1.1527671573},
			[]float64{1.3331761672, 0.2513607668},
		},
		[][]float64{
			[]float64{0.4375865187},
			[]float64{-0.5870363230},
			[]float64{0.3050505510},
		},
	}
	deriv, _, outputNet, err = nn.computeNet(
		[][]float64{
			[]float64{0, 1, 0, 0},
			[]float64{1, 0, 0, 0},
			[]float64{0, 0, 0, 1},
			[]float64{1, 0, 0, 0},
			[]float64{1, 0, 0, 0},
		},
		[][]float64{
			[]float64{0},
			[]float64{1},
			[]float64{1},
			[]float64{1},
			[]float64{0},
		},
	)
	output = nn.calculateDelatMatrix(
		deriv[len(deriv)-1],
		outputNet,
		[][]float64{
			[]float64{0},
			[]float64{1},
			[]float64{1},
			[]float64{1},
			[]float64{0},
		},
	)

	if output[0][0] != 0.12954857503311287 {
		t.Fatalf("Error: exspected 0.20009538257382214 but having %v", output[0][0])
	}
	if output[1][0] != -0.11066323484205977 {
		t.Fatalf("Error: exspected 0.22530695136219045 but having %v", output[1][0])
	}

	if output[2][0] != -0.11477883604778584 {
		t.Fatalf("Error: exspected 0.18198561022644041 but having %v", output[2][0])
	}
	if output[3][0] != -0.11066323484205977 {
		t.Fatalf("Error: exspected 0.22530695136219045 but having %v", output[3][0])
	}

	if output[4][0] != 0.13658823886469879 {
		t.Fatalf("Error: exspected 0.22530695136219045 but having %v", output[4][0])
	}

}

func TestCalculateGradients(t *testing.T) {
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		t.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	nn.LinearOutput = true
	nn.Weights = [][][]float64{
		[][]float64{
			[]float64{-1.07913377493, 0.4154013872},
			[]float64{-1.36424216379, -0.1141982020},
			[]float64{-0.95180654180, -0.5856097147},
			[]float64{-1.98155110105, 0.7127644402},
			[]float64{0.01699809271, 0.1037072587},
		},
		[][]float64{
			[]float64{-0.8158106953},
			[]float64{1.4167744092},
			[]float64{-0.9088682885},
		},
	}
	deriv, allInputCoviariate, outputNet, err := nn.computeNet(
		[][]float64{
			[]float64{0, 1, 0, 0},
			[]float64{1, 0, 0, 0},
			[]float64{0, 0, 0, 1},
			[]float64{1, 0, 0, 0},
			[]float64{1, 0, 0, 0},
		},
		[][]float64{
			[]float64{0},
			[]float64{1},
			[]float64{1},
			[]float64{1},
			[]float64{0},
		},
	)
	deltaMatrix := nn.calculateDelatMatrix(
		deriv[len(deriv)-1],
		outputNet,
		[][]float64{
			[]float64{0},
			[]float64{1},
			[]float64{1},
			[]float64{1},
			[]float64{0},
		},
	)
	gradients, err := nn.calculateGradients(deriv, allInputCoviariate, deltaMatrix)
	if err != nil {
		t.Fatalf("Error: %s\n", err.Error())
	}

	correctGradients := []float64{-1.2930690516450913, -0.5912270225052753, -0.15505457435881592, 0.0000000000,
		-0.5467874547810001, 1.9311169067983491, 1.2606047813500534, 0.24076724226450502,
		0.0000000000, 0.4297448831837909, -8.763780592276198, -1.0967206878450821, -5.017324669256629}

	if len(gradients) != len(correctGradients) {
		t.Fatalf("Error: gradients is length %d instead of %d", len(gradients), len(correctGradients))
	}
	for i := 0; i < len(correctGradients); i++ {
		if gradients[i] != correctGradients[i] {
			t.Fatalf("Error: exspected %f but having %f", correctGradients[i], gradients[i])
		}
	}

}

func TestPlus(t *testing.T) {
	// test step 0
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		t.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	nn.LinearOutput = true
	nn.Weights = [][][]float64{
		[][]float64{
			[]float64{-0.8479596327, 0.3298419564},
			[]float64{-1.6444195826, 1.7066268463},
			[]float64{-0.7796209653, -1.3614137344},
			[]float64{-0.5608931773, -1.1898246798},
			[]float64{1.4300631129, 1.2249354799},
		},
		[][]float64{
			[]float64{-1.156585378},
			[]float64{1.176823318},
			[]float64{1.208972834},
		},
	}

	if nn.LearningRate == nil {
		nn.LearningRate = make([]float64, 13)
		for i := 0; i < len(nn.LearningRate); i++ {
			nn.LearningRate[i] = 0.1
		}
	}

	gradientsOld := nn.plus(
		[]float64{
			-0.37883801419,
			-0.16538575969,
			-0.10426499409,
			0.00000000000,
			-0.10918726042,
			-0.46732624742,
			-0.24586515618,
			-0.15122994591,
			0.00000000000,
			-0.07023114533,
			-3.04093796276,
			-0.51700107254,
			-2.26464623326,
		},
		[]float64{
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		})

	correctGradientsOld := []float64{-1, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1}

	if len(gradientsOld) != len(correctGradientsOld) {
		t.Fatalf("Error: gradients is length %d instead of %d", len(gradientsOld), len(correctGradientsOld))
	}
	for i := 0; i < len(correctGradientsOld); i++ {
		if gradientsOld[i] != correctGradientsOld[i] {
			t.Fatalf("Error: exspected %f but having %f", correctGradientsOld[i], gradientsOld[i])
		}
	}

	/*fmt.Println("gradientsOld", gradientsOld)
	fmt.Println("nn.LearningRate", nn.LearningRate)
	fmt.Println("nn.Weights", nn.Weights)*/

	// test step 1
	gradientsOld = nn.plus(
		[]float64{
			-0.23511787029,
			-0.13517954765,
			-0.08219071770,
			0.00000000000,
			-0.01774760494,
			-0.26974384437,
			-0.14501788288,
			-0.11410558686,
			0.00000000000,
			-0.01062037463,
			-1.74747329615,
			-0.24066645587,
			-1.32793387888,
		},
		correctGradientsOld)

	/*fmt.Println("gradientsOld", gradientsOld)
	fmt.Println("nn.LearningRate", nn.LearningRate)
	fmt.Println("nn.Weights", nn.Weights)*/

	correctGradientsOld = []float64{-1, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1}

	if len(gradientsOld) != len(correctGradientsOld) {
		t.Fatalf("Error: gradients is length %d instead of %d", len(gradientsOld), len(correctGradientsOld))
	}
	for i := 0; i < len(correctGradientsOld); i++ {
		if gradientsOld[i] != correctGradientsOld[i] {
			t.Fatalf("Error: exspected %f but having %f", correctGradientsOld[i], gradientsOld[i])
		}
	}

	// test step 2
	gradientsOld = nn.plus(
		[]float64{
			-0.03251309391,
			-0.07123149862,
			-0.03751161875,
			0.00000000000,
			0.07623002345,
			-0.06202876016,
			-0.05514294458,
			-0.04964233540,
			0.00000000000,
			0.04275651982,
			-0.40529214235,
			0.10985123987,
			-0.29588370949,
		},
		correctGradientsOld)

	/*fmt.Println("gradientsOld", gradientsOld)
	fmt.Println("nn.LearningRate", nn.LearningRate)
	fmt.Println("nn.Weights", nn.Weights)*/

	correctGradientsOld = []float64{-1, -1, -1, 0, 0, -1, -1, -1, 0, 0, -1, 0, -1}

	if len(gradientsOld) != len(correctGradientsOld) {
		t.Fatalf("Error: gradients is length %d instead of %d", len(gradientsOld), len(correctGradientsOld))
	}
	for i := 0; i < len(correctGradientsOld); i++ {
		if gradientsOld[i] != correctGradientsOld[i] {
			t.Fatalf("Error: exspected %f but having %f", correctGradientsOld[i], gradientsOld[i])
		}
	}

	// test step 2
	gradientsOld = nn.plus(
		[]float64{
			0.14083997910,
			0.02271866798,
			0.01836003945,
			0.00000000000,
			0.09976127167,
			0.10601666999,
			0.01471081894,
			0.02668621905,
			0.00000000000,
			0.06461963200,
			0.62455896574,
			0.32675137069,
			0.52031045641,
		},
		correctGradientsOld)

	/*fmt.Println("gradientsOld", gradientsOld)
	fmt.Println("nn.LearningRate", nn.LearningRate)
	fmt.Println("nn.Weights", nn.Weights)*/

	correctGradientsOld = []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0}

	if len(gradientsOld) != len(correctGradientsOld) {
		t.Fatalf("Error: gradients is length %d instead of %d", len(gradientsOld), len(correctGradientsOld))
	}
	for i := 0; i < len(correctGradientsOld); i++ {
		if gradientsOld[i] != correctGradientsOld[i] {
			t.Fatalf("Error: exspected %f but having %f", correctGradientsOld[i], gradientsOld[i])
		}
	}
}

func TestTrain(t *testing.T) {
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		t.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	nn.LinearOutput = true
	nn.Weights = [][][]float64{
		[][]float64{
			[]float64{0.3913129897, 0.9135363600},
			[]float64{0.7483068563, -0.6851971249},
			[]float64{0.6620033544, 0.9480968225},
			[]float64{-0.4293857459, -1.3826671651},
			[]float64{-0.3633092866, 0.6626691715},
		},
		[][]float64{
			[]float64{0.1296913907},
			[]float64{1.0323549699},
			[]float64{-1.7589170540},
		},
	}

	nn.Train(
		[][]float64{
			[]float64{0, 1, 0, 0},
			[]float64{1, 0, 0, 0},
			[]float64{0, 0, 0, 1},
			[]float64{1, 0, 0, 0},
			[]float64{1, 0, 0, 0},
		},
		[][]float64{
			[]float64{0},
			[]float64{1},
			[]float64{1},
			[]float64{1},
			[]float64{0},
		},
	)

	if nn.Weights[0][0][0] != 1.0044329897 {
		t.Fatalf("Error: exspected %f but having 1.0044329897", nn.Weights[0][0][0])
	}
	if nn.Weights[0][1][0] != 0.4558297362999999 {
		t.Fatalf("Error: exspected %f but having 0.4558297362999999", nn.Weights[0][1][0])
	}
	if nn.Weights[0][2][0] != -0.16319664560000005 {
		t.Fatalf("Error: exspected %f but having -0.16319664560000005", nn.Weights[0][2][0])
	}
	if nn.Weights[0][3][0] != -0.4293857459 {
		t.Fatalf("Error: exspected %f but having -0.4293857459", nn.Weights[0][3][0])
	}
	if nn.Weights[0][4][0] != 1.2826907134000003 {
		t.Fatalf("Error: exspected %f but having 1.2826907134000003", nn.Weights[0][4][0])
	}

	if nn.Weights[0][0][1] != 0.5840003600000001 {
		t.Fatalf("Error: exspected %f but having 0.5840003600000001", nn.Weights[0][0][1])
	}
	if nn.Weights[0][1][1] != -0.3927200048999999 {
		t.Fatalf("Error: exspected %f but having -0.3927200048999999", nn.Weights[0][1][1])
	}
	if nn.Weights[0][2][1] != 1.7732968225000008 {
		t.Fatalf("Error: exspected %f but having 1.7732968225000008", nn.Weights[0][2][1])
	}
	if nn.Weights[0][3][1] != -1.3826671651 {
		t.Fatalf("Error: exspected %f but having -1.3826671651", nn.Weights[0][3][1])
	}
	if nn.Weights[0][4][1] != -0.9833308285000001 {
		t.Fatalf("Error: exspected %f but having -0.9833308285000001", nn.Weights[0][4][1])
	}

	if nn.Weights[1][0][0] != 0.4592273906999999 {
		t.Fatalf("Error: exspected %f but having 0.4592273906999999", nn.Weights[1][0][0])
	}
	if nn.Weights[1][1][0] != 1.2308049699 {
		t.Fatalf("Error: exspected %f but having 1.2308049699", nn.Weights[1][1][0])
	}
	if nn.Weights[1][2][0] != -1.4410450539999997 {
		t.Fatalf("Error: exspected %f but having -1.4410450539999997", nn.Weights[1][2][0])
	}
}

// UTILITY FUNCTION FOR TESTING

func createNewStandardNNOneHiddenlayer() (*NeuralNetwork, error) {
	args := NeuralNetworkArguments{
		HiddenLayer:        []int{2},
		InputSize:          4,
		OutputSize:         1,
		Threshold:          0.01,
		StepMax:            999999999999999999,
		LifeSignStep:       1000,
		LinearOutput:       false,
		Minus:              0.5,
		Plus:               1.2,
		ActivationFunction: Logistic,
		DerivateActivation: DerivateLogistic,
		ErrorFunction:      SSE,
		DerivateError:      DerivateSSE,
	}

	return NewNeuralNetworkAndSetup(args)
}

func createNewStandardNNNoHiddenlayer() (*NeuralNetwork, error) {
	args := NeuralNetworkArguments{
		HiddenLayer:        []int{0},
		InputSize:          4,
		OutputSize:         2,
		Threshold:          0.01,
		StepMax:            100,
		LifeSignStep:       100,
		LinearOutput:       false,
		Minus:              10,
		Plus:               100,
		ActivationFunction: Logistic,
		DerivateActivation: DerivateLogistic,
		ErrorFunction:      SSE,
		DerivateError:      DerivateSSE,
	}

	return NewNeuralNetworkAndSetup(args)
}
