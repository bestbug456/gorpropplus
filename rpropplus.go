// gorpropplus is the Golang implementation
// of the machine learning alghoritm rprop+
// Created  by  Danilo  'bestbug'  with the
// support of Franca 'forky' Marinelli

package gorpropplus

import (
	"math"
)

// NeuralNetwork is the actual neural network object
type NeuralNetwork struct {
	// Neural Network properties
	Weights                [][][]float64
	TotalWeights           int
	NrCol                  []int
	NrRow                  []int
	ActivationFunction     func(float64) float64          `bson:"-"`
	DerivateActivation     func(float64) float64          `bson:"-"`
	ErrorFunction          func(float64, float64) float64 `bson:"-"`
	DerivateError          func(float64, float64) float64 `bson:"-"`
	ActivationFunctionName string                         `bson:"-"`
	ErrorFunctionName      string                         `bson:"-"`
	LearningRate           []float64
	// Neural Network configuration
	Threshold    float64
	StepMax      int64
	LifeSignStep int64
	LinearOutput bool
	Minus        float64
	Plus         float64
}

// NeuralNetworkArguments permit to compact all
// the various arguments need by this library.
type NeuralNetworkArguments struct {
	LearningRate       []float64
	HiddenLayer        []int
	InputSize          int
	OutputSize         int
	Threshold          float64
	StepMax            int64
	LifeSignStep       int64
	LinearOutput       bool
	Minus              float64
	Plus               float64
	ActivationFunction func(float64) float64
	DerivateActivation func(float64) float64
	ErrorFunction      func(float64, float64) float64
	DerivateError      func(float64, float64) float64
}

// ValidationResult contain the following infos
// * ConfusionMatrix: A confusion matrix struct as follow:
//
//                   Exspected value
//
//            +---------+----+----+----+
//            |XXXX|    |    |    |    |
//            |XXXX| V1 | V2 | V3 | V4 |
//            +------------------------+
//            |    |    |    |    |    |
//            | V1 |  2 |  0 |  1 |  3 |
//            +------------------------+
//Predicted   |    |    |    |    |    |
//  value     | V2 |  7 |  3 |  9 |  0 |
//            +------------------------+
//            |    |    |    |    |    |
//            | V3 |  1 |  1 |  4 |  1 |
//            +------------------------+
//            |    |    |    |    |    |
//            | V4 |  3 |  2 |  1 |  5 |
//            +----+----+----+----+----+
//
// * CorrectPrediction: contain the number of prediction
//						having distance from the exspected
//						value < threshold.
// * PredictionResult: Contain all the prediction done.
type ValidationResult struct {
	ConfusionMatrix   [][]int
	CorrectPrediction int
	PredictionResult  [][]float64
}

// NewNeuralNetworkAndSetup create a fresh  new
// neural network struct and it  inizialise  it
// internal weights.
func NewNeuralNetworkAndSetup(args NeuralNetworkArguments) (*NeuralNetwork, error) {
	var weights [][][]float64
	var totalWeights int
	var nrRow []int
	var nrCol []int
	previusLayer := args.InputSize
	if len(args.HiddenLayer) == 1 && args.HiddenLayer[0] == 0 {
		sliceWeights := randomNormalSlice((args.InputSize+1)*args.OutputSize, previusLayer)
		totalWeights = (args.InputSize + 1) * args.OutputSize
		weights = make([][][]float64, 1)
		var err error
		weights[0], err = createMatrix(args.InputSize+1, args.OutputSize, sliceWeights)
		if err != nil {
			return nil, err
		}
	} else {
		nrRow = make([]int, len(args.HiddenLayer)+1)
		nrRow[0] = args.InputSize + 1
		nrCol = make([]int, len(args.HiddenLayer)+1)
		nrCol[0] = args.HiddenLayer[0]
		for i := 1; i < len(args.HiddenLayer); i++ {
			nrRow[i] = args.HiddenLayer[i-1] + 1
			nrCol[i] = args.HiddenLayer[i]
		}
		nrRow[len(nrRow)-1] = args.HiddenLayer[len(args.HiddenLayer)-1] + 1
		nrCol[len(nrCol)-1] = args.OutputSize
		weights = make([][][]float64, len(nrCol))
		var err error
		for i := 0; i < len(weights); i++ {
			weights[i], err = createMatrix(nrRow[i], nrCol[i], randomNormalSlice(nrCol[i]*nrRow[i], previusLayer))
			totalWeights = nrCol[i] * nrRow[i]
			if err != nil {
				return nil, err
			}
			previusLayer = len(weights[i]) - 1
		}
	}

	return &NeuralNetwork{
		ActivationFunction: args.ActivationFunction,
		ErrorFunction:      args.ErrorFunction,
		DerivateActivation: args.DerivateActivation,
		DerivateError:      args.DerivateError,
		Weights:            weights,
		TotalWeights:       totalWeights,
		Threshold:          args.Threshold,
		StepMax:            args.StepMax,
		LifeSignStep:       args.LifeSignStep,
		LinearOutput:       args.LinearOutput,
		Minus:              args.Minus,
		Plus:               args.Plus,
		NrCol:              nrCol,
		NrRow:              nrRow,
		LearningRate:       args.LearningRate,
	}, nil
}

// Train use A training dataset is a dataset of
// examples  used  for learning, that is to fit
// the parameters.  If  something went wrong an
// error is return.
func (n *NeuralNetwork) Train(input [][]float64, output [][]float64) error {
	learningRate := make([]float64, n.TotalWeights)
	for i := 0; i < len(learningRate); i++ {
		learningRate[i] = 0.1
	}
	deriv, allInputCoviariate, outputNet, err := n.computeNet(input, output)
	if err != nil {
		return err
	}

	deltaMatrix := n.calculateDelatMatrix(deriv[len(deriv)-1], outputNet, output)
	gradients, err := n.calculateGradients(deriv, allInputCoviariate, deltaMatrix)
	if err != nil {
		return err
	}
	minReachedThreshold := findMaxOfArray(gradients)
	gradientsOld := make([]float64, len(gradients))

	if n.LearningRate == nil {
		n.LearningRate = make([]float64, len(gradients))
		for i := 0; i < len(n.LearningRate); i++ {
			n.LearningRate[i] = 0.1
		}
	}
	var i int64
	for i = 0; i < n.StepMax; i++ {
		if minReachedThreshold <= n.Threshold {
			break
		}
		gradientsOld = n.plus(gradients, gradientsOld)

		deriv, allInputCoviariate, outputNet, err = n.computeNet(input, output)
		if err != nil {
			return err
		}
		deltaMatrix = n.calculateDelatMatrix(deriv[len(deriv)-1], outputNet, output)
		gradients, err = n.calculateGradients(deriv, allInputCoviariate, deltaMatrix)
		if err != nil {
			return err
		}
		reachedThreshold := findMaxOfArray(gradients)
		if reachedThreshold < minReachedThreshold {
			minReachedThreshold = reachedThreshold
		}
	}

	return nil
}

func (n *NeuralNetwork) plus(gradients []float64, gradientsOld []float64) []float64 {

	weights := convertArrayOfMatrixToSlice(n.Weights, false)
	supp := make([]float64, len(gradientsOld))
	for i := 0; i < len(gradientsOld); i++ {
		if gradients[i] > 0 {
			supp[i] = gradientsOld[i]
		}
		if gradients[i] < 0 {
			supp[i] = gradientsOld[i] * -1.0
		}
		if gradients[i] == 0 {
			supp[i] = 0.0
		}
	}

	positive, negative, nonNegative := findPosPositiveNegativeAndNonNegative(supp)

	for i := 0; i < len(positive); i++ {
		supp := n.LearningRate[positive[i]] * n.Plus
		n.LearningRate[positive[i]] = findMin(supp, 0.1)
	}

	if len(negative) != 0 {
		for i := 0; i < len(negative); i++ {
			weights[negative[i]] = weights[negative[i]] + (gradientsOld[negative[i]] * n.LearningRate[negative[i]])
			supp := n.LearningRate[negative[i]] * n.Minus
			n.LearningRate[negative[i]] = findMax(supp, 0.0000000001)
			gradientsOld[negative[i]] = 0.0
		}
		if len(nonNegative) != 0 {
			for i := 0; i < len(nonNegative); i++ {
				var supp float64
				if gradients[nonNegative[i]] > 0 {
					supp = 1.0
				}
				if gradients[nonNegative[i]] < 0 {
					supp = -1.0
				}
				if gradients[nonNegative[i]] == 0 {
					supp = 0.0
				}
				weights[nonNegative[i]] = weights[nonNegative[i]] - (supp * n.LearningRate[nonNegative[i]])
				gradientsOld[nonNegative[i]] = supp
			}
		}
	} else {
		for i := 0; i < len(weights); i++ {
			var supp float64
			if gradients[i] > 0 {
				supp = 1.0
			}
			if gradients[i] < 0 {
				supp = -1.0
			}
			if gradients[i] == 0 {
				supp = 0.0
			}
			weights[i] = weights[i] - (supp * n.LearningRate[i])
			gradientsOld[i] = supp
		}
	}
	n.Weights = updateWeights(weights, n.Weights)
	return gradientsOld
}

func (n *NeuralNetwork) calculateDelatMatrix(derivlastlayer [][]float64, outputNet [][]float64, output [][]float64) [][]float64 {
	deltaMatrix := make([][]float64, len(output))
	for j := 0; j < len(outputNet); j++ {
		deltaMatrix[j] = make([]float64, len(output[j]))
		for k := 0; k < len(outputNet[j]); k++ {
			deltaMatrix[j][k] += n.DerivateError(outputNet[j][k], output[j][k])
		}
	}

	if !n.LinearOutput {
		for j := 0; j < len(derivlastlayer); j++ {
			for k := 0; k < len(derivlastlayer[j]); k++ {
				deltaMatrix[j][k] = derivlastlayer[j][k] * deltaMatrix[j][k]
			}
		}
	}
	return deltaMatrix

}

func (n *NeuralNetwork) calculateGradients(deriv [][][]float64, allInputCoviariate [][][]float64, deltaMatrix [][]float64) ([]float64, error) {

	gradientsMatrix := make([][][]float64, len(n.Weights))
	gradientsMatrix[0] = matrixCrossProduct(transposeMatrix(allInputCoviariate[len(deriv)-1]), deltaMatrix)

	if len(n.Weights) > 1 {
		var err error
		for i := len(n.Weights) - 1; i >= 1; i-- {
			deltaMatrix, err = dot(deriv[i-1], matrixCrossProduct(deltaMatrix, transposeMatrix(n.Weights[i][1:])))
			if err != nil {
				return nil, err
			}
			gradientsMatrix[i] = matrixCrossProduct(transposeMatrix(allInputCoviariate[i-1]), deltaMatrix)
		}
	}
	return convertArrayOfMatrixToSlice(gradientsMatrix, true), nil
}

func (n *NeuralNetwork) computeNet(input [][]float64, output [][]float64) ([][][]float64, [][][]float64, [][]float64, error) {

	allinputCovariate := make([][][]float64, len(n.Weights))
	inputCovariate := make([][]float64, len(input))
	outputNet := make([][]float64, len(output))

	for i := 0; i < len(inputCovariate); i++ {
		covariate := make([]float64, len(input[i])+1)
		covariate[0] = 1
		for j := 0; j < len(input[i]); j++ {
			covariate[j+1] = input[i][j]
		}
		inputCovariate[i] = covariate
	}

	deriv := make([][][]float64, len(n.Weights))
	allinputCovariate[0] = inputCovariate
	var err error
	for i := 0; i < len(n.Weights); i++ {
		allinputCovariate[i] = inputCovariate
		inputCovariate, deriv[i], err = n.activationNeuronAndDerivate(allinputCovariate[i], n.Weights[i])
		if err != nil {
			return nil, nil, nil, err
		}
	}

	if n.LinearOutput {
		// the output of neural network is
		// equal to the previous level input
		// multiply to weights
		outputNet = matrixCrossProduct(allinputCovariate[len(allinputCovariate)-1], n.Weights[len(n.Weights)-1])
		for j := 0; j < len(deriv[len(deriv)-1]); j++ {
			//lastlayer := make([]float64, len(output[0]))
			for i := 0; i < len(deriv[len(deriv)-1][j]); i++ {
				deriv[len(deriv)-1][j][i] = 1.0
			}
			//deriv[len(deriv)-1][j] = lastlayer
		}
	} else {
		// the last inputCovariate is the final
		// result of the neural network we save
		// it without bias
		for j := 0; j < len(inputCovariate); j++ {
			outputNet[j] = inputCovariate[j][1:]
		}
	}

	return deriv, allinputCovariate, outputNet, nil
}

func (n *NeuralNetwork) activationNeuronAndDerivate(x, y [][]float64) ([][]float64, [][]float64, error) {

	activate := make([][]float64, len(x))
	derivate := make([][]float64, len(x))
	var b int
	var v int
	for i := 0; i < len(x); i++ {

		if len(activate[b]) == 0 {
			activate[b] = make([]float64, len(y[0])+1)
			v = 0
			activate[b][v] = 1
			v++
		}
		if len(derivate[b]) == 0 {
			derivate[b] = make([]float64, len(y[0]))
		}
		for m := 0; m < len(y[0]); m++ {
			var k int
			var supp float64
			for f := 0; f < len(x[i]); f++ {
				// if f == 0 {
				// 	supp += 1 * y[k][m]
				// }
				supp += x[i][f] * y[k][m]
				k++
			}
			activate[b][v] = n.ActivationFunction(supp)
			derivate[b][v-1] = n.DerivateActivation(activate[b][v])
			v++
			if v == len(activate[b]) {
				v = 0
				b++
			}
		}
	}

	return activate, derivate, nil
}

// Validate use a validation dataset and create and return a ValidationResult
func (n *NeuralNetwork) Validate(input [][]float64, output [][]float64) (*ValidationResult, error) {

	totalclass := len(output[0])
	if totalclass == 1 {
		totalclass++
	}
	var result ValidationResult
	result.ConfusionMatrix = make([][]int, totalclass)
	result.PredictionResult = make([][]float64, len(input))
	for i := 0; i < len(input); i++ {

		prediction, err := n.Predict(input[i])
		if err != nil {
			return nil, err
		}

		var max float64
		var maxpos int
		var maxcorrect float64
		var maxcorrectpos int
		var correct int
		for j := 0; j < len(prediction); j++ {
			if math.Abs(prediction[j]-output[i][j]) < n.Threshold {
				correct++
			}
			if prediction[j] > max {
				max = prediction[j]
				// Manage the special case of having only 1 output
				if len(output[0]) == 1 && prediction[j] < 0.5 {
					maxpos = 1
				} else {
					maxpos = j
				}
			}
			if output[i][j] > maxcorrect {
				maxcorrect = prediction[j]
				// Manage the special case of having only 1 output
				if len(output[0]) == 1 && prediction[j] < 0.5 {
					maxcorrectpos = 1
				} else {
					maxcorrectpos = j
				}

			}
		}

		if len(result.ConfusionMatrix[maxpos]) == 0 {
			result.ConfusionMatrix[maxpos] = make([]int, totalclass)
		}
		result.ConfusionMatrix[maxpos][maxcorrectpos]++
		result.PredictionResult[i] = prediction
		if correct == len(output[i]) {
			result.CorrectPrediction++
		}
	}

	return &result, nil
}

// Predict permit to have a prediction  of  the
// inputs. it returns the prediction result  or
// an error if something went wrong.
func (n *NeuralNetwork) Predict(input []float64) ([]float64, error) {

	allinputCovariate := make([][][]float64, len(n.Weights))
	inputCovariate := make([][]float64, 1)
	var outputNet []float64

	covariate := make([]float64, len(input)+1)
	covariate[0] = 1
	for j := 0; j < len(input); j++ {
		covariate[j+1] = input[j]
	}
	inputCovariate[0] = covariate

	var err error

	for i := 0; i < len(n.Weights); i++ {
		allinputCovariate[i] = inputCovariate
		inputCovariate, _, err = n.activationNeuronAndDerivate(allinputCovariate[i], n.Weights[i])
		if err != nil {
			return nil, err
		}
	}

	if n.LinearOutput {
		// the output of neural network is
		// equal to the previous level input
		// multiply to weights
		res := matrixCrossProduct(allinputCovariate[len(allinputCovariate)-1], n.Weights[len(n.Weights)-1])
		outputNet = res[0]
	} else {
		// the last inputCovariate is the final
		// result of the neural network we save
		// it without bias
		outputNet = inputCovariate[0][1:]
	}
	return outputNet, nil
}
