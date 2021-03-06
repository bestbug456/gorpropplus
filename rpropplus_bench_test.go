package gorpropplus

import (
	"testing"
)

// Dataset of 5 data
// BenchmarkTrain-4   	  200000	      5695 ns/op	    4304 B/op	     124 allocs/op
// Dataset of 150 data
// BenchmarkTrain-4   	   20000	     88618 ns/op	   98440 B/op	    1614 allocs/op
// Dataset of 300 data
// BenchmarkTrain-4   	   10000	    178695 ns/op	  195672 B/op	    3122 allocs/op

func BenchmarkTrain(b *testing.B) {
	b.StopTimer()
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		b.Fatalf("Error while creating the new neural network: %s\n", err.Error())
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		nn.Train(
			[][]float64{
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
			},
			[][]float64{
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
				{0},
				{1},
				{1},
				{1},
				{0},
			},
		)
	}
}

func BenchmarkActivationNeuronAndDerivateHiddenLayer(b *testing.B) {
	b.StopTimer()
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		b.Fatalf("Error while creating the new neural network: %s\n", err.Error())
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := nn.activationNeuronAndDerivate(
			[][]float64{
				{1, 0, 1, 0, 0},
				{1, 1, 0, 0, 0},
				{1, 0, 0, 0, 1},
				{1, 1, 0, 0, 0},
				{1, 1, 0, 0, 0},
			},
			[][]float64{
				{-0.1788923462, 1.1447291138},
				{-0.9036827427, 1.9007222418},
				{2.2724162157, -0.1313923743},
				{0.5534474491, -0.2396527720},
				{-3.0938322741, 1.9692725150},
			})
		if err != nil {
			b.Fatalf("Error while activate: %s", err.Error())
		}
	}
}

func BenchmarkComputeNet(b *testing.B) {
	b.StopTimer()
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		b.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, err := nn.computeNet(
			[][]float64{
				{0, 1, 0, 0},
				{1, 0, 0, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{1, 0, 0, 0},
			},
			[][]float64{
				{0},
				{0},
				{0},
				{0},
				{0},
			},
		)
		if err != nil {
			b.Fatalf("Error: %s\n", err.Error())
		}
	}
}

func BenchmarkCalculateGradients(b *testing.B) {
	b.StopTimer()
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		b.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	nn.LinearOutput = true
	deriv, allInputCoviariate, outputNet, err := nn.computeNet(
		[][]float64{
			{0, 1, 0, 0},
			{1, 0, 0, 0},
			{0, 0, 0, 1},
			{1, 0, 0, 0},
			{1, 0, 0, 0},
		},
		[][]float64{
			{0},
			{1},
			{1},
			{1},
			{0},
		},
	)
	deltaMatrix := nn.calculateDelatMatrix(
		deriv[len(deriv)-1],
		outputNet,
		[][]float64{
			{0},
			{1},
			{1},
			{1},
			{0},
		},
	)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, err := nn.calculateGradients(deriv, allInputCoviariate, deltaMatrix)
		if err != nil {
			b.Fatalf("Error: %s\n", err.Error())
		}
	}
}

func BenchmarkPredictWithLinearOutputTRUE(b *testing.B) {
	b.StopTimer()
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		b.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	// predict with linear output TRUE
	nn.LinearOutput = true
	nn.Weights = [][][]float64{
		{
			{1.4977972345, 0.16028919024},
			{-2.0576028303, 0.05778128603},
			{-1.9497804732, -0.01387975675},
			{-0.7842283045, -1.20152607718},
			{0.2203101549, -0.80684920968},
		},
		{
			{-2.0760457565},
			{-0.3657459873},
			{0.1165046063},
		},
	}
	nn.Train(
		[][]float64{
			{0, 1, 0, 0},
			{1, 0, 0, 0},
			{0, 0, 0, 1},
			{1, 0, 0, 0},
			{1, 0, 0, 0},
		},
		[][]float64{
			{0},
			{1},
			{1},
			{1},
			{0},
		},
	)
	testInput := []float64{0, 1, 0, 0}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, err := nn.Predict(testInput)
		if err != nil {
			b.Fatalf("Error: %s\n", err.Error())
		}
	}
}

func BenchmarkPredictWithLinearOutputFALSE(b *testing.B) {
	b.StopTimer()
	nn, err := createNewStandardNNOneHiddenlayer()
	if err != nil {
		b.Fatalf("Error while creating a new neural network: %s", err.Error())
	}
	// predict with linear output TRUE
	nn.Weights = [][][]float64{
		{
			{-0.3094936307, 1.1894633375},
			{1.7938942111, -0.7728903275},
			{-2.9713719713, 1.2031703117},
			{-0.1989300481, 1.5803919589},
			{3.2561112564, -4.4571365361},
		},
		{
			{-0.6192761778},
			{3.1272406514},
			{-1.9848503107},
		},
	}
	testInput := []float64{0, 1, 0, 0}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, err := nn.Predict(testInput)
		if err != nil {
			b.Fatalf("Error: %s\n", err.Error())
		}
	}
}

/*
// start
BenchmarkTrain-4                                    	   10000	    176143 ns/op	  195672 B/op	    3122 allocs/op
BenchmarkActivationNeuronAndDerivateHiddenLayer-4   	 1000000	      1003 ns/op	     416 B/op	      12 allocs/op
BenchmarkComputeNet-4                               	  500000	      2358 ns/op	    1584 B/op	      43 allocs/op
BenchmarkCalculateGradients-4                       	  500000	      2935 ns/op	    2520 B/op	      74 allocs/op
// improve transpose
BenchmarkTrain-4                                    	   10000	    165627 ns/op	  151696 B/op	    3049 allocs/op
BenchmarkActivationNeuronAndDerivateHiddenLayer-4   	 2000000	       996 ns/op	     416 B/op	      12 allocs/op
BenchmarkComputeNet-4                               	  500000	      2311 ns/op	    1584 B/op	      43 allocs/op
BenchmarkCalculateGradients-4                       	 1000000	      1990 ns/op	    1936 B/op	      49 allocs/op
// Move nobias inside if len(n.Weights) > 1 { statement
BenchmarkTrain-4                                    	   10000	    165318 ns/op	  151488 B/op	    3043 allocs/op
BenchmarkActivationNeuronAndDerivateHiddenLayer-4   	 2000000	       993 ns/op	     416 B/op	      12 allocs/op
BenchmarkComputeNet-4                               	  500000	      2315 ns/op	    1584 B/op	      43 allocs/op
BenchmarkCalculateGradients-4                       	 1000000	      1737 ns/op	    1728 B/op	      43 allocs/op
// Re-organise part of the code
BenchmarkTrain-4                                    	   10000	    163585 ns/op	  151488 B/op	    3043 allocs/op
BenchmarkActivationNeuronAndDerivateHiddenLayer-4   	 2000000	       995 ns/op	     416 B/op	      12 allocs/op
BenchmarkComputeNet-4                               	 1000000	      2305 ns/op	    1584 B/op	      43 allocs/op
BenchmarkCalculateGradients-4                       	 1000000	      1738 ns/op	    1728 B/op	      43 allocs/op
// Remove input covariate manipulation
BenchmarkTrain-4                                    	   10000	    149621 ns/op	  144288 B/op	    2443 allocs/op
BenchmarkActivationNeuronAndDerivateHiddenLayer-4   	 1000000	      1008 ns/op	     496 B/op	      12 allocs/op
BenchmarkComputeNet-4                               	 1000000	      2109 ns/op	    1464 B/op	      33 allocs/op
BenchmarkCalculateGradients-4                       	 1000000	      1705 ns/op	    1728 B/op	      43 allocs/op
// Remove nobias manipulation in calculate grandients
BenchmarkTrain-4                                    	   10000	    149915 ns/op	  144224 B/op	    2440 allocs/op
BenchmarkActivationNeuronAndDerivateHiddenLayer-4   	 1000000	      1010 ns/op	     496 B/op	      12 allocs/op
BenchmarkComputeNet-4                               	 1000000	      2102 ns/op	    1464 B/op	      33 allocs/op
BenchmarkCalculateGradients-4                       	 1000000	      1613 ns/op	    1664 B/op	      40 allocs/op
*/