package gorpropplus

import (
	"testing"
)

func TestSSE(t *testing.T) {

	nnResult := 0.002113682991
	expected := 0.0
	nnError := SSE(nnResult, expected)
	correctNNerror := 0.000002233827893
	if toFixed(nnError, 6) != toFixed(correctNNerror, 6) {
		t.Fatalf("Error: the nnError is %f instead of %f", nnError, correctNNerror)
	}
}

func TestDerivateSSE(t *testing.T) {

	nnResult := 0.0001307631303
	expected := 0.0
	nnError := DerivateSSE(nnResult, expected)
	correctNNerror := 0.0001307631303
	if nnError != correctNNerror {
		t.Fatalf("Error: the derivate nnError is %f instead of %f", nnError, correctNNerror)
	}
}

func TestCE(t *testing.T) {

	nnResult := 0.00004911231579
	expected := 0.0
	nnError := CE(nnResult, expected)
	correctNNerror := 0.00004911352184
	if toFixed(nnError, 6) != toFixed(correctNNerror, 6) {
		t.Fatalf("Error: the nnError is %f instead of %f", nnError, correctNNerror)
	}
}

// This test is a WIP.
// func TestDerivateCE(t *testing.T) {

// 	nnResult := 0.00000467802729
// 	expected := 0.0
// 	nnError := DerivateCE(nnResult, expected)
// 	correctNNerror := 0.00000467802729
// 	if nnError != correctNNerror {
// 		t.Fatalf("Error: the derivate nnError is %f instead of %f", nnError, correctNNerror)
// 	}
// }
