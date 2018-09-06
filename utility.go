package gorpropplus

import (
	"fmt"
	"math"
	"math/rand"
)

// randomNormalSlice create a normal slice with
// He-et-al Initialization optimisation
func randomNormalSlice(size int, previusSize int) []float64 {
	neurons := make([]float64, size)
	for i := 0; i < len(neurons); i++ {
		neurons[i] = rand.NormFloat64() * math.Sqrt(2.0/float64(previusSize))
	}
	return neurons
}

func createMatrix(row, col int, value []float64) ([][]float64, error) {
	matrix := make([][]float64, row)
	for i := 0; i < len(matrix); i++ {
		matrix[i] = make([]float64, col)
		for j := 0; j < len(matrix[i]); j++ {
			if i+j > len(value) {
				return nil, fmt.Errorf("Input does not have enought data for fill the matrix.")
			}
			matrix[i][j] = value[i+j]
		}
	}
	return matrix, nil
}

func matrixCrossProduct(x, y [][]float64) [][]float64 {
	ris := make([][]float64, len(x))

	var b int
	var v int
	for i := 0; i < len(x); i++ {

		if len(ris[b]) == 0 {
			ris[b] = make([]float64, len(y[0]))
		}
		for m := 0; m < len(y[0]); m++ {
			var k int
			var supp float64
			for f := 0; f < len(x[i]); f++ {
				supp += x[i][f] * y[k][m]
				k++

			}
			ris[b][v] = supp
			v++
			if v == len(ris[b]) {
				v = 0
				b++
			}
		}
	}
	return ris
}

func findMaxOfArray(in []float64) float64 {
	var max float64
	for i := 0; i < len(in); i++ {
		if math.Abs(in[i]) > max {
			max = math.Abs(in[i])
		}
	}
	return max
}

func transposeMatrix(x [][]float64) [][]float64 {
	out := make([][]float64, len(x[0]))
	var k int
	for i := 0; i < len(x); i += 1 {

		for j := 0; j < len(x[0]); j += 1 {
			if len(out[j]) == 0 {
				out[j] = make([]float64, len(x))
			}
			out[j][k] = x[i][j]
		}
		k++
	}
	return out
}

func findPosPositiveNegativeAndNonNegative(input []float64) ([]int, []int, []int) {
	var positive []int
	var negative []int
	var nonNegative []int
	for i := 0; i < len(input); i++ {
		if input[i] > 0 {
			positive = append(positive, i)
		}
		if input[i] < 0 {
			negative = append(negative, i)
		}
		if input[i] == 0 || input[i] > 0 {
			nonNegative = append(nonNegative, i)
		}
	}
	return positive, negative, nonNegative
}

func findMin(a float64, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func findMax(a float64, b float64) float64 {
	if a < b {
		return b
	}
	return a
}

func convertArrayOfMatrixToSlice(input [][][]float64, endToStart bool) []float64 {
	var result []float64
	if endToStart {
		for i := len(input) - 1; i >= 0; i-- {
			for k := 0; k < len(input[i][0]); k++ {
				for j := 0; j < len(input[i]); j++ {
					result = append(result, input[i][j][k])
				}
			}
		}
		return result
	}

	for i := 0; i < len(input); i++ {
		for k := 0; k < len(input[i][0]); k++ {
			for j := 0; j < len(input[i]); j++ {
				result = append(result, input[i][j][k])
			}
		}
	}
	return result

}

func updateWeights(slice []float64, matrix [][][]float64) [][][]float64 {
	var m int
	for i := 0; i < len(matrix); i++ {
		for k := 0; k < len(matrix[i][0]); k++ {
			for j := 0; j < len(matrix[i]); j++ {
				matrix[i][j][k] = slice[m]
				m++
			}
		}
	}
	return matrix
}

func dot(x, y [][]float64) ([][]float64, error) {
	if len(x[0]) != len(y[0]) {
		return nil, fmt.Errorf("Can't do matrix multiplication.")
	}

	out := make([][]float64, len(x))
	for i := 0; i < len(x); i += 1 {
		out[i] = make([]float64, len(x[i]))
		for j := 0; j < len(x[i]); j += 1 {
			out[i][j] = x[i][j] * y[i][j]
		}
	}
	return out, nil
}
