package mathematics

import (
	"math/rand"
)

// Random ...
func Random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

// Matrix ...
func Matrix(row, col int) [][]float64 {
	matrix := make([][]float64, row)
	for i := 0; i < row; i++ {
		matrix[i] = make([]float64, col)
	}
	return matrix
}

// MatrixFill ...
func MatrixFill(index []float64) [][]float64 {
	cols := len(index)
	out := Matrix(1, cols)

	for i := 0; i < 1; i++ {
		for j := 0; j < cols; j++ {
			out[i][j] = index[j]
		}
	}

	return out
}

// Vector ...
func Vector(row int, fill float64) []float64 {
	vector := make([]float64, row)
	for i := 0; i < row; i++ {
		vector[i] = fill
	}
	return vector
}
