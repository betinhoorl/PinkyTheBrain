package word2vec

// Model representa o modelo
type Model struct {
	dimension int
	words     map[string]Vector
}

// Vector representa um vetor de palavras.
type Vector []float32
