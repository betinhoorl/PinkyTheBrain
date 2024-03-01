package neuralnetwork

import (
	"PinkyTheBrain/mathematics"
	"fmt"
	"log"
	"math"
)

// INeuralNetwork representa a classe neural.
type INeuralNetwork interface {
	FeedForward(inputs, hiddens, outputs int)
	Train(patterns [][][]float64, iterations int, learningRate, momentFactor float64, debug bool) []float64
	Test(patterns [][][]float64)
	SetContexts(nContexts int, initValues [][]float64)
}

// NeuralNetwork representa a estrutura da rede
type NeuralNetwork struct {
	// Número dos nós das camadas entrada, oculto, de saída e do contextos
	InputsNeuron   int
	HiddensNeuron  int
	OutputsNeuron  int
	ContextsNeuron int
	// Verificador de regressão
	Regression bool
	// Ativações para nós
	InputActivations, HiddenActivations, OutputActivations []float64
	// Elman RNN contexto
	Contexts [][]float64
	// Pesos
	InputWeights   [][]float64
	OutputWeights  [][]float64
	ContextWeights [][][]float64
	// Última alteração nos pesos por impulso
	InputChanges   [][]float64
	OutputChanges  [][]float64
	ContextChanges [][][]float64
}

// NewNeuralNetwork Cria uma nova instância da rede
func NewNeuralNetwork() INeuralNetwork {
	return &NeuralNetwork{}
}

// FeedForward ...
func (neural *NeuralNetwork) FeedForward(inputs, hiddens, outputs int) {
	var bias = 1

	neural.InputsNeuron = inputs + bias
	neural.HiddensNeuron = hiddens + bias
	neural.OutputsNeuron = outputs

	neural.InputActivations = mathematics.Vector(neural.InputsNeuron, 1.0)
	neural.HiddenActivations = mathematics.Vector(neural.HiddensNeuron, 1.0)
	neural.OutputActivations = mathematics.Vector(neural.OutputsNeuron, 1.0)

	neural.loadInputWeights()
	neural.loadOutputWeights()

	neural.InputChanges = mathematics.Matrix(neural.InputsNeuron, neural.HiddensNeuron)
	neural.OutputChanges = mathematics.Matrix(neural.HiddensNeuron, neural.OutputsNeuron)
}

func (neural *NeuralNetwork) loadInputWeights() {
	neural.InputWeights = mathematics.Matrix(neural.InputsNeuron, neural.HiddensNeuron)
	for i := 0; i < neural.InputsNeuron; i++ {
		for j := 0; j < neural.HiddensNeuron; j++ {
			neural.InputWeights[i][j] = mathematics.Random(-1, 1)
		}
	}
}

func (neural *NeuralNetwork) loadOutputWeights() {
	neural.OutputWeights = mathematics.Matrix(neural.HiddensNeuron, neural.OutputsNeuron)
	for i := 0; i < neural.HiddensNeuron; i++ {
		for j := 0; j < neural.OutputsNeuron; j++ {
			neural.OutputWeights[i][j] = mathematics.Random(-1, 1)
		}
	}
}

// BackPropagate ...
func (neural *NeuralNetwork) BackPropagate(targets []float64, learningRate, momentFactor float64) float64 {
	sigmoid := mathematics.NewSigmoid()
	if len(targets) != neural.OutputsNeuron {
		log.Fatal("Erro: número incorreto de valores")
	}

	outputDeltas := mathematics.Vector(neural.OutputsNeuron, 0.0)
	for i := 0; i < neural.OutputsNeuron; i++ {
		outputDeltas[i] = sigmoid.DerivativeSigmoid(neural.OutputActivations[i]) * (targets[i] - neural.OutputActivations[i])
	}

	hiddenDeltas := mathematics.Vector(neural.HiddensNeuron, 0.0)
	for i := 0; i < neural.HiddensNeuron; i++ {
		var err float64

		for j := 0; j < neural.OutputsNeuron; j++ {
			err += outputDeltas[j] * neural.OutputWeights[i][j]
		}

		hiddenDeltas[i] = sigmoid.DerivativeSigmoid(neural.HiddenActivations[i]) * err
	}

	for i := 0; i < neural.HiddensNeuron; i++ {
		for j := 0; j < neural.OutputsNeuron; j++ {
			change := outputDeltas[j] * neural.HiddenActivations[i]
			neural.OutputWeights[i][j] = neural.OutputWeights[i][j] + learningRate*change + momentFactor*neural.OutputChanges[i][j]
			neural.OutputChanges[i][j] = change
		}
	}

	for i := 0; i < neural.ContextsNeuron; i++ {
		for j := 0; j < neural.HiddensNeuron; j++ {
			for k := 0; k < neural.HiddensNeuron; k++ {
				change := hiddenDeltas[k] * neural.Contexts[i][j]
				neural.ContextWeights[i][j][k] = neural.ContextWeights[i][j][k] + learningRate*change + momentFactor*neural.ContextChanges[i][j][k]
				neural.ContextChanges[i][j][k] = change
			}
		}
	}

	for i := 0; i < neural.InputsNeuron; i++ {
		for j := 0; j < neural.HiddensNeuron; j++ {
			change := hiddenDeltas[j] * neural.InputActivations[i]
			neural.InputWeights[i][j] = neural.InputWeights[i][j] + learningRate*change + momentFactor*neural.InputChanges[i][j]
			neural.InputChanges[i][j] = change
		}
	}

	var err float64
	for i := 0; i < len(targets); i++ {
		err += 0.5 * math.Pow(targets[i]-neural.OutputActivations[i], 2)
	}

	return err
}

// Train ...
func (neural *NeuralNetwork) Train(patterns [][][]float64, iterations int, learningRate, momentFactor float64, debug bool) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var err float64
		for _, pattern := range patterns {
			neural.Update(pattern[0])

			tmp := neural.BackPropagate(pattern[1], learningRate, momentFactor)
			err += tmp
		}

		errors[i] = err

		if debug && i%1000 == 0 {
			fmt.Println(i, err)
		}
	}

	return errors
}

// Update ...
func (neural *NeuralNetwork) Update(inputs []float64) []float64 {
	mathematic := mathematics.NewSigmoid()

	if len(inputs) != neural.InputsNeuron-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	for i := 0; i < neural.InputsNeuron-1; i++ {
		neural.InputActivations[i] = inputs[i]
	}

	for i := 0; i < neural.HiddensNeuron-1; i++ {
		var sum float64

		for j := 0; j < neural.InputsNeuron; j++ {
			sum += neural.InputActivations[j] * neural.InputWeights[j][i]
		}

		// Soma contextos

		for k := 0; k < neural.ContextsNeuron; k++ {
			for j := 0; j < neural.HiddensNeuron-1; j++ {
				sum += neural.Contexts[k][j] * neural.ContextWeights[k][j][i]
			}
		}

		neural.HiddenActivations[i] = mathematic.Sigmoid(sum)
	}

	// Altera contextos
	if len(neural.Contexts) > 0 {
		for i := len(neural.Contexts) - 1; i > 0; i-- {
			neural.Contexts[i] = neural.Contexts[i-1]
		}
		neural.Contexts[0] = neural.HiddenActivations
	}

	for i := 0; i < neural.OutputsNeuron; i++ {
		var sum float64
		for j := 0; j < neural.HiddensNeuron; j++ {
			sum += neural.HiddenActivations[j] * neural.OutputWeights[j][i]
		}

		neural.OutputActivations[i] = mathematic.Sigmoid(sum)
	}

	return neural.OutputActivations
}

// Test ...
func (neural *NeuralNetwork) Test(patterns [][][]float64) {
	for _, pattern := range patterns {
		fmt.Println(fmt.Sprintf("%f", pattern[0]), " -> ", neural.Update(pattern[0]), " : ", fmt.Sprintf("%f", pattern[1]))
	}
}

// SetContexts ...
func (neural *NeuralNetwork) SetContexts(nContexts int, initValues [][]float64) {
	if initValues == nil {
		initValues = make([][]float64, nContexts)

		for i := 0; i < nContexts; i++ {
			initValues[i] = mathematics.Vector(neural.HiddensNeuron, 0.5)
		}
	}

	neural.ContextsNeuron = len(initValues)
	neural.ContextWeights = make([][][]float64, neural.ContextsNeuron)
	neural.ContextChanges = make([][][]float64, neural.ContextsNeuron)

	for i := 0; i < neural.ContextsNeuron; i++ {
		neural.ContextWeights[i] = mathematics.Matrix(neural.HiddensNeuron, neural.HiddensNeuron)
		neural.ContextChanges[i] = mathematics.Matrix(neural.HiddensNeuron, neural.HiddensNeuron)

		for j := 0; j < neural.HiddensNeuron; j++ {
			for k := 0; k < neural.HiddensNeuron; k++ {
				neural.ContextWeights[i][j][k] = mathematics.Random(-1, 1)
			}
		}
	}

	neural.Contexts = initValues
}
