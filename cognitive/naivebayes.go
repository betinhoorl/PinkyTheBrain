package cognitive

import (
	"math"
	"sync/atomic"
)

// DataSet ...
type DataSet struct {
	Frequency    map[string]float64
	FrequencyTfs map[string][]float64
	Total        int
}

// NewDataSet ...
func NewDataSet() *DataSet {
	return &DataSet{
		Frequency:    make(map[string]float64),
		FrequencyTfs: make(map[string][]float64),
	}
}

func (d *DataSet) getWordProb(word string) float64 {
	value, ok := d.Frequency[word]

	if !ok {
		return defaultProb
	}

	return float64(value) / float64(d.Total)
}

// Naivebayes ...
type Naivebayes struct {
	Dictionary []string
	base       map[string]*DataSet

	seen    int32
	trained int
	word    string

	// Term frequency–inverse wordbook frequency
	TFIDF   bool
	Convert bool
}

// Constantes
const defaultProb = 0.00000000001

// NewNaivebayes ...
func NewNaivebayes(dictionary []string) (naivebayes *Naivebayes) {
	size := len(dictionary)

	if size < 2 {
		panic("Necessário informar ao menos duas classes")
	}

	check := make(map[string]bool, size)
	for _, class := range dictionary {
		check[class] = true
	}

	if len(check) != size {
		panic("Cl")
	}

	naivebayes = &Naivebayes{
		Dictionary: dictionary,
		base:       make(map[string]*DataSet, size),
		TFIDF:      false,
		Convert:    false,
	}

	for _, class := range dictionary {
		naivebayes.base[class] = NewDataSet()
	}

	return naivebayes
}

// Learn ...
func (n *Naivebayes) Learn(wordbook []string, entity string) {

	if n.TFIDF {
		if n.Convert {
			panic("Não é possível chamar Conversor de termos de frequência TfIdf mais de uma vez. Repor e reaprender para reconverter..")
		}

		docLen := float64(len(wordbook))
		docTFIDF := make(map[string]float64)

		for _, word := range wordbook {
			docTFIDF[word]++
		}

		for wIndex, wCount := range docTFIDF {
			docTFIDF[wIndex] = wCount / docLen
			n.base[entity].FrequencyTfs[wIndex] = append(n.base[entity].FrequencyTfs[wIndex], docTFIDF[wIndex])
		}
	}

	data := n.base[entity]
	for _, word := range wordbook {
		data.Frequency[word]++
		data.Total++
	}

	n.trained++
}

// LogScores ...
func (n *Naivebayes) LogScores(wordbook []string) (scores []float64, inx int, strict bool, entity string) {
	if n.TFIDF && !n.Convert {
		panic("Usando um classificador TF-IDF. Para Conversor de termos de frequência TfIdf antes de chamar LogScores.")
	}

	size := len(n.Dictionary)
	scores = make([]float64, size, size)
	priors := n.getPriors()

	// calcular a pontuação para cada Entidade
	for index, class := range n.Dictionary {
		data := n.base[class]
		score := math.Log(priors[index])
		for _, word := range wordbook {
			score += math.Log(data.getWordProb(word))
		}
		scores[index] = score

		// log.Print(class, ": ", score, "\n")
	}

	inx, strict = findMax(scores)
	atomic.AddInt32(&n.seen, 1)
	result := n.Dictionary[inx]

	return scores, inx, strict, result
}

func (n *Naivebayes) getPriors() (priors []float64) {
	size := len(n.Dictionary)
	priors = make([]float64, size, size)
	sum := 0

	for index, class := range n.Dictionary {
		total := n.base[class].Total
		priors[index] = float64(total)
		sum += total
	}

	if sum != 0 {
		for i := 0; i < size; i++ {
			priors[i] /= float64(sum)
		}
	}

	return priors
}

func findMax(scores []float64) (inx int, strict bool) {
	findProbable(scores)
	inx = 0
	strict = true

	for i := 1; i < len(scores); i++ {
		if scores[inx] < scores[i] {
			inx = i
			strict = true
		} else if scores[inx] == scores[i] {
			strict = false
		}
	}

	return inx, strict
}

func findProbable(scores []float64) []int {
	//var inx = 0
	var indexs []map[float64]int
	//var ind []int

	for i := 1; i < len(scores); i++ {
		var s = make(map[float64]int)

		s[scores[i]] = i
		indexs = append(indexs, s)
		// for _, maps := range indexs {
		// 	for key, value := range maps {
		// 		if scores[inx] > key {
		// 			ind = append(ind, value)

		// 		}
		// 	}
		// }
		// inx = i
	}

	return []int{}
}
