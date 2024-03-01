package nltk

import (
	"PinkyTheBrain/common"
	"strconv"
	"strings"
)

// Tokenize ...
func Tokenize(text string) []string {
	var words = strings.Split(text, " ")
	var stopwords = stopwords()
	var vocab []string

	for _, word := range words {
		var remove = false
		for _, stopword := range stopwords {
			if stopword == strings.ToLower(word) {
				remove = true
				break
			}
		}

		if !remove {
			word = ExtractChar(word)
			vocab = append(vocab, word)
		}
	}
	var vector = removeDuplicateWords(vocab)
	return vector
}

// FreqDist ...
func FreqDist(words []string) map[string]int {
	var maps = make(map[string]int)
	for _, word := range words {
		maps[word] += +1
	}
	return maps
}

// ConvertWordsVocabularyNCM ...
func ConvertWordsVocabularyNCM(vocabulary map[string]float64, ncm []string) {
	var maps = make(map[string][]string)

	for key, value := range vocabulary {
		for _, classifc := range ncm {
			if ncm, err := strconv.ParseFloat(classifc, 64); err == nil {
				if value == ncm {
					var words = Tokenize(key)
					for key, value := range maps {
						if key == classifc {
							for _, word := range value {
								words = append(words, word)
							}
						}
					}
					var vector = removeDuplicateWords(words)
					maps[classifc] = vector
				}
			}
		}
	}

	common.WriteJSONToken("dictionary.json", maps)

}

// ConvertWordsVocabulary ...
func ConvertWordsVocabulary(vocabulary map[string]float64) [][][]float64 {
	var iterations = len(vocabulary)
	var patterns = make([][][]float64, iterations)
	var vocab = GetVocabulary()

	iterator := 0
	for key, value := range vocabulary {
		words := Tokenize(key)
		var token = ConvertWordsToIndexes(vocab, words)

		// Inicialize
		patterns[iterator] = make([][]float64, 2)

		// Valores de entrada dos neuronios
		patterns[iterator][0] = make([]float64, 1)
		patterns[iterator][0] = []float64{token}

		// valores esperados na saída
		patterns[iterator][1] = make([]float64, 1)
		patterns[iterator][1] = []float64{value}

		iterator++
	}

	return patterns
}

// ConvertWordsToIndexes ...
func ConvertWordsToIndexes(vocab []string, words []string) float64 {
	token := ""
	for _, word := range words {
		for key, test := range vocab {
			if strings.ToLower(word) == strings.ToLower(test) {
				token += strconv.Itoa(key)
				break
			}
		}
	}

	tokenResult := float64(0)
	if ok, err := strconv.ParseFloat(token, 64); err == nil {
		tokenResult = ok
	}

	return tokenResult
}

// RadicalExtract ...
func RadicalExtract(word string) string {
	word = strings.ToLower(word)
	dictionaryOrigin := []string{"á", "é", "í", "ó", "ú", "à", "è", "ì", "ò", "ù", "â", "ê", "î", "ô", "û", "ã", "õ", "ç", ".", ",", "?", "!", ":", ";"}
	dictionaryTranform := []string{"a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "a", "o", "c", "", "", "", "", "", ""}

	for index, char := range dictionaryOrigin {
		word = strings.Replace(word, char, dictionaryTranform[index], -1)
	}

	if strings.TrimSpace(word) != "" {
		word = ExtractChar(word)
	}

	return strings.TrimSpace(word)
}

// ExtractChar ...
func ExtractChar(word string) string {
	var extracted = true

	dictionaryVowels := []string{"a", "e", "i", "o", "u"}
	dictionaryConsonants := []string{"r", "s", "c", ".", ",", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

	lenght := len(word)
	word = strings.ToLower(strings.TrimSpace(word))
	if lenght > 3 {
		charLast := string(word[len(word)-1])
		charPenultimate := string(word[len(word)-2])

		for _, vowel := range dictionaryVowels {
			if charLast == vowel {
				word = strings.TrimRight(word, charLast)
				extracted = false
			}

			if extracted {
				for _, consonant := range dictionaryConsonants {
					if charLast == consonant && charPenultimate == vowel {
						word = strings.TrimRight(word, charLast)
						extracted = false
					}
				}
			}
		}
	}

	if !extracted {
		ExtractChar(word)
	}

	return word
}

func removeDuplicateWords(words []string) []string {
	var maps = make(map[string]string)
	for _, word := range words {
		maps[word] = word
	}

	vector := make([]string, len(maps))

	count := 0
	for _, value := range maps {
		vector[count] = value
		count++
	}

	return vector
}
