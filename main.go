package main

import (
	"PinkyTheBrain/cognitive"
	"PinkyTheBrain/common"
	"PinkyTheBrain/neuralnetwork"
	"PinkyTheBrain/nltk"
	"fmt"
	"log"
	"math/rand"
	"strconv"
	"strings"
)

func main() {
	vocabularyNCM := nltk.GetVocabularyNCM()

	vocabularyTraining := nltk.GetVocabularyTraining()
	patterns := nltk.ConvertWordsVocabulary(vocabularyTraining)
	nltk.ConvertWordsVocabularyNCM(vocabularyTraining, vocabularyNCM)

	naivebayes := cognitive.NewNaivebayes(vocabularyNCM)
	naivebayes.TFIDF = true
	var wordbook = common.ReadJSONToken("dictionary.json")
	for _, entity := range vocabularyNCM {
		if wordbook[entity] != nil {
			naivebayes.Learn(wordbook[entity], entity)
		}
	}

	naivebayes.Convert = true

	var sampling = 0
	var hitRate = 0
	var errorRate = 0

	// ANALISE TESTE DE DADOS
	analyze := nltk.GetVocabularyTest()
	for description, clasfisc := range analyze {
		bagWord := []string{}
		sentenceList := strings.SplitAfter(description, " ")
		for _, word := range sentenceList {
			bagWord = append(bagWord, nltk.ExtractChar(word))
		}

		_, _, _, ncm := naivebayes.LogScores(bagWord)

		strconvNcm := strconv.FormatFloat(clasfisc, 'f', -1, 64)

		if strconvNcm == ncm {
			hitRate++
		} else {
			// log.Print("| DESCRIÇÃO: ", description, " => NCM PREVISTO: ", strconvNcm, " <=> PROPOSTO: ", ncm, "\n")
			errorRate++
		}

		sampling++
	}

	rate := ((float64(hitRate) * float64(100)) / float64(sampling))
	log.Print("                       ANÁLISE \n")
	log.Print("********************************************************* \n")
	log.Print("| AMOSTRAGEM: ", sampling, " | ACERTOS: ", hitRate, " | ERROS: ", errorRate, " |  TAXA DE ACERTOS = ", fmt.Sprintf("%.2f", rate), " |")

	var lenght = 1

	// defina a geração aleatória como 0
	rand.Seed(0)

	// dados para ser analizado pela rede a rede
	vocabularyTest := nltk.GetVocabularyTest()
	patternsTest := nltk.ConvertWordsVocabulary(vocabularyTest)

	contexts := [][]float64{
		{0.5, 0.8, 0.1},
	}

	neural := neuralnetwork.NewNeuralNetwork()
	neural.FeedForward(lenght, lenght, 1)
	neural.Train(patterns, 1000, 0.6, 0.4, true)
	neural.SetContexts(1, contexts)
	neural.Test(patternsTest)
}
