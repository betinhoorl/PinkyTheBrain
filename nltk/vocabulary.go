package nltk

import (
	"bufio"
	"log"
	"os"
	"strconv"
	"strings"
)

// GetVocabulary ...
func GetVocabulary() []string {
	var vocab []string

	file, err := os.Open("training.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	iterator := 0
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var descriptions = strings.Split(scanner.Text(), ";")
		for _, description := range descriptions {
			var words = Tokenize(description)
			for _, word := range words {
				vocab = append(vocab, word)
				iterator++
			}
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	vocabulary := removeDuplicateWords(vocab)
	return vocabulary
}

// GetVocabularyTraining ...
func GetVocabularyTraining() map[string]float64 {
	var maps = make(map[string]float64)

	file, err := os.Open("training.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var words = strings.Split(scanner.Text(), ";")
		if ncm, err := strconv.ParseFloat(words[3], 64); err == nil {
			maps[words[0]] = ncm
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	return maps
}

// GetVocabularyTest ...
func GetVocabularyTest() map[string]float64 {
	var maps = make(map[string]float64)

	file, err := os.Open("test.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var words = strings.Split(scanner.Text(), ";")
		if ncm, err := strconv.ParseFloat(words[3], 64); err == nil {
			maps[words[0]] = ncm
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	return maps
}

// GetVocabularyNCM ...
func GetVocabularyNCM() []string {
	var vocabulary []string

	file, err := os.Open("ncm.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	iterator := 0
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		vocabulary = append(vocabulary, scanner.Text())
		iterator++
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	return vocabulary
}
