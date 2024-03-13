package common

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

// NCM ...
type NCM struct {
	WORDS map[string][]string
}

// ReadJSONToken ...
func ReadJSONToken(fileName string) map[string][]string {
	ncms := make(map[string][]string)

	file, err := ioutil.ReadFile(fileName)
	if err != nil {
		fmt.Println(err.Error())
		os.Exit(1)
	}

	json.Unmarshal(file, &ncms)
	return ncms
}

// WriteJSONToken ...
func WriteJSONToken(fileName string, jsonStream map[string][]string) {
	// file, _ := os.OpenFile(fileName, os.O_CREATE, os.ModePerm)
	// log.Print("FILE: ", file) 
	// defer file.Close()
	// encoder := json.NewEncoder(file)
	// encoder.Encode(jsonStream)

	jsonString, err := json.Marshal(jsonStream)
	if err != nil {
		fmt.Println(err)
	}

	err = ioutil.WriteFile(fileName, jsonString, 0644)
	if err != nil {
		fmt.Println(err)
	}
	
}
