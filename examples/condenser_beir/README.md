
# Introduction
Here we provide the code for reproducing results for BEIR using deep prompt learning of prompts' length as 128.

## Data Preparation
### Data Download
Training Datasets: [BERRI](https://github.com/facebookresearch/tart/tree/main/BERRI)  
Evaluation Datasets: [BEIR](https://github.com/beir-cellar/beir)

### Data Format
Decompose task instructions into multiple task attributes, and format the data as follows:

**Training**: Each line of the Train file is a trainng instance:
```
{'query': TEXT_TYPE, 'positives': List[TEXT_TYPE], 'negatives': List[TEXT_TYPE], 'attributes': List[RAW_STRING]}
...
```

**Inference/Encoding Queries**: Each line of the encoding file is a piece of query to be encoded:
```
{'_id': ANY_TYPE, 'text': TEXT_TYPE, 'attributes': List[RAW_STRING]}
...
```

**Inference/Encoding Passages**: Each line of the encoding file is a piece of passage to be encoded:
```
{'_id': ANY_TYPE, 'text': TEXT_TYPE}
...
```
`TEXT_TYPE` can be either raw string or pre-tokenized ids. `ANY_TYPE` means any type of objects in python. `RAW_STRING` indicates attribute labels for each instance should only be raw strings. The attribute labels can be obtained in various ways (e.g., task instruction decomposition), in the data preprocessing stage.


## Reproduction
With the training data and evaluation data ready, we can start training the retrieval modules, yeah!  
(FYI, all given programs are for reference only, please read the source codes for better understanding and feel free to modify them according to your needs.)

1. Train a general prompt on a group of general retrieval tasks: `sh train_phase1.sh`
2. Jointly train multiple attribute prompts as well as the trained general prompt on all available tasks: `sh train_phase2.sh`
3. Based on the trained prompts which should be saved on the *trained_prompts* directory, we can encode the queries and passages on evaluation tasks: `sh encode.sh`
4. Search the closest passages for the given queries: `sh search.sh`  
