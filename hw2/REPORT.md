# Report
CBOW model is implemented.

## Implementation Choices
- n_epochs = 10 (due to CPU limitation :( )
- Validate every 1 epoch

### Encoding
- Word-level tokenizer with 3000 words in dictionary (including 4 other special tokens \<pad\>, \<start\>, \<end\>, and \<unk\>)
#### CBOW
- Input: context window length of 4 is used. 2 words before a token is parsed as the input of that token. 2 different processings are experimented here:
  1. Parse context starting from the very 1st token and ending at the very last token within the sentence length (e.g.: 2 previous context words of the 1st token will be padding)
  2. Start from the token that has a valid context window (e.g.: if the context window length is 4, then the first token that will be processed is the 3rd token in the sentence)
- Output: target token
#### Skipgram
- Input: individual tokens within sentences
- Output: 4 words surrounding the token (2 words before and 2 words after) are used as the context

### Hyperparameters
| Parameter | Value | Description |
|:---------:|:-----:|:-----------:|
| n_vocab | 3000 | number of vocabulary in the dictionary
| n_embedding | 100 | dimension of embedding layer (chosen arbitrarily)
| context_window_len | 4 | total number of tokens used as the context of the target token

### Model
#### CBOW
- Embedding layer with dimension of 100
- 1 fully connected layer for predicting token (out_feature=n_vocab)
- Forward: embedding vectors of 4 context words are first summed up and then passed into the fully connected layer to predict the final token

#### Skipgram
- Embedding layer with dimension of 100
- 1 fully connected layer for predicting the context (out_feature=n_vocab)
- Forward: the input token is first embedded and then is passed into the fully connected layer to predict its context

### Optimizer
Used Adam instead of SGD for better convergence to global minimum with momentum

### Loss Criterion
#### CBOW
Used cross entropy loss for multi-class classification
#### Skipgram
Used BCEWithLogitsLoss for multi-label classification


## Released Code Analysis
### Tasks
- In vitro: tested on validation dataset. For instance, for CBOW, the model is still tested on its ability of predicting the target word based on input context. 
- In vivo: Several tasks are involved for in vivo evaluation. All tasks are broken down into 2 categories: semantic and syntactic. 
  - Relations of semantic tasks: capitals, binary gender, antonym, member, hypernomy, similar… 
  - Relations of syntatic tasks: adj and adv, comparative, superlative, plural nouns...

### Metrics
#### In vitro
For accuracy, we select the token with the highest probability and then check if it matches with the actual token. For loss, cross entropy loss is used here. 
#### In vivo 
- For one relation of a task, 1 or more entries are included. For each entry, there are 2 pairs of words that follow the declared relation. 
- Let the 4 words be A, B, C, and D. A and B follows the stated relation, so as C and D. 
- The model is evaluated on whether the learned word embedding space satisfies the equality A-B = C-D
- The evaluation util function first uses <code>gensim.models.KeyedVectors</code> to store and calculate C + B - A. Then, top 1000 words are sampled from the space based on the result.
- 3 values are calculated:
  - Exact: top-1 choice matches with the actual word
  - MRR: if top-1 choice does not match the actual word, then the score is calculated based on the predicted word's ranking (1/word's ranking)
  - MR: inverse of MRR


### Assumptions/Simplifications (what might go "wrong", over/under-estimate model performance)
- The first potential issue that I noticed is that the number of entries for each relation is not well-balanced. For instance, for semantic tasks, the "hypernomy" relation has 542 entries, which is way more than rest of the relations (detail can be seen in graphs in the _**Performance**_ section). The out-of-distribution issue may cause the metric to underestimate the learned word embedding's ability of handling other relations.
- TODO

## Performance
### CBOW: In Vivo
#### Parse context starting from the very 1st token and ending at the very last token
![CBOW_train_acc](output_graphs/training_acc(CBOW_with_all_tokens).png)
![CBOW_train_loss](output_graphs/training_loss(CBOW_with_all_tokens).png)
![CBOW_val_acc](output_graphs/validation_acc(CBOW_with_all_tokens).png)
![CBOW_val_loss](output_graphs/validation_loss(CBOW_with_all_tokens).png)

|                   |   Loss   | Accuracy |
|:-----------------:|:--------:|:--------:|
|     Training      |  1.6403  |  0.7030  | 
|    Validation     |  1.2243  |  0.7829  |

#### Start from the token that has a valid context window
![CBOW_train_acc](output_graphs/training_acc(CBOW_with_valid_window).png)
![CBOW_train_loss](output_graphs/training_loss(CBOW_with_valid_window).png)
![CBOW_val_acc](output_graphs/validation_acc(CBOW_with_valid_window).png)
![CBOW_val_loss](output_graphs/validation_loss(CBOW_with_valid_window).png)

|                   |   Loss   | Accuracy |
|:-----------------:|:--------:|:--------:|
|     Training      |  1.6312  |  0.7140  | 
|    Validation     |  1.2354  |  0.7892  |

Comments: 
- As can be seen from above, extracting tokens with valid context windows (i.e. ignore words that do not have enough context words before/after them) increases models' performance (by 1%).
This is reasonable since there will be much fewer padding tokens in the input data, and most 2 words, at the beginning of a sentence for instance, are usually stop words, which have less influence on other relatively more important words.
- Noted: the validation and training loss are still both decreasing, and validation and training accuracy are still both increasing at the end of 10 epochs, so maybe training the model with more epochs can further increase model's performance.

### CBOW: In Vivo
#### Semantics
![CBOW_in_vivo_sem](output_graphs/in_vivo_semantics(CBOW).png)
#### Syntax
![CBOW_in_vivo_syn](output_graphs/in_vivo_syntactic(CBOW).png)

Comments: as can be seen from above, the model performs better at capturing syntactic information than semantics, which is reasonable since only 4 words surrounding a target word are used as the word's context, so it's difficult for the model to learn global semantic relations. 

### Skipgram TODO
#### In Vitro TODO
#### In Vivo TODO


## Other possible concerns/improvements
- Try with other hyperparameters, such as having 2 or more stacked LSTM layers instead of 1 to represent more complex relations
- Try different dimensions for the embedding layer
- Try other model architecture, such as adding layers to learn a more complex relation
- Add negative sampling to Skipgram