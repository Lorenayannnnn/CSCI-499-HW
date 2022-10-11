# Report
CBOW model is implemented and is experimented with 2 different context window lengths.

Table of Content
- [Implementation Choices](#implementation-choices)
  - [Encoding](#encoding)
  - [Hyperparameters](#hyperparameters)
  - [Model](#model)
  - [Optimizer](#optimizer)
  - [Loss Criterion](#loss-criterion)
- [Released Code Analysis](#released-code-analysis)
  - [Tasks](#tasks)
  - [Metrics](#metrics)
  - [Assumptions/Simplifications](#assumptionssimplifications-what-might-go-wrong-overunder-estimate-model-performance)
- [Performance](#Performance)
  - [CBOW-4-word-context: in vitro](#cbow-4-word-context-in-vitro)
    - [Parse context with all tokens](#parse-context-starting-from-the-very-1st-token-and-ending-at-the-very-last-token)
    - [Parse context with only tokens that have valid context window](#start-from-the-token-that-has-a-valid-context-window)
  - [CBOW-4-word-context: in vivo](#4-word-context-in-vivo)
  - [CBOW-8-word-context: in vitro](#8-words-as-context-in-vitro)
  - [CBOW-8-word-context: in vivo](#8-word-context-in-vivo)
- [Other Comments](#other-possible-concernsimprovements)

## Implementation Choices
- n_epochs = 10 (due to CPU limitation :( )
- Validate every 1 epoch
- 70% training, 30% validation

### Encoding
- Word-level tokenizer with 3000 words in dictionary (including 4 other special tokens \<pad\>, \<start\>, \<end\>, and \<unk\>)
#### CBOW
- Input: context window length of 4 is used. 2 words before a token is parsed as the input of that token. 2 different processings are experimented here:
  1. Parse context starting from the very 1st token and ending at the very last token within the sentence length (e.g.: 2 previous context words of the 1st token will be padding)
  2. Start from the token that has a valid context window (e.g.: if the context window length is 4, then the first token that will be processed is the 3rd token in the sentence)
- Output: target token

### Hyperparameters
| Parameter | Value | Description |
|:---------:|:-----:|:-----------:|
| n_vocab | 3000 | number of vocabulary in the dictionary
| n_embedding | 100 | dimension of embedding layer (chosen arbitrarily)
| context_window_len | 4/8 | total number of tokens used as the context of the target token

### CBOW Model
- Embedding layer with dimension of 100
- 1 fully connected layer for predicting token (out_feature=n_vocab)
- Forward: embedding vectors of 4/8 context words are first summed up and then passed into the fully connected layer to predict the final token

### Optimizer
Used Adam instead of SGD for better convergence to global minimum with momentum

### Loss Criterion
- CBOW: Used cross entropy loss for multi-class classification

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
- The first potential issue that I noticed is that the number of entries for each relation is not well-balanced. For instance, for semantic tasks, the "hypernomy" relation has 542 entries, whereas "capitals" has only 1 entry. Insufficient number of entries for certain relation/task may cause the metric to both overestimate/underestimate model's performance (the model made a correct/wrong prediction but the result is not representative)).
- Given that evaluation is done based on the result of C + B - A and actual word D, the metric assumes that the word embedding contains vector for the word D (i.e. D exists in our training corpus). However, if we train our model on a smaller dataset / has a smaller vocab size (smaller than 3000 in this case), then we cannot make a correct prediction.
- TODO

## Performance
### CBOW-4-word-context: in vitro
#### Parse context starting from the very 1st token and ending at the very last token
![CBOW_4_word_context_train_acc](output_graphs/training_acc(CBOW_with_all_tokens).png)
![CBOW_4_word_context_train_loss](output_graphs/training_loss(CBOW_with_all_tokens).png)
![CBOW_4_word_context_val_acc](output_graphs/validation_acc(CBOW_with_all_tokens).png)
![CBOW_4_word_context_val_loss](output_graphs/validation_loss(CBOW_with_all_tokens).png)

|                   |   Loss   | Accuracy |
|:-----------------:|:--------:|:--------:|
|     Training      |  1.6403  |  0.7030  | 
|    Validation     |  1.2243  |  0.7829  |

#### Start from the token that has a valid context window
##### 4-word-context: in vitro
![CBOW_4_word_context_train_acc](output_graphs/training_acc(CBOW_with_valid_window).png)
![CBOW_4_word_context_train_loss](output_graphs/training_loss(CBOW_with_valid_window).png)
![CBOW_4_word_context_val_acc](output_graphs/validation_acc(CBOW_with_valid_window).png)
![CBOW_4_word_context_val_loss](output_graphs/validation_loss(CBOW_with_valid_window).png)

|                   |   Loss   | Accuracy |
|:-----------------:|:--------:|:--------:|
|     Training      |  1.6312  |  0.7140  | 
|    Validation     |  1.2354  |  0.7892  |

Comments: 
- As can be seen from above, extracting tokens with valid context windows (i.e. ignore words that do not have enough context words before/after them) increases models' performance (by 1%).
This is reasonable since there will be much fewer padding tokens in the input data, and most 2 words, at the beginning of a sentence for instance, are usually stop words, which have less influence on other relatively more important words.
- Noted: the validation and training loss are still both decreasing, and validation and training accuracy are still both increasing at the end of 10 epochs, so maybe training the model with more epochs can further increase model's performance.

---- For the rest of the report, only tokens with valid context window are included in the dataset ----

##### 4-word-context: in vivo
|                   |   Exact   | MRR | MR |
|:-----------------:|:--------:|:--------:|:--------:|
|     Overall      |  0.0290  |  0.0547  |  18  | 
|    Semantics     |  0.0062  |  0.0197  |  51  |
|    Syntax     |  0.0941  |  0.1545  |  6  |

###### Semantics
![CBOW_in_vivo_sem](output_graphs/in_vivo_semantics(CBOW).png)
###### Syntax
![CBOW_in_vivo_syn](output_graphs/in_vivo_syntactic(CBOW).png)

Comments: as can be seen from above, the model performs better at capturing syntactic information than semantics, which is reasonable since only 4 words surrounding a target word are used as the word's context, so it's difficult for the model to learn global semantic relations.

##### 8 words as context in vitro
![CBOW_8_word_context_train_acc](output_graphs/training_acc(CBOW_larger_window).png)
![CBOW_8_word_context_train_loss](output_graphs/training_loss(CBOW_larger_window).png)
![CBOW_8_word_context_val_acc](output_graphs/validation_acc(CBOW_larger_window).png)
![CBOW_8_word_context_val_loss](output_graphs/validation_loss(CBOW_larger_window).png)

|                   |   Loss   | Accuracy |
|:-----------------:|:--------:|:--------:|
|     Training      |  1.5751  |  0.7272  | 
|    Validation     |  1.1877  |  0.8030  |

Comments: Compared with the result of 4-word-context (in vitro) CBOW model, the performance slightly increases, which is expected as the model is taking more words into account and learning more context information (both semantic and syntactic information).  

##### 8-word-context: in vivo
|                   |   Exact   | MRR | MR |
|:-----------------:|:--------:|:--------:|:--------:|
|     Overall      |  0.0260  |  0.0459  |  22  | 
|    Semantics     |  0.0072  |  0.0176  |  57  |
|    Syntax     |  0.0794  |  0.1267  |  8  |

Comments:
- Compared with the in-vivo evaluation result of 4-word-context CBOW model, the 8-word-context model performs slightly better at semantic tasks. Specifically, it performs slightly better on the "exact" metric (though the value is still very low), which is also expected as the model is taking more words as context, so it may be better at capturing more global/semantic information.
- However, it overall performs worse than the 4-word-context model. My speculation is that given the model only has 1 embedding layer and 1 linear layer, the model may not be complex enough to learn all the underlying relations based on input 8-word-context. 


## Other possible concerns/improvements
- Try different dimensions for the embedding layer
- Try other model architecture, such as adding layers to learn a more complex relation
- Add sampling scheme