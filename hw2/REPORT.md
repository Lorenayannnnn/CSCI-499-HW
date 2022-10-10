# Report
CBOW model is implemented.

## Implementation Choices
- n_epochs = 10
- Validate every 1 epoch

### Encoding
- Word-level tokenizer with 3000 words in dictionary and 4 other special tokens (\<pad\>, \<start\>, \<end\>, and \<unk\>)
#### CBOW
- Input: context window length of 4 is used. 2 words before a token is parsed as the input of that token
- Output: target token
#### Skipgram
- Input: individual tokens within sentences
- Output: 4 words surrounding the token (2 words before and 2 words after) are used as the context

### Hyperparameters
| Parameter | Value | Description |
|:---------:|:-----:|:-----------:|
| n_vocab | 3000+4 | number of vocabulary in the dictionary
| n_embedding | 128 | dimension of embedding layer (chosen arbitrarily)
| context_window_len | 4 | total number of tokens used as the context of the target token

### Model
#### CBOW
- Embedding layer with dimension of 128
- 1 fully connected layer for predicting token (out_feature=n_vocab)
- Forward: embedding vectors of 4 context words are first summed up and then passed into the fully connected layer to predict the final token

#### Skipgram
- Embedding layer with dimension of 128
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
### In Vitro TODO
|   Task   | Metric |
|:--------:|:--------:|
|    |    | 
|    |    |
- Assumptions/Simplifications (what might go "wrong", over/under-estimate model performance)

### In Vivo TODO
|   Task   | Metric |
|:--------:|:--------:|
|    |    | 
|    |    |
- Assumptions/Simplifications (what might go "wrong", over/under-estimate model performance)

## Performance
### CBOW
![CBOW_train_acc](output_graphs/training_acc(CBOW).png)
![CBOW_train_loss](output_graphs/training_loss(CBOW).png)
![CBOW_val_acc](output_graphs/validation_acc(CBOW).png)
![CBOW_val_loss](output_graphs/validation_loss(CBOW).png)

|                   |   Loss   | Accuracy |
|:-----------------:|:--------:|:--------:|
|     Training      | 0.7675   |  0.8297  | 
|    Validation     | 0.9352   |  0.8206  |

- Comments: TODO

#### In Vitro TODO
#### In Vivo TODO

### Skipgram TODO
#### In Vitro TODO
#### In Vivo TODO


## Other possible concerns/improvements
- Try with other hyperparameters, such as having 2 or more stacked LSTM layers instead of 1 to represent more complex relations
- Try different dimensions for the embedding layer
- Try other model architecture, such as adding layers to learn a more complex relation
- Add negative sampling to Skipgram