# Report

## Implementation Choices
- n_epochs = 50
- Validate every 5 epochs

### Encoding
- Word-level tokenizer with 1000 words in dictionary and <unk> for rest of unknown words

### Hyperparameters
| Parameter | Value | Description |
|:---------:|:-----:|:-----------:|
| n_vocab | 1000 | Different number of vocabulary in the final dictionary
| n_embedding | 100 | dimension of embedding layer. Choose 100 for GloVe word embeddings
| n_hidden | 64 | dimension of hidden LSTM layer
| dropout_rate | 0.3 | dropout rate for dropout layer after the LSTM layer
| n_actions | 8 (num of distinct actions in provided dataset) | dimension of final fully connected layer for predicting action
| n_targets | 80 (num of distinct targets in provided dataset) | dimension of final fully connected layer for predicting target


### Model
- Embedding layer with dimension of 100 and weights initialized from GloVe word embeddings (To help the model to capture sub-linear relations between words in instructions more effectively than just randomly initializing weights for the embedding layer)
- LSTM with dimension of 64
- Dropout layer: to mitigate overfitting
- 2 separate Linear layers for predicting action(out_feature = 8) and target(out_feature = 80)

### Optimizer
Used Adam instead of SGD for better convergence to global minimum with momentum

### Loss Criterion
Used cross entropy loss for multi-class classification

## Performance
### LSTM with GloVe Embeddings & 2 independent prediction heads
![Result_2](performance_figures/result_1_(GloVe_with_2_independent_prediction_heads).png)
- Final Result (Approximate values)

    |                   |   Loss   | Accuracy |
    |:-----------------:|:--------:|:--------:|
    |  Training Action  | 0.001366 |  0.9848  | 
    |  Training Target  | 0.013345 |  0.8704  |
    | Validation Action | 0.005685 |  0.9608  |
    | Validation Target |  0.04248 |  0.7332  |
- Comments:
  - Easier to train the classifier on action since there's fewer number of action classes than target
  - Validation loss starts to increasing before training 10 epochs, especially for target objects, so there's overfitting to some extent
### LSTM with GloVe Embeddings & Target head taking in action head prediction
  ![Result_2](performance_figures/result_2(GloVe_with_Target_takes_in_action).png)
- Final Result (Approximate values)

  |                   |   Loss   | Accuracy |
  |:-----------------:|:--------:|:--------:|
  |  Training Action  | 0.001899 |  0.9818  | 
  |  Training Target  | 0.01809  |  0.8382  |
  | Validation Action | 0.005963 |  0.9596  |
  | Validation Target | 0.03956  |  0.7351  |
- Comment: Performance is slightly inferior to having 2 independent prediction heads
  
### LSTM with GloVe Embeddings & Action head taking in Target head prediction
![Result_3](performance_figures/result_3(GloVe_with_Action_takes_in_target).png)

  |                   |   Loss   | Accuracy |
  |:-----------------:|:--------:|:--------:|
  |  Training Action  | 0.001974 |  0.9808  | 
  |  Training Target  | 0.01817  |  0.8380  |
  | Validation Action | 0.006732 |  0.9583  |
  | Validation Target | 0.03779  |  0.7393  |
- Comment: have almost the same performance with having target head taking in prediction of action head

Having 2 separate prediction heads performs better than both having target head taking the prediction of action head and action head taking the prediction of target head. Intuitionally, action may depend on target since, for instance, the target "countertop" may more likely to be corresponded to a "GotoLocation" action. Nonetheless, from the performance displayed in the graphs above, this correlation may not be concluded.



## Other possible concerns/improvements
- Try with other hyperparameters, such as having 2 stacked LSTM layers instead of 1
- Try larger dimension for LSTM hidden layer