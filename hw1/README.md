This project utilizes LSTM and GloVe word embeddings. Please use the following instructions to run the program:

(P.S.: The original README.md is renamed to [Assignment_Instructions](Assignment_Instructions.md))

### Clone the repo & install requirements
```
git clone git@github.com:Lorenayannnnn/CSCI-499-HW.git
cd hw1/
virtualenv -p $(which python3) ./hw1
source ./hw1/bin/activate
pip3 install -r requirements.txt
```

### Download GloVe
GloVe with dimension of 100 is used. Please download [GloVe](https://drive.google.com/file/d/1n15zWXLjxjqX72R6dHyAPTA3N5b22c_U/view?usp=sharing) and put it directly under `hw1` directory. 

### Run the program
(Due to the limitation of my computer I only ran 50 epochs🥲)
```
cd hw1/
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/lstm \
    --batch_size=32 \
    --num_epochs=50 \
    --val_every=5 \
    --force_cpu 
```

### Report
Discussions of implementation choices and performances are documented in [REPORT.md](REPORT.md)