# CSCI-499 Natural Language for Interactive AI
- Author: Tianyi(Lorena) Yan

- This repo contains all coding assignments for this course.

## HW1
This project utilizes LSTM and GloVe word embeddings. Please use the following instructions to run the program:

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
```
cd hw1/
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/lstm \
    --batch_size=1000 \
    --num_epochs=100 \
    --val_every=5 \
    --force_cpu 
```    

### Report
Discussions of implementation choices and performances are documented in `hw1/REPORT.md`