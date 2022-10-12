This project implements CBOW, which is experimented with 2 different context window length.

### Report
Discussions of implementation choices and performances are documented in [REPORT.md](REPORT.md)

(P.S.: The original README.md is renamed to [HW2_Instructions](HW2_Instructions.md))

Please use the following instructions to run the program:
### Clone the repo & install requirements
```
git clone git@github.com:Lorenayannnnn/CSCI-499-HW.git
cd hw2/
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt
bash get_books.sh
```

### Run the program
```
cd hw2/

- Training:
python3 train.py \
    --analogies_fn analogies_v3000_1309.json \
    --data_dir books/ \
    --word_vector_fn learned_word_vectors.txt

- Evaluation:
python train.py \
    --analogies_fn analogies_v3000_1309.json \
    --data_dir books/ \
    --downstream_eval \
    --outputs_dir output/ \
    --word_vector_fn learned_word_vectors_CBOW_len_2.txt

word_vector_fn = learned_word_vectors_CBOW_len_2.txt or learned_word_vectors_CBOW_len_4.txt
```