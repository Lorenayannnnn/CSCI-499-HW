python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --outputs_dir=outputs/experiments/s2s_with_attention \
    --batch_size=512 \
    --num_epochs=10 \
    --val_every=1 \
    --force_cpu \
    --run_seq_2_seq_bert = True