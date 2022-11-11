python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/s2s \
    --batch_size=128 \
    --num_epochs=5 \
    --val_every=1 \
    --force_cpu