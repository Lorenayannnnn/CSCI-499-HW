python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --outputs_dir=outputs/experiments/s2s \
    --batch_size=32 \
    --num_epochs=5 \
    --val_every=1 \
    --force_cpu \
    --teacher_forcing=True