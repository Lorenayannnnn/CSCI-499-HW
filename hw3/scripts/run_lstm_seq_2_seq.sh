python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --outputs_dir=outputs/experiments/s2s/ \
    --batch_size=16 \
    --num_epochs=50 \
    --val_every=5 \
    --force_cpu \
    --teacher_forcing=True \
    --encoder_decoder_attention=False