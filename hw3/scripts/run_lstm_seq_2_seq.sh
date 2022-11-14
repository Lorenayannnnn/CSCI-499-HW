python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --outputs_dir=outputs/experiments/s2s/ \
    --batch_size=8 \
    --num_epochs=10 \
    --val_every=1 \
    --force_cpu \
    --teacher_forcing=True \
    --encoder_decoder_attention=False