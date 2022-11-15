python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --outputs_dir=outputs/experiments/s2s/ \
    --model_output_filename=s2s_model.ckpt \
    --batch_size=1 \
    --num_epochs=100 \
    --val_every=2 \
    --force_cpu \
    --teacher_forcing=True \
    --encoder_decoder_attention=False