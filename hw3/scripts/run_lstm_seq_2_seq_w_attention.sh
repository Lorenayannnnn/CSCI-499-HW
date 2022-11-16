caffeinate -s python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --outputs_dir=outputs/experiments/s2s_with_attention/ \
    --model_output_filename=s2s_w_attention_model.ckpt \
    --batch_size=256 \
    --num_epochs=10 \
    --val_every=2 \
    --force_cpu \
    --teacher_forcing \
    --encoder_decoder_attention