CUDA_VISIBLE_DEVICES=2 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/Encoder_Forecaster \
    --model EncoderForecaster \
    --train \
    --test \
    --predict \
    --sample-index 16840 \
    --max-iterations 100000 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > Encoder_Forecaster.log 2>&1 &
