CUDA_VISIBLE_DEVICES=1 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/SmaAt_UNet \
    --model SmaAt_UNet \
    --train \
    --test \
    --predict \
    --sample-indices 16840 17190 \
    --max-iterations 50000 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > results/SmaAt_UNet.log 2>&1 &
