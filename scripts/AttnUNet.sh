CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/AttnUNet \
    --model AttnUNet \
    --train \
    --test \
    --predict \
    --train-ratio 0.7 \
    --valid-ratio 0.1 \
    --sample-indices 16840 17190 \
    --max-iterations 50000 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 10 \
    > results/AttnUNet.log 2>&1 &
