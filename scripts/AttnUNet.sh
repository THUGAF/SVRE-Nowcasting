CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/AttnUNet \
    --model AttnUNet \
    --train \
    --test \
    --predict \
    --sample-index 16840 \
    --batch-size 16 \
    --random-crop-num 1 \
    --max-iterations 100000 \
    --early-stopping \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 10 \
    > AttnUNet.log 2>&1 &
