CUDA_VISIBLE_DEVICES=1 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRPt \
    --output-path results/AttnUNet_SVRE \
    --model AttnUNet \
    --predict \
    --train-ratio 0.7 \
    --valid-ratio 0.1 \
    --sample-indices 16840 17190 \
    --max-iterations 80000 \
    --early-stopping \
    --lambda-var 10 \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 10 \
    > results/AttnUNet_SVRE.log 2>&1 &
