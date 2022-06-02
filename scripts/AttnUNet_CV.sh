CUDA_VISIBLE_DEVICES=1 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/AttnUNet_CV \
    --model AttnUNet \
    --train \
    --test \
    --predict \
    --sample-index 16840 \
    --max-iterations 100000 \
    --early-stopping \
    --batch-size 16 \
    --var-reg 0.5 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > AttnUNet_CV.log 2>&1 &
