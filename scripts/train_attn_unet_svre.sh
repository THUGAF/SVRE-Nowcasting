CUDA_VISIBLE_DEVICES=0 \
nohup python -u train_det.py \
    --train \
    --test \
    --predict \
    --data-path /data/gaf/SBandCRPt \
    --output-path results/AttnUNet_SVRE \
    --model AttnUNet \
    --max-iterations 100000 \
    --early-stopping \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --weight-recon 10 \
    --weight-svre 1 \
    --num-threads 8 \
    --num-workers 8 \
    --case-indices 16840 17190 \
    --display-interval 50 \
    --thresholds 20 30 40 \
    > results/train_attn_unet_svre.log 2>&1 &
