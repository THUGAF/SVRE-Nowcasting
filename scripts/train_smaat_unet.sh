CUDA_VISIBLE_DEVICES=0 \
nohup python -u train_det.py \
    --train \
    --test \
    --predict \
    --data-path /data/gaf/SBandCRPt \
    --output-path results/SmaAt_UNet \
    --model SmaAt_UNet \
    --max-iterations 100000 \
    --early-stopping \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --weight-recon 10 \
    --num-threads 8 \
    --num-workers 8 \
    --case-indices 16840 17190 \
    --display-interval 50 \
    --thresholds 20 30 40 \
    > results/train_smaat_unet.log 2>&1 &
