CUDA_VISIBLE_DEVICES=1 \
nohup python -u train_gan.py \
    --train \
    --test \
    --predict \
    --data-path /data2/gaf/SBandCR_PT \
    --output-path results/AGAN_SVRE \
    --num-ensembles 6 \
    --max-iterations 100000 \
    --early-stopping \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --weight-recon 10 \
    --weight-svre 1 \
    --num-threads 8 \
    --num-workers 8 \
    --case-indices 16840 17190 \
    --display-interval 50 \
    --thresholds 20 30 40 \
    > results/train_agan_svre.log 2>&1 &
