CUDA_VISIBLE_DEVICES=1 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/AttnUNet_GAN \
    --model AttnUNet \
    --add-gan \
    --train \
    --test \
    --predict \
    --ensemble-members 3 \
    --sample-index 16840 \
    --max-iterations 50000 \
    --early-stopping \
    --gan-reg 0.05 \
    --var-reg 0.05 \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > AttnUNet_GAN.log 2>&1 &
