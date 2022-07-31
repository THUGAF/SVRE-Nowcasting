CUDA_VISIBLE_DEVICES=2 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/AttnUNet_GAN_CV \
    --model AttnUNet \
    --add-gan \
    --train \
    --test \
    --predict \
    --ensemble-members 3 \
    --sample-index 16840 \
    --max-iterations 100000 \
    --early-stopping \
    --gan-reg 0.15 \
    --var-reg 0.1 \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > AttnUNet_GAN_CV.log 2>&1 &
