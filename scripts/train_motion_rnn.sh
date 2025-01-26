CUDA_VISIBLE_DEVICES=2 \
nohup python -u train_det.py \
    --test \
    --predict \
    --data-path /data2/gaf/SBandCR_PT \
    --output-path results/MotionRNN \
    --model MotionRNN \
    --max-iterations 100000 \
    --early-stopping \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --weight-recon 10 \
    --num-threads 4 \
    --num-workers 4 \
    --case-indices 16840 17190 \
    --display-interval 50 \
    --thresholds 20 30 40 \
    > results/train_motion_rnn.log 2>&1 &
