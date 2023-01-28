CUDA_VISIBLE_DEVICES=3 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRPt \
    --output-path results/MotionRNN \
    --model MotionRNN \
    --test \
    --predict \
    --train-ratio 0.7 \
    --valid-ratio 0.1 \
    --sample-indices 16840 17190 \
    --max-iterations 80000 \
    --early-stopping \
    --batch-size 1 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 10 \
    > results/MotionRNN.log 2>&1 &
