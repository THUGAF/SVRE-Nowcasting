CUDA_VISIBLE_DEVICES=1 \
nohup python -u main.py \
        --data-path /data/gaf/SBandCRNpz \
        --output-path results/ConvLSTM_EF \
        --predict \
        --early-stopping \
        --generator EncoderForecaster \
        --hidden-channels 32 64 128 \
        --start-point 0 \
		--end-point 18016 \
		--sample-point 16060 \
        --max-iterations 50000 \
        --log-interval 10 \
        --batch-size 8 \
        --num-workers 8 \
        --num-threads 8 \
        > out_ConvLSTM_EF.log 2>&1 &
