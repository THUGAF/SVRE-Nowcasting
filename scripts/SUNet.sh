CUDA_VISIBLE_DEVICES=1 \
nohup python -u main.py \
        --data-path /data/gaf/SBandCRNpz \
        --output-path results/SUNet \
        --predict \
        --early-stopping \
        --generator SmaAt_UNet \
        --start-point 0 \
	--end-point 18016 \
	--sample-point 16060 \
        --max-iterations 50000 \
        --log-interval 10 \
        --batch-size 8 \
        --num-workers 8 \
        --num-threads 8 \
        > out_SUNet.log 2>&1 &
