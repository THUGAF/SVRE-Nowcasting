CUDA_VISIBLE_DEVICES=1 \
nohup python -u main.py \
        --data-path /data/gaf/SBandCRNpz \
        --output-path results/AttnUNet \
        --predict \
        --generator AttnUNet \
        --start-point 0 \
	--end-point 18016 \
	--sample-point 16060 \
        --max-iterations 50000 \
        --log-interval 10 \
        --batch-size 8 \
        --num-workers 4 \
        --num-threads 4 \
        > out_AttnUNet.log 2>&1 &
