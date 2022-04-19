CUDA_VISIBLE_DEVICES=1 \
nohup python -u main.py \
        --data-path /data/gaf/SBandCRNpz \
        --output-path results/AttnUNet_CV \
        --train \
        --test \
        --predict \
        --generator AttnUNet \
        --start-point 0 \
	--end-point 18016 \
	--sample-point 16060 \
        --max-iterations 50000 \
        --global-var-reg 0.1 \
        --log-interval 10 \
        --batch-size 8 \
        --num-workers 4 \
        --num-threads 4 \
        > out_AttnUNet_CV.log 2>&1 &
