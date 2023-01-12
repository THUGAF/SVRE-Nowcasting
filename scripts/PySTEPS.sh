nohup python -u pysteps_baseline.py \
    --data-path /data/gaf/SBandCRPt \
    --output-path results/PySTEPS \
    --test \
    --predict \
    --sample-indices 16840 17190 \
    --display-interval 10 \
    > results/PySTEPS.log 2>&1 &