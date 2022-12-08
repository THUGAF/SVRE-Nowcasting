nohup python -u pysteps_baseline.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/PySTEPS \
    --sample-indices 16840 17190 \
    > results/PySTEPS.log 2>&1 &