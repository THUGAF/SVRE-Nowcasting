nohup python -u test_pysteps.py \
    --test \
    --predict \
    --data-path /data/gaf/SBandCRPt \
    --output-path results/PySTEPS \
    --case-indices 16840 17190 \
    --display-interval 50 \
    > results/test_pysteps.log 2>&1 &