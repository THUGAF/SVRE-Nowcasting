nohup python -u test_pysteps.py \
    --test \
    --predict \
    --data-path /data2/gaf/SBandCR_PT \
    --output-path results/PySTEPS \
    --case-indices 16840 17190 \
    --display-interval 50 \
    > results/test_pysteps.log 2>&1 &