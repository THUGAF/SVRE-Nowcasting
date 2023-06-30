# Enhancing Spatial Variability Representation of Radar Nowcasting with Generative Adversarial Networks

This repo contains a PyTorch implementation of the **Spatial Variability Representation Enhancement (SVRE)** loss function and the **Attentional Generative Adversarial Network (AGAN)** for improving radar nowcasting.

## Dependencies

Since the codes are based on Python, you need to install Python 3.8 first. The following dependencies are also needed.

```pytorch=1.11.0
numpy=1.20.3
netcdf4=1.5.7
pandas=1.4.3
matplotlib=3.5.1
cartopy=0.20.3
pyproj=3.3.1
pysteps=1.4.1
```

## Usage

Run the bash scripts to train the model with the radar dataset.

* Ablation experiments for SVRE and AGAN

```cd
sh train_attn_unet.sh
sh train_attn_unet_svre.sh
sh train_agan.sh
sh train_agan_svre.sh
```

* Comparison experiments for baseline models

```cd
sh test_pysteps.sh
sh train_motion_rnn.sh
sh train_smaat_unet.sh
```

## Citation

If you find this repo helpful, please cite the following article.

> Gong, A.; Li, R.; Pan, B.; Chen, H.; Ni, G.; Chen, M. Enhancing Spatial Variability Representation of Radar Nowcasting with Generative Adversarial Networks. Remote Sens. 2023, 15, 3306. [https://doi.org/10.3390/rs15133306](https://doi.org/10.3390/rs15133306)
