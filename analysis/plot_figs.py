import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from utils.visualizer import *


plt.rcParams['font.sans-serif'] = 'Arial'

def plot_maps(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    input_ = torch.load(os.path.join(model_dirs[0], stage, 'input', 'input.pt'))
    truth = torch.load(os.path.join(model_dirs[0], stage, 'truth', 'truth.pt'))
    input_ = np.flip(input_[0, -1, 0].numpy(), axis=0)
    truth = np.flip(truth[0, -1, 0].numpy(), axis=0)
    
    num_subplot = len(model_names) + 1
    fig = plt.figure(figsize=(num_subplot // 2 * 6, 12), dpi=600)
    for i in range(num_subplot):
        ax = fig.add_subplot(2, num_subplot // 2, i + 1, projection=ccrs.Mercator())
        if i == 0:
            tensor = truth
            title = 'Observation (+60 min)'
        else:
            pred = torch.load(os.path.join(model_dirs[i - 1], stage, 'pred', 'pred.pt'))
            tensor = np.flip(pred[0, -1, 0].numpy(), axis=0)
            title = model_names[i - 1]
        ax.set_extent(AREA, crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES)
        ax.imshow(tensor, cmap=CMAP, norm=NORM, extent=AREA, transform=ccrs.PlateCarree())

        xticks = np.arange(np.ceil(2 * AREA[0]) / 2, np.ceil(2 * AREA[1]) / 2, 0.5)
        yticks = np.arange(np.ceil(2 * AREA[2]) / 2, np.ceil(2 * AREA[3]) / 2, 0.5)
        ax.set_xticks(np.arange(np.ceil(AREA[0]), np.ceil(AREA[1]), 1), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(np.ceil(AREA[2]), np.ceil(AREA[3]), 1), crs=ccrs.PlateCarree())
        ax.gridlines(crs=ccrs.PlateCarree(), xlocs=xticks, ylocs=yticks, draw_labels=False, 
                    linewidth=1, linestyle=':', color='k', alpha=0.8)

        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=18)
        ax.set_title(title, fontsize=22)
    
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.94, 0.14, 0.012, 0.72])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax, orientation='vertical')
    cbar.set_label('dBZ', fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


def plot_psd(model_names, model_dirs, stage, img_path_1, img_path_2):
    print('Plotting {} ...'.format(img_path_1))
    print('Plotting {} ...'.format(img_path_2))
    psd_x_df = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_x.csv'.format(stage)))
    psd_y_df = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_y.csv'.format(stage)))
    wavelength_x, truth_psd_x = psd_x_df['wavelength_x'], psd_x_df['truth_psd_x']
    wavelength_y, truth_psd_y = psd_y_df['wavelength_y'], psd_y_df['truth_psd_y']
    
    fig1 = plt.figure(figsize=(8, 4), dpi=600)
    fig2 = plt.figure(figsize=(8, 4), dpi=600)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)

    ax1.plot(wavelength_x, truth_psd_x, color='k')
    ax2.plot(wavelength_y, truth_psd_y, color='k')
    
    legend = ['Observation']
    for i in range(len(model_names)):
        psd_x_df = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_x.csv'.format(stage)))
        psd_y_df = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_y.csv'.format(stage)))
        pred_psd_x, pred_psd_y = psd_x_df['pred_psd_x'], psd_y_df['pred_psd_y']
        ax1.plot(wavelength_x, pred_psd_x)
        ax2.plot(wavelength_y, pred_psd_y)
        legend.append(model_names[i])
    
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('Wave Length (km)', fontsize=14)
    ax1.set_ylabel('Power spectral density of X axis', fontsize=14)
    ax1.legend(legend)

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave Length (km)', fontsize=14)
    ax2.set_ylabel('Power spectral density of Y axis', fontsize=14)
    ax2.legend(legend)

    fig1.savefig(img_path_1, bbox_inches='tight')
    fig2.savefig(img_path_2, bbox_inches='tight')
    print('{} saved'.format(img_path_1))
    print('{} saved'.format(img_path_2))


def plot_psd_ablation(model_names, model_dirs):
    plot_psd(model_names, model_dirs, 'case_0', 'img/psd_ablation_case_0_x.jpg', 'img/psd_ablation_case_0_y.jpg')
    plot_psd(model_names, model_dirs, 'case_1', 'img/psd_ablation_case_1_x.jpg', 'img/psd_ablation_case_1_y.jpg')


def plot_psd_comparison(model_names, model_dirs):
    plot_psd(model_names, model_dirs, 'case_0', 'img/psd_comparison_case_0_x.jpg', 'img/psd_comparison_case_0_y.jpg')
    plot_psd(model_names, model_dirs, 'case_1', 'img/psd_comparison_case_1_x.jpg', 'img/psd_comparison_case_1_y.jpg')


def plot_all(model_names, model_dirs):
    plot_maps(model_names, model_dirs, 'case_0', 'img/vis_case_0.jpg')
    plot_maps(model_names, model_dirs, 'case_1', 'img/vis_case_1.jpg')


if __name__ == '__main__':
    plot_psd_ablation(['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'], ['results/AttnUNet',
                      'results/AttnUNet_SVRE', 'results/AGAN', 'results/AGAN_SVRE'])
    plot_psd_comparison(['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AGAN+SVRE'], ['results/PySTEPS',
                        'results/SmaAt_UNet', 'results/MotionRNN', 'results/AGAN_SVRE'])
    plot_all(['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'], 
             ['results/PySTEPS', 'results/SmaAt_UNet', 'results/MotionRNN', 'results/AttnUNet',
              'results/AttnUNet_SVRE', 'results/AGAN', 'results/AGAN_SVRE'])
