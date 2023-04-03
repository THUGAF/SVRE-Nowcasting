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
from utils.taylor_diagram import TaylorDiagram


def plot_maps(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    input_ = torch.load(os.path.join(model_dirs[0], stage, 'input', 'input.pt'))[0]
    truth = torch.load(os.path.join(model_dirs[0], stage, 'truth', 'truth.pt'))[0]
    input_ = input_[0, -1, 0].numpy()
    truth = truth[0, -1, 0].numpy()
    
    num_subplot = len(model_names) + 1
    fig = plt.figure(figsize=(num_subplot // 2 * 6, 12), dpi=600)
    for i in range(num_subplot):
        ax = fig.add_subplot(2, num_subplot // 2, i + 1, projection=ccrs.UTM(50))
        if i == 0:
            tensor = truth
            title = 'Observation (+60 min)'
        else:
            pred = torch.load(os.path.join(model_dirs[i - 1], stage, 'pred', 'pred.pt'))[0]
            tensor = pred[0, -1, 0].numpy()
            title = model_names[i - 1]
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES)
        ax.pcolormesh(UTM_X[X_RANGE[0]: X_RANGE[1] + 1], UTM_Y[Y_RANGE[0]: Y_RANGE[1] + 1],
                      tensor, cmap=CMAP, norm=NORM, transform=ccrs.UTM(50))

        xticks = np.arange(np.ceil(STUDY_AREA[0]), np.ceil(STUDY_AREA[1]))
        yticks = np.arange(np.ceil(STUDY_AREA[2]), np.ceil(STUDY_AREA[3]))
        gl = ax.gridlines(crs=ccrs.PlateCarree(), xlocs=xticks, ylocs=yticks, draw_labels=True,
                          linewidth=1, linestyle=':', color='k', alpha=0.8)
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
        gl.top_labels = False
        gl.right_labels = False
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=12)
        ax.set_aspect('equal')
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


def plot_taylor_diagram(model_names: str, model_dirs: list, stage: str, img_path: str, 
                        std_range: tuple = (0, 1), std_num: int = 6):
    fig = plt.figure(figsize=(4, 4), dpi=600)
    truth = torch.load(os.path.join(model_dirs[0], stage, 'truth', 'truth.pt'))[0]
    truth_60min = truth[0, -1, 0].numpy()
    ref_std_60min = np.std(truth_60min)
    taylor_diagram_60min = TaylorDiagram(ref_std_60min, fig, rect=111, 
                                         std_min=std_range[0], std_max=std_range[1],
                                         std_label_format='%.1f', num_std=std_num, 
                                         label='Observation', normalized=True)
    # Add grid
    taylor_diagram_60min.add_grid()

    # Add RMS contours, and label them
    contours_60 = taylor_diagram_60min.add_contours(colors='grey')
    plt.clabel(contours_60, inline=1, fontsize='medium', fmt='%.2f')

    # Add scatters
    markers = ['o', '^', 'p', 'd']
    for i, model_dir in enumerate(model_dirs):
        pred = torch.load(os.path.join(model_dir, stage, 'pred', 'pred.pt'))[0]
        pred_60min = pred[0, -1, 0].numpy()
        stddev_60min = np.std(pred_60min)
        corrcoef_60min = np.corrcoef(truth_60min.flatten(), pred_60min.flatten())[0, 1]
        taylor_diagram_60min.add_sample(stddev_60min / ref_std_60min, corrcoef_60min, 
                                        ms=5, ls='', marker=markers[i], label=model_names[i])
    
    # Add a figure legend
    taylor_diagram_60min.ax.legend(taylor_diagram_60min.samplePoints,
                                   [p.get_label() for p in taylor_diagram_60min.samplePoints],
                                   numpoints=1, fontsize='small', bbox_to_anchor=(1.1, 1.1))
    
    # Add title
    fig.tight_layout()
    fig.savefig(img_path)


def plot_psd_ablation(model_names, model_dirs):
    plot_psd(model_names, model_dirs, 'case_0', 'img/psd_ablation_case_0_x.jpg', 'img/psd_ablation_case_0_y.jpg')
    plot_psd(model_names, model_dirs, 'case_1', 'img/psd_ablation_case_1_x.jpg', 'img/psd_ablation_case_1_y.jpg')


def plot_psd_comparison(model_names, model_dirs):
    plot_psd(model_names, model_dirs, 'case_0', 'img/psd_comparison_case_0_x.jpg', 'img/psd_comparison_case_0_y.jpg')
    plot_psd(model_names, model_dirs, 'case_1', 'img/psd_comparison_case_1_x.jpg', 'img/psd_comparison_case_1_y.jpg')


def plot_taylor_diagram_ablation(model_names, model_dirs):
    plot_taylor_diagram(model_names, model_dirs, 'case_0', 'img/taylor_ablation_case_0.jpg')
    plot_taylor_diagram(model_names, model_dirs, 'case_1', 'img/taylor_ablation_case_1.jpg')


def plot_taylor_diagram_comparison(model_names, model_dirs):
    plot_taylor_diagram(model_names, model_dirs, 'case_0', 'img/taylor_comparison_case_0.jpg')
    plot_taylor_diagram(model_names, model_dirs, 'case_1', 'img/taylor_comparison_case_1.jpg')


def plot_maps_all(model_names, model_dirs):
    plot_maps(model_names, model_dirs, 'case_0', 'img/vis_case_0.jpg')
    plot_maps(model_names, model_dirs, 'case_1', 'img/vis_case_1.jpg')


if __name__ == '__main__':
    plot_psd_ablation(['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'], 
                      ['results/AttnUNet', 'results/AttnUNet_SVRE', 'results/AGAN', 'results/AGAN_SVRE'])
    plot_psd_comparison(['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AGAN+SVRE'], 
                        ['results/PySTEPS', 'results/SmaAt_UNet', 'results/MotionRNN', 'results/AGAN_SVRE'])
    plot_taylor_diagram_ablation(['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'], 
                                 ['results/AttnUNet', 'results/AttnUNet_SVRE', 'results/AGAN', 'results/AGAN_SVRE'])
    plot_taylor_diagram_comparison(['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AGAN+SVRE'], 
                                   ['results/PySTEPS', 'results/SmaAt_UNet', 'results/MotionRNN', 'results/AGAN_SVRE'])
    plot_maps_all(['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'], 
                  ['results/PySTEPS', 'results/SmaAt_UNet', 'results/MotionRNN', 'results/AttnUNet', 
                   'results/AttnUNet_SVRE', 'results/AGAN', 'results/AGAN_SVRE'])
