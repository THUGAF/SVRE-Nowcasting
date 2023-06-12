import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as pcolors
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.stats import gaussian_kde
from utils.visualizer import *
from utils.taylor_diagram import TaylorDiagram


COLORS = ['tab:orange', 'tab:green', 'tab:brown', 'cyan', 'deepskyblue', 'tab:blue', 'darkblue']
MARKERS = ['o', '^', 'd', 'X', 'X', 'X', 'X']


def plot_maps(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    input_ = torch.load(os.path.join(model_dirs[0], stage, 'input', 'input.pt'))[0]
    truth = torch.load(os.path.join(model_dirs[0], stage, 'truth', 'truth.pt'))[0]
    input_ = input_[0, -1, 0].numpy()
    truth = truth[0, -1, 0].numpy()
    
    num_subplot = len(model_names) + 1
    fig = plt.figure(figsize=(num_subplot // 2 * 6, 12), dpi=300)
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

        xticks = np.arange(np.floor(STUDY_AREA[0]), np.ceil(STUDY_AREA[1]), 0.5)
        yticks = np.arange(np.floor(STUDY_AREA[2]), np.ceil(STUDY_AREA[3]), 0.5)
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
    plt.close(fig)


def plot_scatter(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    truth = torch.load(os.path.join(model_dirs[0], stage, 'truth', 'truth.pt'))[0]
    xs = truth[0, -1, 0].numpy().flatten()
    idx = np.random.choice(np.arange(len(xs)), 10000)
    
    num_subplot = len(model_names)
    fig = plt.figure(figsize=((num_subplot + 1) // 2 * 6, 12), dpi=300)
    for i in range(num_subplot):
        ax = fig.add_subplot(2, (num_subplot + 1) // 2, i + 1)
        pred = torch.load(os.path.join(model_dirs[i], stage, 'pred', 'pred.pt'))[0]
        ys = pred[0, -1, 0].numpy().flatten()
        x, y = xs[idx], ys[idx]
        data = np.vstack([x, y])
        kde = gaussian_kde(data)
        density = kde.evaluate(data)
        sc = ax.scatter(x, y, c=density, s=10, cmap='jet', norm=pcolors.Normalize(0, 0.0018))

        ax.set_title(model_names[i], fontsize=20)
        ax.set_xlabel('Observation (dBZ)', fontsize=18)
        if i == 0 or i == 4:
            ax.set_ylabel('Prediction (dBZ)', fontsize=18, labelpad=10)
        ax.set_xlim([0, 60])
        ax.set_ylim([0, 60])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.axline((0, 0), (1, 1), color='k', linewidth=1, transform=ax.transAxes)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=16)
        print('Subplot ({}, {}, {}) added'.format(2, (num_subplot + 1) // 2, i + 1))
    
    plt.rc('font', size=16)
    cax = fig.add_subplot(2, (num_subplot + 1) // 2, num_subplot + 1)
    cax.set_position([cax.get_position().x0, cax.get_position().y0, 
                      cax.get_position().width * 0.1, cax.get_position().height])
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    fmt.set_scientific(True)
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical', format=fmt)
    cbar.set_label('Probability Density', fontsize=20, labelpad=20)
    cbar.ax.tick_params(labelsize=16)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))
    plt.close(fig)


def plot_psd(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    psd_x_df = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_x.csv'.format(stage)))
    psd_y_df = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_y.csv'.format(stage)))
    wavelength_x, truth_psd_x = psd_x_df['wavelength_x'], psd_x_df['truth_psd_x']
    wavelength_y, truth_psd_y = psd_y_df['wavelength_y'], psd_y_df['truth_psd_y']
    
    fig = plt.figure(figsize=(16, 4), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(wavelength_x, truth_psd_x, color='k')
    ax2.plot(wavelength_y, truth_psd_y, color='k')
    
    legend = ['Observation']
    for i in range(len(model_names)):
        psd_x_df = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_x.csv'.format(stage)))
        psd_y_df = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_y.csv'.format(stage)))
        pred_psd_x, pred_psd_y = psd_x_df['pred_psd_x'], psd_y_df['pred_psd_y']
        ax1.plot(wavelength_x, pred_psd_x, color=COLORS[i])
        ax2.plot(wavelength_y, pred_psd_y, color=COLORS[i])
        legend.append(model_names[i])
    
    ax1.text(-0.12, 1.05, '(a)', fontsize=18, transform=ax1.transAxes)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('Wave Length (km)', fontsize=14)
    ax1.set_ylabel('Power spectral density of X axis', fontsize=14)
    ax1.legend(legend, loc='lower left', fontsize='small', edgecolor='w', fancybox=False)

    ax2.text(-0.12, 1.05, '(b)', fontsize=18, transform=ax2.transAxes)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave Length (km)', fontsize=14)
    ax2.set_ylabel('Power spectral density of Y axis', fontsize=14)
    ax2.legend(legend, loc='lower left', fontsize='small', edgecolor='w', fancybox=False)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))
    plt.close(fig)


def plot_taylor_diagram(model_names: str, model_dirs: list, stage: str, img_path: str, 
                        std_range: tuple = (0, 1), std_num: int = 6):
    fig = plt.figure(figsize=(4, 4), dpi=300)
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
    for i, model_dir in enumerate(model_dirs):
        pred = torch.load(os.path.join(model_dir, stage, 'pred', 'pred.pt'))[0]
        pred_60min = pred[0, -1, 0].numpy()
        stddev_60min = np.std(pred_60min)
        corrcoef_60min = np.corrcoef(truth_60min.flatten(), pred_60min.flatten())[0, 1]
        taylor_diagram_60min.add_sample(stddev_60min / ref_std_60min, corrcoef_60min, 
                                        color=COLORS[i], marker=MARKERS[i], label=model_names[i], 
                                        markersize=4, linestyle='')
    
    # Add a figure legend
    taylor_diagram_60min.ax.legend(taylor_diagram_60min.samplePoints,
                                   [p.get_label() for p in taylor_diagram_60min.samplePoints],
                                   numpoints=1, loc='lower center', bbox_to_anchor=(0.5, -0.4),
                                   ncols=2, fontsize='small', edgecolor='w', fancybox=False)
    
    # Add title
    fig.savefig(img_path, bbox_inches='tight')
    plt.close(fig)


def plot_psd_all(model_names, model_dirs):
    plot_psd(model_names, model_dirs, 'case_0', 'results/img/psd_case_0.jpg')
    plot_psd(model_names, model_dirs, 'case_1', 'results/img/psd_case_1.jpg')


def plot_taylor_diagram_all(model_names, model_dirs):
    plot_taylor_diagram(model_names, model_dirs, 'case_0', 'results/img/taylor_case_0.jpg')
    plot_taylor_diagram(model_names, model_dirs, 'case_1', 'results/img/taylor_case_1.jpg')


def plot_maps_all(model_names, model_dirs):
    plot_maps(model_names, model_dirs, 'case_0', 'results/img/vis_case_0.jpg')
    plot_maps(model_names, model_dirs, 'case_1', 'results/img/vis_case_1.jpg')


def plot_scatter_all(model_names, model_dirs):
    plot_scatter(model_names, model_dirs, 'case_0', 'results/img/scatter_case_0.jpg')
    plot_scatter(model_names, model_dirs, 'case_1', 'results/img/scatter_case_1.jpg')


if __name__ == '__main__':
    model_names = ['PySTEPS', 'SmaAt-UNet', 'MotionRNN',
                   'AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE']
    model_dirs = ['results/PySTEPS', 'results/SmaAt_UNet', 'results/MotionRNN', 'results/AttnUNet',
                  'results/AttnUNet_SVRE', 'results/AGAN', 'results/AGAN_SVRE']
    plot_psd_all(model_names, model_dirs)
    plot_taylor_diagram_all(model_names, model_dirs)
    plot_maps_all(model_names, model_dirs)
    plot_scatter_all(model_names, model_dirs)
