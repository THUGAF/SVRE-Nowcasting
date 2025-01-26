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
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.stats import gaussian_kde
from utils.taylor_diagram import TaylorDiagram
from utils.transform import ref_to_R


# Coordinate transformation
TRANS_LONLAT_TO_UTM = pyproj.Transformer.from_crs('epsg:4326', 'epsg:32650')
TRANS_UTM_TO_LONLAT = pyproj.Transformer.from_crs('epsg:32650', 'epsg:4326')

# Global information
CENTER_LON, CENTER_LAT = 116.47195, 39.808887
CENTER_UTM_X, CENTER_UTM_Y = TRANS_LONLAT_TO_UTM.transform(CENTER_LAT, CENTER_LON)
LEFT_BOTTOM_LAT, LEFT_BOTTOM_LON = TRANS_UTM_TO_LONLAT.transform(CENTER_UTM_X - 128000, CENTER_UTM_Y - 64000)
RIGHT_TOP_LAT, RIGHT_TOP_LON = TRANS_UTM_TO_LONLAT.transform(CENTER_UTM_X + 128000, CENTER_UTM_Y + 192000)
STUDY_AREA = [LEFT_BOTTOM_LON, RIGHT_TOP_LON, LEFT_BOTTOM_LAT, RIGHT_TOP_LAT]
UTM_X = CENTER_UTM_X + np.arange(-400000, 401000, 1000)
UTM_Y = CENTER_UTM_Y + np.arange(-400000, 401000, 1000)
X_RANGE, Y_RANGE = [272, 528], [336, 592]
CMAP = pcolors.ListedColormap(['#ffffff', '#2aedef', '#0a22f4', '#29fd2f', '#139116',
                               '#fffd38', '#fb9124', '#f90f1c', '#bd0713', '#da66fb'])
NORM = pcolors.BoundaryNorm([0, 0.1, 1, 2, 3, 5, 10, 20, 30, 50], CMAP.N, extend='max')
COLORS = ['tab:orange', 'tab:green', 'tab:brown', 'cyan', 'deepskyblue', 'tab:blue', 'darkblue']
MARKERS = ['o', '^', 'd', 'X', 'X', 'X', 'X']

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'               # 用于正常数学文本
plt.rcParams['mathtext.it'] = 'Arial:italic'        # 用于斜体数学文本


def plot_bar(model_names, stage, img_path):
    df = pd.read_excel('results/metrics.xlsx', sheet_name=stage, index_col=0)
    print(df)
    
    fig = plt.figure(figsize=(7, 9), dpi=300)
    num_models = len(model_names)
    width = 1 / (num_models + 1)
    
    ax1 = fig.add_subplot(3, 1, 1)
    labels = df.columns.values[:3]
    metrics = df.values[:, :3]
    l = np.arange(len(labels))
    bars = []
    for m in range(num_models):
        b = ax1.bar((l + width * (m - (num_models - 1) / 2)), metrics[m], width,
                    label=model_names[m], color=COLORS[m], linewidth=0)
        ax1.bar_label(b, fmt='%.3f', padding=1, fontsize=8, rotation=90)
        bars.append(b)
    ax1.set_xticks(l, labels=labels)
    ax1.set_ylim(0, 1)
    ax1.tick_params(labelsize=10)
    ax1.legend(bars, model_names, edgecolor='w', fancybox=False, fontsize=8, ncols=3)
    print('Subplot (3, 1, 1) added')
    
    ax2 = fig.add_subplot(3, 1, 2)
    labels = df.columns.values[3:6]
    metrics = df.values[:, 3:6]
    l = np.arange(len(labels))
    bars = []
    for m in range(num_models):
        b = ax2.bar((l + width * (m - (num_models - 1) / 2)), metrics[m], width,
                    label=model_names[m], color=COLORS[m], linewidth=0)
        ax2.bar_label(b, fmt='%.3f', padding=1, fontsize=8, rotation=90)
        bars.append(b)
    ax2.set_xticks(l, labels=labels)
    ax2.set_ylim(0, np.ceil(np.max(metrics)))
    ax2.tick_params(labelsize=10)
    ax2.legend(bars, model_names, edgecolor='w', fancybox=False, fontsize=8, ncols=3)
    print('Subplot (3, 1, 2) added')
    
    ax3 = fig.add_subplot(3, 1, 3)
    labels = df.columns.values[6:8]
    metrics = df.values[:, 6:8]
    l = np.arange(len(labels))
    bars = []
    for m in range(num_models):
        b = ax3.bar((l + width * (m - (num_models - 1) / 2)), metrics[m], width,
                    label=model_names[m], color=COLORS[m], linewidth=0)
        ax3.bar_label(b, fmt='%.3f', padding=1, fontsize=8, rotation=90)
        bars.append(b)
    ax3.set_xticks(l, labels=labels)
    ax3.set_ylim(0, 1)
    ax3.tick_params(labelsize=10)
    ax3.legend(bars, model_names, edgecolor='w', fancybox=False, fontsize=8, ncols=3)
    print('Subplot (3, 1, 3) added')
    
    fig.savefig(img_path, bbox_inches='tight')
    print('{}'.format(img_path))
    plt.close(fig)
    

def plot_map(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    input_ = torch.load(os.path.join(model_dirs[0], stage, 'input', 'input.pt'))[0]
    truth = torch.load(os.path.join(model_dirs[0], stage, 'truth', 'truth.pt'))[0]
    input_R, truth_R = ref_to_R(input_), ref_to_R(truth)
    input_R, truth_R = input_R[0, -1, 0].numpy(), truth_R[0, -1, 0].numpy()
    num_subplot = len(model_names) + 1
    num_row = 2
    num_col = num_subplot // num_row
    fig = plt.figure(figsize=(num_col* 6, num_row * 6), dpi=300)
    for n in range(num_subplot):
        ax = fig.add_subplot(num_row, num_col, n + 1, projection=ccrs.UTM(50))
        if n == 0:
            tensor = truth_R
            title = 'OBS'
        else:
            pred = torch.load(os.path.join(model_dirs[n - 1], stage, 'pred', 'pred.pt'))[0]
            pred_R = ref_to_R(pred)
            pred_R = pred_R[0, -1, 0].numpy()
            tensor = pred_R
            title = model_names[n - 1]
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES)
        ax.pcolorfast(UTM_X[X_RANGE[0]: X_RANGE[1] + 1], UTM_Y[Y_RANGE[0]: Y_RANGE[1] + 1],
                      tensor, cmap=CMAP, norm=NORM, transform=ccrs.UTM(50))

        xticks = np.arange(np.floor(STUDY_AREA[0]), np.ceil(STUDY_AREA[1]), 0.5)
        yticks = np.arange(np.floor(STUDY_AREA[2]), np.ceil(STUDY_AREA[3]), 0.5)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), xlocs=xticks, ylocs=yticks, draw_labels=True,
                          linewidth=1, linestyle=':', color='k', alpha=0.8)
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        if n % num_col < num_col - 1:
            gl.right_labels = False
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=12)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=20, pad=10)
    
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.94, 0.14, 0.012, 0.72])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax, orientation='vertical')
    cbar.set_label('降水强度 (mm/h)', fontsize=20, fontfamily='SimHei')
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(img_path, bbox_inches='tight')
    print('{}'.format(img_path))
    plt.close(fig)


def plot_scatter(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    truth = torch.load(os.path.join(model_dirs[0], stage, 'truth', 'truth.pt'))[0]
    truth_R = ref_to_R(truth)
    xs = truth_R[0, -1, 0].numpy().flatten()
    idx = np.random.choice(np.arange(len(xs)), 10000)
    
    num_subplot = len(model_names)
    num_col = 4
    fig = plt.figure(figsize=(num_col * 6, (num_subplot // num_col + 1) * 6), dpi=300)
    for n in range(num_subplot):
        ax = fig.add_subplot(2, (num_subplot + 1) // 2, n + 1)
        pred = torch.load(os.path.join(model_dirs[n], stage, 'pred', 'pred.pt'))[0]
        pred_R = ref_to_R(pred)
        ys = pred_R[0, -1, 0].numpy().flatten()
        x, y = xs[idx], ys[idx]
        data = np.vstack([x, y])
        kde = gaussian_kde(data)
        density = kde.evaluate(data)
        sc = ax.scatter(x, y, c=density, s=10, cmap='jet', norm=pcolors.Normalize(0, 0.01))
        ax.set_title(model_names[n], fontsize=20)
        if n >= num_subplot - num_col:
            ax.set_xlabel('观测值 (mm/h)', fontsize=18, fontfamily='SimHei')
        if n % num_col == 0:
            ax.set_ylabel('预报值 (mm/h)', fontsize=18, labelpad=10, fontfamily='SimHei')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.axline((0, 0), (1, 1), color='k', linewidth=1, transform=ax.transAxes)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=16)
        print('Subplot ({}, {}, {}) added'.format(2, (num_subplot + 1) // 2, n + 1))
    
    plt.rc('font', size=16)
    cax = fig.add_subplot(2, (num_subplot + 1) // 2, num_subplot + 1)
    cax.set_position([cax.get_position().x0, cax.get_position().y0, 
                      cax.get_position().width * 0.1, cax.get_position().height])
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    fmt.set_scientific(True)
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical', format=fmt)
    cbar.set_label('概率密度', fontsize=20, labelpad=20, fontfamily='SimHei')
    cbar.ax.tick_params(labelsize=16)

    fig.savefig(img_path, bbox_inches='tight')
    print('{}'.format(img_path))
    plt.close(fig)


def plot_psd(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    psd_x_df = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_x.csv'.format(stage)))
    psd_y_df = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_y.csv'.format(stage)))
    wavelength_x, truth_psd_x = psd_x_df['wavelength_x'], psd_x_df['truth_psd_x']
    wavelength_y, truth_psd_y = psd_y_df['wavelength_y'], psd_y_df['truth_psd_y']
    
    fig = plt.figure(figsize=(14, 4), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(wavelength_x, truth_psd_x, color='k')
    ax2.plot(wavelength_y, truth_psd_y, color='k')
    
    legend = ['OBS']
    for i in range(len(model_names)):
        psd_x_df = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_x.csv'.format(stage)))
        psd_y_df = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_y.csv'.format(stage)))
        pred_psd_x, pred_psd_y = psd_x_df['pred_psd_x'], psd_y_df['pred_psd_y']
        ax1.plot(wavelength_x, pred_psd_x, color=COLORS[i])
        ax2.plot(wavelength_y, pred_psd_y, color=COLORS[i])
        legend.append(model_names[i])
    
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('波长 (km)', fontsize=14, fontfamily='SimHei')
    ax1.set_ylabel('X方向功率谱密度', fontsize=14, fontfamily='SimHei')
    ax1.legend(legend, loc='lower left', fontsize=10, edgecolor='w', fancybox=False)
    ax1.tick_params(labelsize=10)

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('波长 (km)', fontsize=14, fontfamily='SimHei')
    ax2.set_ylabel('Y方向功率谱密度', fontsize=14, fontfamily='SimHei')
    ax2.legend(legend, loc='lower left', fontsize=10, edgecolor='w', fancybox=False)
    ax2.tick_params(labelsize=10)

    fig.savefig(img_path, bbox_inches='tight')
    print('{}'.format(img_path))
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
                                         label='OBS', normalized=True)
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
                                   ncol=2, fontsize='small', edgecolor='w', fancybox=False)
    
    # Add title
    fig.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    


if __name__ == '__main__':
    model_names = ['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AN+L1', 'AN+SVRE', 'AGAN+L1', 'AGAN+SVRE']
    model_dirs = ['results/PySTEPS', 'results/SmaAt_UNet', 'results/MotionRNN', 'results/AN', 
                  'results/AN_SVRE', 'results/AGAN', 'results/AGAN_SVRE']
    plot_bar(model_names, 'test', 'results/img_cn/bar_test.jpg')
    for i in range(2):
        plot_bar(model_names, 'case_{}'.format(i), 'results/img_cn/bar_case_{}.jpg'.format(i))
        plot_map(model_names, model_dirs, 'case_{}'.format(i), 'results/img_cn/vis_case_{}.jpg'.format(i))
        plot_scatter(model_names, model_dirs, 'case_{}'.format(i), 'results/img_cn/scatter_case_{}.jpg'.format(i))
        plot_taylor_diagram(model_names, model_dirs, 'case_{}'.format(i), 'results/img_cn/taylor_case_{}.jpg'.format(i))
        plot_psd(model_names, model_dirs, 'case_{}'.format(i), 'results/img_cn/psd_case_{}.jpg'.format(i))
