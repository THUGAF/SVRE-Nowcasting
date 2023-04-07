import os
import datetime
import torch
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import matplotlib.cm as cm
import scipy.signal
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


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
X_RANGE = [272, 528]
Y_RANGE = [336, 592]
CMAP = pcolors.ListedColormap(['#ffffff', '#2aedef', '#1caff4', '#0a22f4', '#29fd2f',
                               '#1ec722', '#139116', '#fffd38', '#e7bf2a', '#fb9124',
                               '#f90f1c', '#d00b15', '#bd0713', '#da66fb', '#bb24eb'])
NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), CMAP.N)
plt.rcParams['font.sans-serif'] = 'Arial'


def save_tensor(tensor: torch.Tensor, timestamp: torch.Tensor, root: str, stage: str, name: str):
    if not os.path.exists(os.path.join(root, stage)):
        os.mkdir(os.path.join(root, stage))
    path = os.path.join(root, stage, name)
    if not os.path.exists(path):
        os.mkdir(path)
    tensor = tensor.detach().cpu()
    torch.save((tensor, timestamp), '{}/{}.pt'.format(path, name))


def plot_loss(train_loss: list, val_loss: list, filename: str):
    print('Plotting loss...')
    fig = plt.figure(figsize=(8, 4), dpi=300)
    ax = plt.subplot(111)
    ax.plot(range(1, len(train_loss) + 1), train_loss, 'b')
    ax.plot(range(1, len(val_loss) + 1), val_loss, 'r')
    ax.set_xlabel('epoch')
    ax.legend(['train loss', 'val loss'])
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_figs(tensor: torch.Tensor, timestamp: torch.Tensor, root: str, stage: str, name: str):
    if not os.path.exists(os.path.join(root, stage)):
        os.mkdir(os.path.join(root, stage))
    path = os.path.join(root, stage, name)
    if not os.path.exists(path):
        os.mkdir(path)
    tensor = tensor.detach().cpu().numpy()
    tensor = tensor[0, :, 0]
    seq_len = tensor.shape[0]

    # plot long fig
    num_rows = 2 if seq_len > 10 else 1
    num_cols = seq_len // num_rows
    fig = plt.figure(figsize=(num_cols, num_rows), dpi=300)
    for i in range(seq_len):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.pcolormesh(tensor[i], cmap=CMAP, norm=NORM)
        ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    fig.savefig('{}/{}.png'.format(path, name))
    plt.close(fig)
    print('{}.png saved'.format(name))

    # plot figs
    image_list = []
    for i in range(seq_len):
        # minus represents the time before current moment
        if 'input' in name:
            str_min = str(6 * (i - seq_len + 1))
        else:
            str_min = str(6 * (i + 1))
        file_path = '{}/{}_{}.png'.format(path, name, str_min)
        time_str = datetime.datetime.utcfromtimestamp(int(timestamp[0, i]))
        time_str = time_str.strftime('%Y-%m-%d %H:%M:%S')
        plot_single_fig(tensor[i], file_path, time_str, CMAP, NORM)
        print('{}_{}.png saved'.format(name, str_min))
        image_list.append(imageio.imread(file_path))

    # make gif
    imageio.mimsave('{}/{}.gif'.format(path, name), image_list, 'GIF', duration=0.2)
    print('{}.gif saved'.format(name))


def plot_single_fig(tensor: torch.Tensor, file_path: str, time_str: str, 
                    cmap: pcolors.ListedColormap, norm: pcolors.BoundaryNorm):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.subplot(1, 1, 1, projection=ccrs.UTM(50))
    ax.set_title(time_str, fontsize=14, loc='right')
    ax.set_title('Composite Reflectivity', fontsize=14, loc='left')
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.pcolormesh(UTM_X[X_RANGE[0]: X_RANGE[1] + 1], UTM_Y[Y_RANGE[0]: Y_RANGE[1] + 1],
                  tensor, cmap=cmap, norm=norm, transform=ccrs.UTM(50))

    xticks = np.arange(np.floor(STUDY_AREA[0]), np.ceil(STUDY_AREA[1]), 0.5)
    yticks = np.arange(np.floor(STUDY_AREA[2]), np.ceil(STUDY_AREA[3]), 0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), xlocs=xticks, ylocs=yticks, draw_labels=True,
                      linewidth=1, linestyle=':', color='k', alpha=0.8)
    gl.top_labels = False
    gl.right_labels = False
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=12)
    ax.set_aspect('equal')

    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.7, aspect=30)
    cbar.set_label('dBZ', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(file_path, bbox_inches='tight')
    plt.close(fig)


def plot_psd(pred: torch.Tensor, truth: torch.Tensor, root: str, stage: str):
    print('Plotting PSD...')
    pred, truth = pred[0, -1, 0].detach().cpu(), truth[0, -1, 0].cpu()
    len_y, len_x = pred.size(0), pred.size(1)
    xx, yy = np.arange(len_x), np.arange(len_y)
    xx, yy = np.meshgrid(xx, yy)
    
    freq_x, pred_psd_x = scipy.signal.welch(pred, nperseg=pred.shape[1], axis=1)
    freq_y, pred_psd_y = scipy.signal.welch(pred, nperseg=pred.shape[0], axis=0)
    _, truth_psd_x = scipy.signal.welch(truth, nperseg=truth.shape[1], axis=1)
    _, truth_psd_y = scipy.signal.welch(truth, nperseg=truth.shape[0], axis=0)
    pred_psd_x, truth_psd_x = np.mean(pred_psd_x, axis=0)[1:], np.mean(truth_psd_x, axis=0)[1:]
    pred_psd_y, truth_psd_y = np.mean(pred_psd_y, axis=1)[1:], np.mean(truth_psd_y, axis=1)[1:]
    wavelength_x = 1 / freq_x[1:]
    wavelength_y = 1 / freq_y[1:]

    psd_x_data = {
        'wavelength_x': wavelength_x,
        'pred_psd_x': pred_psd_x,
        'truth_psd_x': truth_psd_x,
    }
    psd_y_data = {
        'wavelength_y': wavelength_y,
        'pred_psd_y': pred_psd_y,
        'truth_psd_y': truth_psd_y
    }
    psd_x_df = pd.DataFrame(psd_x_data)
    psd_y_df = pd.DataFrame(psd_y_data)
    psd_x_df.to_csv('{}/{}_psd_x.csv'.format(root, stage), float_format='%.6f', index=False)
    psd_y_df.to_csv('{}/{}_psd_y.csv'.format(root, stage), float_format='%.6f', index=False)

    fig = plt.figure(figsize=(12, 12), dpi=300)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(wavelength_x, pred_psd_x, color='b')
    ax1.plot(wavelength_x, truth_psd_x, color='r')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('Wave length (km)', fontsize=12)
    ax1.set_ylabel('Power Spectral Density of X axis', fontsize=12)
    ax1.legend(['Prediction', 'Observation'])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(wavelength_y, pred_psd_y, color='b')
    ax2.plot(wavelength_y, truth_psd_y, color='r')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave length (km)', fontsize=12)
    ax2.set_ylabel('Power Spectral Density of Y axis', fontsize=12)
    ax2.legend(['Prediction', 'Observation'])

    fig.savefig('{}/{}_psd.png'.format(root, stage), bbox_inches='tight')
    plt.close(fig)
    print('PSD saved')
