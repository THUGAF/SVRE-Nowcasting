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


plt.rcParams['font.sans-serif'] = 'Arial'

# Coordinate transformation
TRANS_WGS84_TO_UTM = pyproj.Transformer.from_crs('epsg:4326', 'epsg:32650')
TRANS_UTM_TO_WGS84 = pyproj.Transformer.from_crs('epsg:32650', 'epsg:4326')

# Global information
CENTER_LON, CENTER_LAT = 116.47195, 39.808887
CENTER_UTM_X, CENTER_UTM_Y = TRANS_WGS84_TO_UTM.transform(CENTER_LAT, CENTER_LON)
LEFT_BOTTOM_LAT, LEFT_BOTTOM_LON = TRANS_UTM_TO_WGS84.transform(CENTER_UTM_X - 128000, CENTER_UTM_Y - 64000)
RIGHT_TOP_LAT, RIGHT_TOP_LON = TRANS_UTM_TO_WGS84.transform(CENTER_UTM_X + 128000, CENTER_UTM_Y + 192000)
AREA = [LEFT_BOTTOM_LON, RIGHT_TOP_LON, LEFT_BOTTOM_LAT, RIGHT_TOP_LAT]

CMAP = pcolors.ListedColormap(['#ffffff', '#2aedef', '#1caff4', '#0a22f4', '#29fd2f',
                               '#1ec722', '#139116', '#fffd38', '#e7bf2a', '#fb9124',
                               '#f90f1c', '#d00b15', '#bd0713', '#da66fb', '#bb24eb'])
NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), CMAP.N)


def plot_loss(train_loss: list, val_loss: list, output_path: str, filename: str = 'loss.png') -> None:
    print('Plotting loss...')
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.subplot(111)
    ax.plot(range(1, len(train_loss) + 1), train_loss, 'b')
    ax.plot(range(1, len(val_loss) + 1), val_loss, 'r')
    ax.set_xlabel('epoch')
    ax.legend(['train loss', 'val loss'])
    fig.savefig(os.path.join(output_path, filename), bbox_inches='tight')


def plot_map(input_: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor, timestamp: torch.Tensor, 
             root: str, stage: str) -> None:
    print('Plotting maps...')
    if not os.path.exists(os.path.join(root, stage)):
        os.mkdir(os.path.join(root, stage))
    input_steps, forecast_steps = input_.size(1), pred.size(1)
    plot_figs(input_, root, timestamp[:, :input_steps], stage, type_='input',
              cmap=CMAP, norm=NORM)
    plot_figs(pred, root, timestamp[:, input_steps: input_steps + forecast_steps], 
              stage, type_='pred', cmap=CMAP, norm=NORM)
    plot_figs(truth, root, timestamp[:, input_steps: input_steps + forecast_steps],
              stage, type_='truth', cmap=CMAP, norm=NORM)


def plot_figs(tensor: torch.Tensor, root: str, timestamp: torch.Tensor, stage: str, type_: str, 
              cmap: pcolors.ListedColormap, norm: pcolors.BoundaryNorm) -> None:
    path = os.path.join(root, stage, type_)
    if not os.path.exists(path):
        os.mkdir(path)

    # save tensor
    tensor = tensor.detach().cpu()
    torch.save(tensor, '{}/{}.pt'.format(path, type_))

    seq_len = tensor.size(1)
    image_list = []
    for i in range(seq_len):
        # minus represents the time before current moment
        if type_ == 'input':
            str_min = str(6 * (i - seq_len + 1))
        else:
            str_min = str(6 * (i + 1))
        file_path = '{}/{}.png'.format(path, str_min)
        current_datetime = datetime.datetime.utcfromtimestamp(int(timestamp[0, i]))
        plot_single_fig(tensor[0, i, 0], file_path, current_datetime, cmap, norm)
        image_list.append(imageio.imread(file_path))

    # plot the long image
    num_rows = 2 if seq_len > 10 else 1
    num_cols = seq_len // num_rows
    fig = plt.figure(figsize=(num_cols, num_rows), dpi=300)
    for i in range(seq_len):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(np.flip(tensor[0, i, 0].numpy(), axis=0), cmap=cmap, norm=norm)
        ax.axis('off')
    
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    fig.savefig('{}/{}.png'.format(path, type_))
    
    # make gif
    imageio.mimsave('{}/{}.gif'.format(path, type_), image_list, 'GIF', duration=0.2)
    print('{} saved'.format(type_))


def plot_single_fig(tensor_slice: torch.Tensor, file_path: str, current_datetime: datetime.datetime, 
                    cmap: pcolors.ListedColormap, norm: pcolors.BoundaryNorm) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=300)
    fig.suptitle('\n' + current_datetime.strftime('%Y-%m-%d %H:%M:%S'), fontsize=24)
    ax = plt.subplot(111, projection=ccrs.Mercator())
    ax.set_title('CR', fontsize=18)
    ax.set_extent(AREA, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)

    tensor_slice = np.flip(tensor_slice.numpy(), axis=0)
    ax.imshow(tensor_slice, cmap=cmap, norm=norm, extent=AREA, transform=ccrs.PlateCarree())

    xticks = np.arange(np.ceil(2 * AREA[0]) / 2, np.ceil(2 * AREA[1]) / 2, 0.5)
    yticks = np.arange(np.ceil(2 * AREA[2]) / 2, np.ceil(2 * AREA[3]) / 2, 0.5)
    ax.set_xticks(np.arange(np.ceil(AREA[0]), np.ceil(AREA[1]), 1), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.ceil(AREA[2]), np.ceil(AREA[3]), 1), crs=ccrs.PlateCarree())
    ax.gridlines(crs=ccrs.PlateCarree(), xlocs=xticks, ylocs=yticks, draw_labels=False, 
                 linewidth=1, linestyle=':', color='k', alpha=0.8)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=18)

    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), pad=0.05, shrink=0.7, aspect=20, 
                        orientation='vertical', extend='both')
    cbar.set_label('dBZ', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    fig.subplots_adjust(left=0.1, right=1, bottom=0, top=1)
    fig.savefig(file_path)


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

    fig = plt.figure(figsize=(14, 6), dpi=600)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(wavelength_x, pred_psd_x, color='b')
    ax1.plot(wavelength_x, truth_psd_x, color='r')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('Wave length (km)', fontsize=12)
    ax1.set_ylabel('Power Spectral Density of X axis', fontsize=12)
    ax1.legend(['Prediction', 'Observation'])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(wavelength_y, pred_psd_y, color='b')
    ax2.plot(wavelength_y, truth_psd_y, color='r')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave length (km)', fontsize=12)
    ax2.set_ylabel('Power Spectral Density of Y axis', fontsize=12)
    ax2.legend(['Prediction', 'Observation'])

    fig.savefig('{}/{}_psd.png'.format(root, stage), bbox_inches='tight')
    print('PSD saved')
