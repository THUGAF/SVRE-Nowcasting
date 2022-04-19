import os
import datetime
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import matplotlib.cm as cm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


plt.rcParams['font.sans-serif'] = 'Arial'

CMAP = pcolors.ListedColormap([[255 / 255, 255 / 255, 255 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                               [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                               [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                               [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                               [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255]])
NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), CMAP.N)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def plot_loss(train_loss, val_loss, root, title, filename='loss'):
    mkdir(root)
    mkdir(os.path.join(root, 'loss'))
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax1 = plt.subplot(111)
    _plot_loss_fig(ax1, train_loss, val_loss, title)
    fig.savefig(os.path.join(root, 'loss', filename))
    plt.close(fig)


def _plot_loss_fig(ax, train_loss, val_loss, title):
    ax.plot(range(1, len(train_loss) + 1), train_loss, 'b')
    ax.plot(range(1, len(val_loss) + 1), val_loss, 'r')
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.legend(['train loss', 'val loss'])


def plot_map(input_, pred, truth, seconds, root, stage, lon_range, lat_range):
    mkdir(root)
    mkdir(os.path.join(root, 'images'))
    mkdir(os.path.join(root, 'images', stage))
    _plot_map_figs(input_, root, seconds[:input_.size(0)], stage, data_type='input', 
                   cmap=CMAP, norm=NORM, lon_range=lon_range, lat_range=lat_range)
    _plot_map_figs(pred, root, seconds[input_.size(0): input_.size(0) + pred.size(0)], 
                   stage, data_type='pred', cmap=CMAP, norm=NORM, 
                   lon_range=lon_range, lat_range=lat_range)
    _plot_map_figs(truth, root, seconds[input_.size(0): input_.size(0) + truth.size(0)], 
                   stage, data_type='truth', cmap=CMAP, norm=NORM, 
                   lon_range=lon_range, lat_range=lat_range)


def _plot_map_figs(tensor, root, seconds, stage, data_type, cmap, norm, lon_range, lat_range):
    path = os.path.join(root, 'images', stage, data_type)
    mkdir(path)

    # inverse scaling
    tensor = tensor.detach().cpu()

    image_list = []
    for i in range(tensor.size(0)):
        # minus represents the time before current moment
        if data_type == 'input':
            str_min = str(6 * (i - tensor.size(0) + 1))
        else:
            str_min = str(6 * (i + 1))
        file_path = '{}/{}.png'.format(path, str_min)
        current_datetime = datetime.datetime(year=1970, month=1, day=1) + datetime.timedelta(seconds=int(seconds[i, 0]))
        _plot_map_fig(tensor[i, 0, 0], file_path, current_datetime, cmap, norm, lon_range, lat_range)
        image_list.append(imageio.imread(file_path))

    # plot the long image
    num_rows = 2 if tensor.size(0) > 10 else 1
    num_cols = tensor.size(0) // num_rows
    fig = plt.figure(figsize=(num_cols, num_rows), dpi=300)
    for i in range(tensor.size(0)):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(np.flip(tensor[i, 0, 0].numpy(), axis=0), cmap=cmap, norm=norm)
        ax.axis('off')
    
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    fig.savefig('{}/{}.png'.format(path, data_type))
    plt.close(fig)
    
    # make gif
    imageio.mimsave('{}/{}.gif'.format(path, data_type), image_list, 'GIF', duration=0.2)


def _plot_map_fig(tensor_slice, file_path, current_datetime, cmap, norm, lon_range, lat_range):
    fig = plt.figure(figsize=(8, 8), dpi=300)
    fig.suptitle('\n' + current_datetime.strftime('%Y-%m-%d %H:%M:%S'), fontsize=24)
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.set_title('CR', fontsize=18)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)

    tensor_slice = np.flip(tensor_slice.numpy(), axis=0)
    area = [112.47 + lon_range[0] * 0.01, 112.47 + lon_range[1] * 0.01,
            35.80 + lat_range[0] * 0.01, 35.80 + lat_range[1] * 0.01]
    ax.imshow(tensor_slice, cmap=cmap, norm=norm, extent=area)
    ax.set_extent(area, crs=ccrs.PlateCarree())

    ax.set_xticks(np.arange(np.ceil(area[0]), np.ceil(area[1]), 1), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.ceil(area[2]), np.ceil(area[3]), 1), crs=ccrs.PlateCarree())
    ax.gridlines(crs=ccrs.PlateCarree(), 
                xlocs=np.arange(np.ceil(2 * area[0]) / 2, np.ceil(2 * area[1]) / 2, 0.5), 
                ylocs=np.arange(np.ceil(2 * area[2]) / 2, np.ceil(2 * area[3]) / 2, 0.5), 
                draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=18)

    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), pad=0.05, shrink=0.7, aspect=20, orientation='vertical', extend='both')
    cbar.set_label('dBZ', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    plt.subplots_adjust(left=0.1, right=1, bottom=0, top=1)
    fig.savefig(file_path)
    plt.close(fig)
