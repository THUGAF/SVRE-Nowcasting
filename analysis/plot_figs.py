import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import matplotlib.cm as cm
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


plt.rcParams['font.sans-serif'] = 'Arial'

# Coordinate transformation
TRANS_WGS84_TO_UTM = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857')
TRANS_UTM_TO_WGS84 = pyproj.Transformer.from_crs('epsg:3857', 'epsg:4326')

# Global information
CENTER_LON, CENTER_LAT = 116.47195, 39.808887
CENTER_UTM_X, CENTER_UTM_Y = TRANS_WGS84_TO_UTM.transform(CENTER_LAT, CENTER_LON)
LEFT_BOTTOM_LAT, LEFT_BOTTOM_LON = TRANS_UTM_TO_WGS84.transform(CENTER_UTM_X - 128000, CENTER_UTM_Y - 64000)
RIGHT_TOP_LAT, RIGHT_TOP_LON = TRANS_UTM_TO_WGS84.transform(CENTER_UTM_X + 128000, CENTER_UTM_Y + 192000)
AREA = [LEFT_BOTTOM_LON, RIGHT_TOP_LON, LEFT_BOTTOM_LAT, RIGHT_TOP_LAT]

CMAP = pcolors.ListedColormap([[255 / 255, 255 / 255, 255 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                                   [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                                   [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                                   [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                                   [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255]])
NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), CMAP.N)


plt.rcParams['font.sans-serif'] = 'Arial'

def plot_maps(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    fig = plt.figure(figsize=(18, 12), dpi=600)
    input_ = torch.load(os.path.join(model_dirs[0], stage, 'input', 'input.pt'))
    truth = torch.load(os.path.join(model_dirs[0], stage, 'truth', 'truth.pt'))
    input_ = np.flip(input_[0, 8, 0].numpy(), axis=0)
    truth = np.flip(truth[0, 8, 0].numpy(), axis=0)
    for i in range(len(model_names) + 2):
        ax = fig.add_subplot(2, len(model_names) // 2 + 1, i + 1, projection=ccrs.Mercator())
        if i == 0:
            tensor = input_
            title = 'observation (0 min)'
        elif i == 1:
            tensor = truth
            title = 'observation (+60 min)'
        else:
            pred = torch.load(os.path.join(model_dirs[i - 2], stage, 'pred', 'pred.pt'))
            tensor = np.flip(pred[0, 8, 0].numpy(), axis=0)
            title = model_names[i - 2]
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
        ax.set_title(title, fontsize=18)

    fig.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    print('{} saved'.format(img_path))


if __name__ == '__main__':
    model_names = ['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE']
    model_dirs = ['results/AttnUNet', 'results/AttnUNet_SVRE', 'results/AttnUNet_GA', 'results/AttnUNet_GASVRE']
    plot_maps(model_names, model_dirs, 'sample_0', 'img/vis_ablation_sample_0.png')
    plot_maps(model_names, model_dirs, 'sample_1', 'img/vis_ablation_sample_1.png')