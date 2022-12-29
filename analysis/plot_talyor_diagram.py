import os
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from taylor_diagram import TaylorDiagram


plt.rcParams['font.sans-serif'] = 'Arial'

def plot_taylor_diagram(root: str, paths: list, models: list, target_path: str, std_range: list, std_num: int, colors: list):
    fig = plt.figure(figsize=(10, 4), dpi=600)

    truth = torch.load(os.path.join(root, paths[-1], 'truth', 'truth.pt'))
    truth_30min, truth_60min = truth[0, 4, 0], truth[0, 9, 0]
    ref_std_30min, ref_std_60min = torch.std(truth_30min), torch.std(truth_60min)

    taylor_diagram_30min = TaylorDiagram(ref_std_30min.numpy(), fig, rect=121, 
                                         std_min=std_range[0], std_max=std_range[1],
                                         std_label_format='%.0f', num_std=std_num, 
                                         label='Observation', ylabel_text='$\sigma_{\hat{y}}$')
    taylor_diagram_60min = TaylorDiagram(ref_std_60min.numpy(), fig, rect=122, 
                                         std_min=std_range[0], std_max=std_range[1],
                                         std_label_format='%.0f', num_std=std_num, 
                                         label='Observation', ylabel_text='$\sigma_{\hat{y}}$')

    for i, path in enumerate(paths):
        pred = torch.load(os.path.join(root, path, 'pred', 'pred.pt'))
        pred_30min, pred_60min = pred[0, 4, 0], pred[0, 9, 0]
        stddev_30min, stddev_60min = torch.std(pred_30min), torch.std(pred_60min)
        corrcoef_30min = torch.corrcoef(torch.stack([truth_30min.flatten(), pred_30min.flatten()]))[0, 1]
        corrcoef_60min = torch.corrcoef(torch.stack([truth_60min.flatten(), pred_60min.flatten()]))[0, 1]
        taylor_diagram_30min.add_sample(stddev_30min.numpy(), corrcoef_30min.numpy(),
                                        marker='$%d$' % (i + 1), ms=5, ls='',
                                        mfc=colors[i], mec=colors[i], label=models[i])
        taylor_diagram_60min.add_sample(stddev_60min.numpy(), corrcoef_60min.numpy(),
                                        marker='$%d$' % (i + 1), ms=5, ls='',
                                        mfc=colors[i], mec=colors[i], label=models[i])
    
    # Add grid
    taylor_diagram_30min.add_grid()
    taylor_diagram_60min.add_grid()

    # Add RMS contours, and label them
    contours_30 = taylor_diagram_30min.add_contours(colors='0.5')
    plt.clabel(contours_30, inline=1, fontsize='medium', fmt='%.2f')
    contours_60 = taylor_diagram_60min.add_contours(colors='0.5')
    plt.clabel(contours_60, inline=1, fontsize='medium', fmt='%.2f')
    
    # Add a figure legend
    taylor_diagram_30min.ax.legend(taylor_diagram_30min.samplePoints,
                                   [p.get_label() for p in taylor_diagram_30min.samplePoints],
                                   numpoints=1, fontsize=8, bbox_to_anchor=(1.2, 1.1))
    taylor_diagram_60min.ax.legend(taylor_diagram_60min.samplePoints,
                                   [p.get_label() for p in taylor_diagram_60min.samplePoints],
                                   numpoints=1, fontsize=8, bbox_to_anchor=(1.2, 1.1))
    
    # Add title
    fig.axes[0].set_title('(a)\n', loc='left')
    fig.axes[1].set_title('(b)\n', loc='left')

    fig.tight_layout()
    fig.savefig(target_path)


if __name__ == '__main__':
    colors = cm.get_cmap('tab10')
    plot_taylor_diagram('results', 
                        ['AttnUNet/sample_0', 'AttnUNet_SVRE/sample_0', 'AttnUNet_GA/sample_0', 'AttnUNet_GASVRE/sample_0'], 
                        ['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'], 
                        'img/taylor_ablation_0.jpg', std_range=[9, 19], std_num=6, 
                        colors=colors.colors)
    plot_taylor_diagram('results',
                        ['AttnUNet/sample_1', 'AttnUNet_SVRE/sample_1', 'AttnUNet_GA/sample_1', 'AttnUNet_GASVRE/sample_1'],
                        ['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'],
                        'img/taylor_ablation_1.jpg', std_range=[9, 19], std_num=6,
                        colors=colors.colors)
