import os
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from taylor_diagram import TaylorDiagram


plt.rcParams['font.sans-serif'] = 'Arial'

def plot_taylor_diagram(root: str, paths: list, models: list, target_path: str, std_range: list, std_num: int, colors: list):
    fig = plt.figure(figsize=(4, 4), dpi=600)
    truth = torch.load(os.path.join(root, paths[-1], 'truth', 'truth.pt'))
    truth_60min = truth[0, 8, 0]
    ref_std_60min = torch.std(truth_60min).numpy()
    taylor_diagram_60min = TaylorDiagram(ref_std_60min, fig, rect=111, 
                                         std_min=std_range[0], std_max=std_range[1],
                                         std_label_format='%.1f', num_std=std_num, 
                                         label='Observation', normalized=True)
    markers = ['o', '^', 'p', 'd']
    for i, path in enumerate(paths):
        pred = torch.load(os.path.join(root, path, 'pred', 'pred.pt'))
        pred_60min = pred[0, 8, 0]
        stddev_60min = torch.std(pred_60min).numpy()
        # if i == 3:
        #     stddev_60min *= 1.2
        corrcoef_60min = torch.corrcoef(torch.stack([truth_60min.flatten(), pred_60min.flatten()]))[0, 1].numpy()
        taylor_diagram_60min.add_sample(stddev_60min / ref_std_60min, corrcoef_60min, 
                                        ms=5, ls='', marker=markers[i],
                                        mfc=colors[i], mec=colors[i], label=models[i])

    # Add grid
    taylor_diagram_60min.add_grid()

    # Add RMS contours, and label them
    contours_60 = taylor_diagram_60min.add_contours(colors='0.5')
    plt.clabel(contours_60, inline=1, fontsize='medium', fmt='%.2f')
    
    # Add a figure legend
    taylor_diagram_60min.ax.legend(taylor_diagram_60min.samplePoints,
                                   [p.get_label() for p in taylor_diagram_60min.samplePoints],
                                   numpoints=1, fontsize=8, bbox_to_anchor=(1.1, 1.1))
    
    # Add title
    fig.tight_layout()
    fig.savefig(target_path)


if __name__ == '__main__':
    colors = cm.get_cmap('tab10')
    plot_taylor_diagram('results', 
                        ['AttnUNet/sample_0', 'AttnUNet_SVRE/sample_0', 'AttnUNet_GA/sample_0', 'AttnUNet_GASVRE/sample_0'], 
                        ['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'], 
                        'img/taylor_ablation_sample_0.jpg', std_range=[0.6, 1.6], std_num=6, colors=colors.colors)
    plot_taylor_diagram('results',
                        ['AttnUNet/sample_1', 'AttnUNet_SVRE/sample_1', 'AttnUNet_GA/sample_1', 'AttnUNet_GASVRE/sample_1'],
                        ['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'],
                        'img/taylor_ablation_sample_1.jpg', std_range=[0.6, 1.6], std_num=6, colors=colors.colors)
    plot_taylor_diagram('results', 
                        ['PySTEPS/sample_0', 'SmaAt_UNet/sample_0', 'MotionRNN/sample_0', 'AttnUNet_GASVRE/sample_0'], 
                        ['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AGAN+SVRE'],
                        'img/taylor_comparison_sample_0.jpg', std_range=[0.6, 1.6], std_num=6, colors=colors.colors)
    plot_taylor_diagram('results',
                        ['PySTEPS/sample_1', 'SmaAt_UNet/sample_1', 'MotionRNN/sample_1', 'AttnUNet_GASVRE/sample_1'],
                        ['PySTEPS', 'SmaAt-UNet', 'MotionRNN', 'AGAN+SVRE'],
                        'img/taylor_comparison_sample_1.jpg', std_range=[0.6, 1.6], std_num=6, colors=colors.colors)
