import os
import argparse
import warnings

import torch
import pandas as pd
import numpy as np
import pysteps

import utils.dataloader as dataloader
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.scaler as scaler


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='PySTEPS Basline')

# global settings
parser.add_argument('--data-path', type=str, default='/data/gaf/SBandCRUnzip')
parser.add_argument('--output-path', type=str, default='results/PySTEPS')
parser.add_argument('--sample-indices', type=int, nargs='+', default=[0])
parser.add_argument('--lon-range', type=int, nargs='+', default=[271, 527])
parser.add_argument('--lat-range', type=int, nargs='+', default=[335, 591])
parser.add_argument('--seed', type=int, default=2023)

# input and output settings
parser.add_argument('--input-steps', type=int, default=10)
parser.add_argument('--forecast-steps', type=int, default=10)

# evaluation settings
parser.add_argument('--thresholds', type=int, nargs='+', default=[10, 20, 30, 40])
parser.add_argument('--vmax', type=float, default=70.0)
parser.add_argument('--vmin', type=float, default=-10.0)

args = parser.parse_args()


def main():
    # fix the random seed
    np.random.seed(args.seed)
    predict(args)


def predict(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    # load data
    print('Loading data...')
    sample_loader = dataloader.load_sample(args.data_path, args.sample_indices, args.input_steps, 
                                           args.forecast_steps, args.lon_range, args.lat_range)
    
    # predict
    for i, (tensor, timestamp) in enumerate(sample_loader):
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        input_ = tensor[:, :args.input_steps]
        truth = tensor[:, args.input_steps:]
        input_pysteps = input_[0, :, 0].numpy()
        velocity = pysteps.motion.get_method('darts')(input_pysteps)
        pred_pystpes = pysteps.nowcasts.get_method('sprog')(input_pysteps, velocity, args.forecast_steps, R_thr=0)
        pred = torch.from_numpy(np.nan_to_num(pred_pystpes)).view_as(truth)

        # visualization
        print('Visualizing...')
        visualizer.plot_map(input_, pred, truth, timestamp, args.output_path, 'sample_{}'.format(i))
    
        # evaluation
        print('Evaluating...')
        pred_rev, truth_rev = pred, truth
        pred = scaler.minmax_norm(pred, args.vmax, args.vmin)
        truth = scaler.minmax_norm(truth, args.vmax, args.vmin)
        
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        for threshold in args.thresholds:
            pod, far, csi, hss = evaluation.evaluate_forecast(pred_rev, truth_rev, threshold)
            metrics['POD-{}dBZ'.format(str(threshold))] = pod
            metrics['FAR-{}dBZ'.format(str(threshold))] = far
            metrics['CSI-{}dBZ'.format(str(threshold))] = csi
        metrics['CC'] = evaluation.evaluate_cc(pred_rev, truth_rev)
        metrics['ME'] = evaluation.evaluate_me(pred_rev, truth_rev)
        metrics['MAE'] = evaluation.evaluate_mae(pred_rev, truth_rev)
        metrics['RMSE'] = evaluation.evaluate_rmse(pred_rev, truth_rev)
        metrics['SSIM'] = evaluation.evaluate_ssim(pred, truth)
        metrics['PSNR'] = evaluation.evaluate_psnr(pred, truth)
        
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(args.output_path, 'sample_{}_metrics.csv'.format(i)), float_format='%.4g', index=False)

        print('\nBaseline Done.')


if __name__ == '__main__':
    main()
