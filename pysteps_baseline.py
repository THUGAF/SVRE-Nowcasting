import os
import argparse
import warnings

import torch
import pandas as pd
import numpy as np
import pysteps

import utils.visualizer as visualizer
import utils.saver as saver
import utils.evaluation as evaluation
import utils.scaler as scaler


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='PySTEPS Basline')

# global settings
parser.add_argument('--data-path', type=str, default='/data/gaf/SBandCRNpz')
parser.add_argument('--output-path', type=str, default='results/PySTEPS')
parser.add_argument('--sample-point', type=int, default=16060)
parser.add_argument('--lon-range', type=int, nargs='+', default=[273, 529])
parser.add_argument('--lat-range', type=int, nargs='+', default=[270, 526])
parser.add_argument('--seed', type=int, default=2021)

# input and output settings
parser.add_argument('--input-steps', type=int, default=10)
parser.add_argument('--forecast-steps', type=int, default=10)

# evaluation settings
parser.add_argument('--thresholds', type=int, nargs='+', default=[10, 15, 20, 25, 30, 35, 40])
parser.add_argument('--vmax', type=float, default=80.0)
parser.add_argument('--vmin', type=float, default=0.0)

args = parser.parse_args()


def main():
    # fix the random seed
    np.random.seed(args.seed)
    nowcast(args)


def nowcast(args):
    # nowcast
    print('Loading data...')
    data_file = np.load(os.path.join(args.data_path, str(args.sample_point) + '.npz'))
    data, seconds = data_file['DBZ'][:, 0, args.lat_range[0]: args.lat_range[1], args.lon_range[0]: args.lon_range[1]], data_file['UNIX_Time']
    input_, truth = data[:args.input_steps], data[args.input_steps: args.input_steps + args.forecast_steps]

    print('Nowcasting...')
    velocity = pysteps.motion.darts.DARTS(input_)
    pred = pysteps.nowcasts.steps.forecast(input_[-3:], velocity, args.forecast_steps,
                                           n_ens_members=24, R_thr=5, kmperpixel=1, timestep=6)
    pred = np.mean(pred, axis=0)
    pred[np.isnan(pred)] = 0

    # visualization
    print('Visualizing...')
    input_, pred, truth = torch.from_numpy(input_).unsqueeze(1).unsqueeze(2), \
        torch.from_numpy(pred).unsqueeze(1).unsqueeze(2), torch.from_numpy(truth).unsqueeze(1).unsqueeze(2)
    seconds = torch.from_numpy(seconds).unsqueeze(1)
    visualizer.plot_map(input_, pred, truth, seconds, args.output_path,
                        stage='sample', lon_range=args.lon_range, lat_range=args.lat_range)
    saver.save_tensors(input_, pred, truth, args.output_path, stage='sample')
    
    # evaluation
    print('Evaluating...')
    pred_rev, truth_rev = pred, truth
    pred = scaler.minmax_norm(pred, args.vmax, args.vmin)
    truth = scaler.minmax_norm(truth, args.vmax, args.vmin)
    metrics = {}
    for threshold in args.thresholds:
        pod, far, csi, hss = evaluation.evaluate_forecast(pred_rev, truth_rev, threshold)
        metrics['POD-{}dBZ'.format(str(threshold))] = pod
        metrics['FAR-{}dBZ'.format(str(threshold))] = far
        metrics['CSI-{}dBZ'.format(str(threshold))] = csi
        metrics['HSS-{}dBZ'.format(str(threshold))] = hss

    metrics['CC'] = evaluation.evaluate_cc(pred_rev, truth_rev)
    metrics['ME'] = evaluation.evaluate_me(pred_rev, truth_rev)
    metrics['MAE'] = evaluation.evaluate_mae(pred_rev, truth_rev)
    metrics['RMSE'] = evaluation.evaluate_rmse(pred_rev, truth_rev)
    metrics['SSIM'] = evaluation.evaluate_ssim(pred, truth)
    metrics['PSNR'] = evaluation.evaluate_psnr(pred, truth)
    metrics['CVR'] = evaluation.evaluate_cvr(pred_rev, truth_rev)
    metrics['SSDR'] = evaluation.evaluate_ssdr(pred_rev, truth_rev)

    if not os.path.exists(os.path.join(args.output_path, 'metrics')):
            os.mkdir(os.path.join(args.output_path, 'metrics'))
    
    df = pd.DataFrame(data=metrics)
    df.to_csv(os.path.join(args.output_path, 'metrics', 'sample_metrics.csv'), float_format='%.8f')

    print('\nBaseline Done.')


if __name__ == '__main__':
    main()
