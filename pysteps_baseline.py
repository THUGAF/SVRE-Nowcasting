import os
import argparse
import warnings
import random
import torch
import pandas as pd
import numpy as np
from pysteps.motion.darts import DARTS as motion_method
from pysteps.nowcasts.sprog import forecast as nowcast_method
import utils.dataloader as dataloader
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.transform as transform
from utils.trainer import HiddenPrints


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='PySTEPS Basline')

# global settings
parser.add_argument('--data-path', type=str, default='/data/gaf/SBandCRPt')
parser.add_argument('--output-path', type=str, default='results/PySTEPS')
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--valid-ratio', type=float, default=0.1)
parser.add_argument('--sample-indices', type=int, nargs='+', default=[0])
parser.add_argument('--lon-range', type=int, nargs='+', default=[271, 527])
parser.add_argument('--lat-range', type=int, nargs='+', default=[335, 591])
parser.add_argument('--display-interval', type=int, default=1)
parser.add_argument('--seed', type=int, default=2023)

# input and output settings
parser.add_argument('--input-steps', type=int, default=10)
parser.add_argument('--forecast-steps', type=int, default=10)

# training settings
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')

# evaluation settings
parser.add_argument('--thresholds', type=int, nargs='+', default=[30, 40])
parser.add_argument('--vmax', type=float, default=70.0)
parser.add_argument('--vmin', type=float, default=-10.0)

args = parser.parse_args()


def main(args):
    # fix the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.test:
        test(args)
    if args.predict:
        predict(args)


def test(args):
    metrics = {}
    metrics['Time'] = np.linspace(6, 60, 10)
    for threshold in args.thresholds:
        metrics['POD-%ddBZ' % threshold] = []
        metrics['FAR-%ddBZ' % threshold] = []
        metrics['CSI-%ddBZ' % threshold] = []
    metrics['ME'] = []
    metrics['MAE'] = []
    metrics['SSIM'] = []
    metrics['KLD'] = []

    test_loader = dataloader.load_data(args.data_path, args.input_steps, args.forecast_steps, 1, 1,
                                       args.train_ratio, args.valid_ratio, args.lon_range, args.lat_range)[2]
    print('\n[Test]')
    for i, (tensor, timestamp) in enumerate(test_loader):
        input_ = tensor[:, :args.input_steps]
        truth = tensor[:, args.input_steps:]
        input_pysteps = input_[0, :, 0].numpy()
        with HiddenPrints():
            velocity = motion_method(input_pysteps)
            pred_pystpes = nowcast_method(input_pysteps[-2:], velocity, args.forecast_steps, R_thr=-10, ar_order=1)
        pred = torch.from_numpy(np.nan_to_num(pred_pystpes)).view_as(truth)

        if (i + 1) % args.display_interval == 0:
                print('Batch: [{}][{}]'.format(i + 1, len(test_loader)))

        pred_norm = transform.minmax_norm(input_, args.vmax, args.vmin)
        truth_norm = transform.minmax_norm(truth, args.vmax, args.vmin)
        for threshold in args.thresholds:
            pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
            metrics['POD-%ddBZ' % threshold].append(pod)
            metrics['FAR-%ddBZ' % threshold].append(far)
            metrics['CSI-%ddBZ' % threshold].append(csi)
        metrics['ME'].append(evaluation.evaluate_me(pred, truth))
        metrics['MAE'].append(evaluation.evaluate_mae(pred, truth))
        metrics['SSIM'].append(evaluation.evaluate_ssim(pred_norm, truth_norm))
        metrics['KLD'].append(evaluation.evaluate_kld(pred, truth))

    print('\nEvaluating...')
    for key in metrics.keys():
        if key != 'Time':
            metrics[key] = np.mean(metrics[key], axis=0)
    df = pd.DataFrame(data=metrics)
    df.to_csv(os.path.join(args.output_path, 'test_metrics.csv'), float_format='%.4g', index=False)
    print('Evaluation complete')
    print('\nVisualizing...')
    visualizer.plot_map(input_, pred, truth, timestamp, args.output_path, 'test')
    print('Visualization complete')
    print('\nTest complete')


def predict(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    # load data
    sample_loader = dataloader.load_sample(args.data_path, args.sample_indices, args.input_steps, 
                                           args.forecast_steps, args.lon_range, args.lat_range)
    
    print('\n[Predict]')
    # predict
    for i, (tensor, timestamp) in enumerate(sample_loader):
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        input_ = tensor[:, :args.input_steps]
        truth = tensor[:, args.input_steps:]
        input_pysteps = input_[0, :, 0].numpy()
        with HiddenPrints():
            velocity = motion_method(input_pysteps)
            pred_pystpes = nowcast_method(input_pysteps[-2:], velocity, args.forecast_steps, R_thr=-10, ar_order=1)
        pred = torch.from_numpy(np.nan_to_num(pred_pystpes)).view_as(truth)

        # visualization
        print('\nVisualizing...')
        visualizer.plot_map(input_, pred, truth, timestamp, args.output_path, 'sample_{}'.format(i))
        visualizer.plot_psd(pred, truth, args.output_path, 'sample_{}'.format(i))
        print('Visualization done')
    
        # evaluation
        print('\nEvaluating...')
        pred_norm = transform.minmax_norm(pred, args.vmax, args.vmin)
        truth_norm = transform.minmax_norm(truth, args.vmax, args.vmin)
        
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        for threshold in args.thresholds:
            pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
            metrics['POD-{}dBZ'.format(str(threshold))] = pod
            metrics['FAR-{}dBZ'.format(str(threshold))] = far
            metrics['CSI-{}dBZ'.format(str(threshold))] = csi
        metrics['ME'] = evaluation.evaluate_me(pred, truth)
        metrics['MAE'] = evaluation.evaluate_mae(pred, truth)
        metrics['SSIM'] = evaluation.evaluate_ssim(pred_norm, truth_norm)
        metrics['KLD'] = evaluation.evaluate_kld(pred, truth)
        
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(args.output_path, 'sample_{}_metrics.csv'.format(i)), float_format='%.4g', index=False)
        print('Evaluation complete')

        print('\nAll tasks complete')


if __name__ == '__main__':
    main(args)
