import os
import sys
import time
import argparse
import random
import torch
import pandas as pd
import numpy as np
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.extrapolation.semilagrangian import extrapolate
from torch.utils.data import DataLoader
import utils.dataloader as dataloader
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.transform as transform


parser = argparse.ArgumentParser(description='PySTEPS Basline')

# input and output settings
parser.add_argument('--data-path', type=str)
parser.add_argument('--output-path', type=str, default='results/PySTEPS')
parser.add_argument('--input-steps', type=int, default=10)
parser.add_argument('--forecast-steps', type=int, default=10)
parser.add_argument('--case-indices', type=int, nargs='+', default=[0])

# training settings
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--valid-ratio', type=float, default=0.1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--display-interval', type=int, default=1)
parser.add_argument('--random-seed', type=int, default=2023)

# nowcasting settings
parser.add_argument('--resolution', type=float, default=6.0)
parser.add_argument('--x-range', type=int, nargs='+', default=[272, 528])
parser.add_argument('--y-range', type=int, nargs='+', default=[336, 592])

# evaluation settings
parser.add_argument('--thresholds', type=int, nargs='+', default=[20, 30, 40])

args = parser.parse_args()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def main(args):
    print('### Initialize settings ###')
    # Fix the random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    args.device = 'cpu'

    # Make dir
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Test and predict
    print('\n### Start tasks ###')
    if args.test:
        test_loader = dataloader.load_data(args.data_path, args.input_steps, args.forecast_steps,
                                           1, args.num_workers, args.train_ratio, args.valid_ratio,
                                           args.x_range, args.y_range, pin_memory=False)[2]
        test(test_loader)
    if args.predict:
        case_loader = dataloader.load_case(args.data_path, args.case_indices, args.input_steps,
                                           args.forecast_steps, args.x_range, args.y_range)
        predict(case_loader)

    print('\n### All tasks complete ###')


@torch.no_grad()
def test(test_loader: DataLoader):
    # Init metric dict
    metrics = {}
    for threshold in args.thresholds:
        metrics['POD_{:.1f}'.format(threshold)] = 0
        metrics['FAR_{:.1f}'.format(threshold)] = 0
        metrics['CSI_{:.1f}'.format(threshold)] = 0
    metrics['MBE'] = 0
    metrics['MAE'] = 0
    metrics['RMSE'] = 0
    metrics['SSIM'] = 0
    metrics['JSD'] = 0

    print('\n[Test]')
    
    # Timer
    test_timer = time.time()
    test_batch_timer = time.time()

    for i, (tensor, _) in enumerate(test_loader):
        tensor = tensor.to(args.device)
        input_ = tensor[:, :args.input_steps]
        truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
        input_pysteps = input_[0, :, 0].numpy()
        with HiddenPrints():
            velocity = dense_lucaskanade(input_pysteps)
            pred_pysteps = extrapolate(input_pysteps[-1], velocity, args.forecast_steps)
            pred_pysteps = np.nan_to_num(pred_pysteps)
        pred = torch.from_numpy(pred_pysteps).view_as(truth)
        truth_R = transform.ref_to_R(truth)
        pred_R = transform.ref_to_R(pred)

        if (i + 1) % args.display_interval == 0:
            print('Batch: [{}][{}]\t\tTime: {:.4f}'.format(
                i + 1, len(test_loader), time.time() - test_batch_timer))
            test_batch_timer = time.time()

        pred_norm = transform.minmax_norm(pred)
        truth_norm = transform.minmax_norm(truth)
        
        # Evaluation
        for threshold in args.thresholds:
            pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
            metrics['POD_{:.1f}'.format(threshold)] += pod
            metrics['FAR_{:.1f}'.format(threshold)] += far
            metrics['CSI_{:.1f}'.format(threshold)] += csi
        metrics['MBE'] += evaluation.evaluate_mbe(pred_R, truth_R)
        metrics['MAE'] += evaluation.evaluate_mae(pred_R, truth_R)
        metrics['RMSE'] += evaluation.evaluate_rmse(pred_R, truth_R)
        metrics['SSIM'] += evaluation.evaluate_ssim(pred_norm, truth_norm)
        metrics['JSD'] += evaluation.evaluate_jsd(pred, truth)

    # Print time
    print('Time: {:.4f}'.format(time.time() - test_timer))

    # Save metrics
    for key in metrics.keys():
        metrics[key] /= len(test_loader)
    df = pd.DataFrame(data=metrics, index=[0])
    df.to_csv(os.path.join(args.output_path, 'test_metrics.csv'), 
              float_format='%.6f', index=False)
    print('Test metrics saved')


@torch.no_grad()
def predict(case_loader: DataLoader):
    # Init metric dict
    metrics = {}
    # predict
    print('\n[Predict]')
    for i, (tensor, timestamp) in enumerate(case_loader):
        tensor = tensor.to(args.device)
        input_ = tensor[:, :args.input_steps]
        truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
        input_pysteps = input_[0, :, 0].numpy()
        velocity = dense_lucaskanade(input_pysteps)
        pred_pysteps = extrapolate(input_pysteps[-1], velocity, args.forecast_steps)
        pred_pysteps = np.nan_to_num(pred_pysteps)
        pred = torch.from_numpy(pred_pysteps).view_as(truth)
        truth_R = transform.ref_to_R(truth)
        pred_R = transform.ref_to_R(pred)
    
        # Evaluation
        for threshold in args.thresholds:
            pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
            metrics['POD_{:.1f}'.format(threshold)] = pod
            metrics['FAR_{:.1f}'.format(threshold)] = far
            metrics['CSI_{:.1f}'.format(threshold)] = csi
        metrics['MBE'] = evaluation.evaluate_mbe(pred_R, truth_R)
        metrics['MAE'] = evaluation.evaluate_mae(pred_R, truth_R)
        metrics['RMSE'] = evaluation.evaluate_rmse(pred_R, truth_R)
        pred_norm = transform.minmax_norm(pred)
        truth_norm = transform.minmax_norm(truth)
        metrics['SSIM'] = evaluation.evaluate_ssim(pred_norm, truth_norm)
        metrics['JSD'] = evaluation.evaluate_jsd(pred, truth)
        df = pd.DataFrame(data=metrics, index=[0])
        df.to_csv(os.path.join(args.output_path, 'case_{}_metrics.csv'.format(i)), 
                  float_format='%.6f', index=False)
        print('Case {} metrics saved'.format(i))

        # Save tensors and figures
        visualizer.save_tensor(input_, timestamp[:, :args.input_steps],
                               args.output_path, 'case_{}'.format(i), 'input')
        visualizer.save_tensor(truth, timestamp[:, args.input_steps: args.input_steps + args.forecast_steps],
                               args.output_path, 'case_{}'.format(i), 'truth')
        visualizer.save_tensor(pred, timestamp[:, args.input_steps: args.input_steps + args.forecast_steps],
                               args.output_path, 'case_{}'.format(i), 'pred')
        print('Tensors saved')
        visualizer.plot_figs(input_, timestamp[:, :args.input_steps],
                             args.output_path, 'case_{}'.format(i), 'input')
        visualizer.plot_figs(truth, timestamp[:, args.input_steps: args.input_steps + args.forecast_steps],
                             args.output_path, 'case_{}'.format(i), 'truth')
        visualizer.plot_figs(pred, timestamp[:, args.input_steps: args.input_steps + args.forecast_steps],
                             args.output_path, 'case_{}'.format(i), 'pred')
        visualizer.plot_psd(pred, truth, args.output_path, 'case_{}'.format(i))
        print('Figures saved')
    
    print('\nPrediction complete')


if __name__ == '__main__':
    main(args)
