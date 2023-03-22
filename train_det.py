import os
import time
import shutil
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.transform as transform
import utils.dataloader as dataloader
import models


parser = argparse.ArgumentParser()

# input and output settings
parser.add_argument('--data-path', type=str)
parser.add_argument('--output-path', type=str, default='results')
parser.add_argument('--input-steps', type=int, default=10)
parser.add_argument('--forecast-steps', type=int, default=10)

# data loading settings
parser.add_argument('--train-ratio', type=float, default=0.64)
parser.add_argument('--valid-ratio', type=float, default=0.16)
parser.add_argument('--case-indices', type=int, nargs='+', default=[0])

# model settings
parser.add_argument('--model', type=str, default='AttnUNet')

# training settings
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--max-iterations', type=int, default=100000)
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--weight-svre', type=float, default=0)
parser.add_argument('--weight-recon', type=float, default=10)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--display-interval', type=int, default=1)
parser.add_argument('--random-seed', type=int, default=2023)

# nowcasting settings
parser.add_argument('--resolution', type=float, default=6.0)
parser.add_argument('--x-range', type=int, nargs='+', default=[272, 528])
parser.add_argument('--y-range', type=int, nargs='+', default=[336, 592])
parser.add_argument('--vmax', type=float, default=70.0)
parser.add_argument('--vmin', type=float, default=0.0)

# evaluation settings
parser.add_argument('--thresholds', type=int, nargs='+', default=[20, 30, 40])

args = parser.parse_args()


def main(args):
    print('### Initialize settings ###')

    # Fix the random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'

    # Set model and optimizer
    if args.model == 'AttnUNet':
        model = models.AttnUNet(args.input_steps, args.forecast_steps).to(args.device)
    elif args.model == 'ConvLSTM': 
        model = models.ConvLSTM(args.forecast_steps)
    elif args.model == 'SmaAt_UNet': 
        model = models.SmaAt_UNet(args.input_steps, args.forecast_steps)
    elif args.model == 'MotionRNN':
        model = models.MotionRNN(args.forecast_steps,
                                 args.y_range[1] - args.y_range[0], 
                                 args.x_range[1] - args.x_range[0])
    count_params(model)
    optimizer = optim.Adam(model.parameters(), args.learning_rate,
                           betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay)
    
    # Make dir
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Load data
    if args.train or args.test:
        train_loader, val_loader, test_loader = dataloader.load_data(args.data_path, 
            args.input_steps, args.forecast_steps, args.batch_size, args.num_workers, 
            args.train_ratio, args.valid_ratio, args.x_range, args.y_range)
    if args.predict:
        case_loader = dataloader.load_case(args.data_path, args.case_indices, args.input_steps, 
            args.forecast_steps, args.x_range, args.y_range)

    # Train, test, and predict
    print('\n### Start tasks ###')
    if args.train:
        train(model, optimizer, train_loader, val_loader)
    if args.test:
        test(model, test_loader)
    if args.predict:
        case_loader = dataloader.load_case(args.data_path, args.case_indices, args.input_steps,
                                           args.forecast_steps, args.x_range, args.y_range)
        predict(model, case_loader)

    print('\n### All tasks complete ###')


def count_params(model: nn.Module):
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([p.numel() for p in model_params])
    print('\nModel name: {}'.format(type(model).__name__))
    print('Total params: {}'.format(num_params))


def save_checkpoint(filename: str, current_iteration: int, train_loss: list, val_loss: list,
                    model: nn.Module, optimizer: optim.Optimizer):
    states = {
        'iteration': current_iteration,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, filename)


def load_checkpoint(filename: str, device: str):
    states = torch.load(filename, map_location=device)
    return states


def early_stopping(metric: list, patience: int = 10):
    early_stopping_flag = False
    counter = 0
    current_epoch = len(metric)
    if current_epoch == 1:
        min_val_loss = np.inf
    else:
        min_val_loss = min(metric[:-1])
    if min_val_loss > metric[-1]:
        print('Metric decreased: {:.4f} --> {:.4f}'.format(min_val_loss, metric[-1]))
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
        shutil.copyfile(checkpoint_path, bestparams_path)
    else:
        min_val_loss_epoch = metric.index(min(metric))
        if current_epoch > min_val_loss_epoch:
            counter = current_epoch - min_val_loss_epoch
            print('EarlyStopping counter: {} out of {}'.format(counter, patience))
            if counter == patience:
                early_stopping_flag = True
    return early_stopping_flag


def weighted_l1_loss(pred: torch.Tensor, truth: torch.Tensor, vmax: float, vmin: float) -> torch.Tensor:
    points = torch.tensor([10.0, 20.0, 30.0, 40.0])
    points = transform.minmax_norm(points, vmax, vmin)
    weight = (truth < points[0]) * 1 \
        + (torch.logical_and(truth >= points[0], truth < points[1])) * 2 \
        + (torch.logical_and(truth >= points[1], truth < points[2])) * 5 \
        + (torch.logical_and(truth >= points[2], truth < points[3])) * 10 \
        + (truth >= points[3]) * 30
    return torch.mean(weight * torch.abs(pred - truth))


def svre_loss(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_cv = torch.std(pred, dim=(2, 3, 4)) / (torch.mean(pred, dim=(2, 3, 4)) + eps)
    truth_cv = torch.std(truth, dim=(2, 3, 4)) / (torch.mean(truth, dim=(2, 3, 4)) + eps)
    return F.l1_loss(pred_cv, truth_cv)


def train(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader):
    # Pretrain
    if args.pretrain:
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        states = load_checkpoint(checkpoint_path, args.device)
        current_iteration = states['iteration']
        train_loss = states['train_loss']
        val_loss = states['val_loss']
        model.load_state_dict(states['model'])
        optimizer.load_state_dict(states['optimizer'])
        start_epoch = int(np.floor(current_iteration / len(train_loader)))
    else:
        current_iteration = 0
        train_loss = []
        val_loss = []
        start_epoch = 0

    # Train and validation
    total_epochs = int(np.ceil((args.max_iterations - current_iteration) / len(train_loader)))
    print('\nMax iterations:', args.max_iterations)
    print('Total epochs:', total_epochs)

    for epoch in range(start_epoch, total_epochs):
        train_loss_epoch = 0
        val_loss_epoch = 0

        # Train
        print('\n[Train]')
        print('Epoch: [{}][{}]'.format(epoch + 1, total_epochs))
        model.train()

        # Timers
        train_epoch_timer = time.time()
        train_batch_timer = time.time()

        for i, (tensor, _) in enumerate(train_loader):
            # Check max iterations
            current_iteration += 1
            if current_iteration > args.max_iterations:
                print('Max iterations reached. Exit!')
                break
            
            # Forward propagation
            tensor = tensor.to(args.device)
            input_ = tensor[:, :args.input_steps]
            truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
            input_norm = transform.minmax_norm(input_, args.vmax, args.vmin)
            truth_norm = transform.minmax_norm(truth, args.vmax, args.vmin)
            pred_norm = model(input_norm)
            loss = args.weight_recon * weighted_l1_loss(pred_norm, truth_norm, args.vmax, args.vmin) + \
                args.weight_svre * svre_loss(pred_norm, truth_norm)

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record and print loss
            train_loss_epoch += loss.item()
            if (i + 1) % args.display_interval == 0:
                print('Epoch: [{}][{}]\tBatch: [{}][{}]\tLoss: {:.4f}\tTime: {:.4f}'.format(
                    epoch + 1, total_epochs, i + 1, len(train_loader), loss.item(), 
                    time.time() - train_batch_timer))
                train_batch_timer = time.time()
            
        # Save train loss
        train_loss_epoch = train_loss_epoch / len(train_loader)
        print('Epoch: [{}][{}]\tLoss: {:.4f}\tTime: {:.4f}'.format(
            epoch + 1, total_epochs, train_loss_epoch, time.time() - train_epoch_timer))
        train_epoch_timer = time.time()
        train_loss.append(train_loss_epoch)
        np.savetxt(os.path.join(args.output_path, 'train_loss.txt'), train_loss)
        print('Train loss saved')

        # Validate
        print('\n[Validate]')
        print('Epoch: [{}][{}]'.format(epoch + 1, total_epochs))
        model.eval()

        # Timers
        val_epoch_timer = time.time()
        val_batch_timer = time.time()

        with torch.no_grad():
            for i, (tensor, _) in enumerate(val_loader):
                # Forward propagation
                tensor = tensor.to(args.device)
                input_ = tensor[:, :args.input_steps]
                truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
                input_norm = transform.minmax_norm(input_, args.vmax, args.vmin)
                truth_norm = transform.minmax_norm(truth, args.vmax, args.vmin)
                pred_norm = model(input_norm)
                loss = args.weight_recon * weighted_l1_loss(pred_norm, truth_norm, args.vmax, args.vmin) + \
                    args.weight_svre * svre_loss(pred_norm, truth_norm)

                # Record and print loss
                val_loss_epoch += loss.item()
                if (i + 1) % args.display_interval == 0:
                    print('Epoch: [{}][{}]\tBatch: [{}][{}]\tLoss: {:.4f}\tTime: {:.4f}'.format(
                        epoch + 1, total_epochs, i + 1, len(val_loader), loss.item()), 
                        time.time() - val_batch_timer)
                    val_batch_timer = time.time()
            
        # Save val loss
        val_loss_epoch = val_loss_epoch / len(val_loader)
        print('Epoch: [{}][{}]\tLoss: {:.4f}\tTime: {:.4f}'.format(
            epoch + 1, total_epochs, val_loss_epoch, time.time() - val_epoch_timer))
        val_epoch_timer = time.time()
        val_loss.append(val_loss_epoch)
        np.savetxt(os.path.join(args.output_path, 'val_loss.txt'), val_loss)
        print('Val loss saved')

        # Plot loss
        visualizer.plot_loss(train_loss, val_loss, args.output_path, 'loss.jpg')
        print('Loss figure saved')

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        save_checkpoint(checkpoint_path, current_iteration, train_loss, val_loss, model, optimizer)
        if args.early_stopping:
            early_stopping_flag = early_stopping(val_loss)
            if early_stopping_flag:
                print('Early stopped')
                break


@torch.no_grad()
def test(model: nn.Module, test_loader: DataLoader):
    # Init metric dict
    metrics = {}
    metrics['Step'] = np.linspace(1, args.forecast_steps) * args.resolution
    for threshold in args.thresholds:
        metrics['POD_{:.1f}'.format(threshold)] = 0
        metrics['FAR_{:.1f}'.format(threshold)] = 0
        metrics['CSI_{:.1f}'.format(threshold)] = 0
    metrics['MBE'] = 0
    metrics['MAE'] = 0
    metrics['RMSE'] = 0
    metrics['SSIM'] = 0
    metrics['KLD'] = 0
    test_loss = 0

    # Test
    print('\n[Test]')
    bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
    states = load_checkpoint(bestparams_path, args.device)
    model.load_state_dict(states['model'])
    model.eval()

    # Timer
    test_timer = time.time()
    test_batch_timer = time.time()

    for i, (tensor, _) in enumerate(test_loader):
        # Forward propagation
        tensor = tensor.to(args.device)
        input_ = tensor[:, :args.input_steps]
        truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
        input_norm = transform.minmax_norm(input_, args.vmax, args.vmin)
        truth_norm = transform.minmax_norm(truth, args.vmax, args.vmin)
        pred_norm = model(input_norm)
        loss = args.weight_recon * weighted_l1_loss(pred_norm, truth_norm, args.vmax, args.vmin) + \
            args.weight_svre * svre_loss(pred_norm, truth_norm)
        pred = transform.reverse_minmax_norm(pred_norm, args.vmax, args.vmin)

        # Record and print loss
        test_loss += loss.item()
        if (i + 1) % args.display_interval == 0:
            print('Batch: [{}][{}]\tLoss: {:.4f}\tTime: {:.4f}'.format(
                i + 1, len(test_loader), loss.item(), time.time() - test_batch_timer))
            test_batch_timer = time.time()

        # Evaluation
        for threshold in args.thresholds:
            pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
            metrics['POD_{:.1f}'.format(threshold)] += pod
            metrics['FAR_{:.1f}'.format(threshold)] += far
            metrics['CSI_{:.1f}'.format(threshold)] += csi
        metrics['MBE'] += evaluation.evaluate_mbe(pred, truth)
        metrics['MAE'] += evaluation.evaluate_mae(pred, truth)
        metrics['RMSE'] += evaluation.evaluate_rmse(pred, truth)
        metrics['SSIM'] += evaluation.evaluate_ssim(pred_norm, truth_norm)
        metrics['KLD'] += evaluation.evaluate_kld(pred, truth)
    
    # Print test loss
    test_loss = test_loss / len(test_loader)
    print('Loss: {:.4f}\tTime: {:.4f}'.format(test_loss, time.time() - test_timer))

    # Save metrics
    for key in metrics.keys():
        if key != 'Step':
            metrics[key] /= len(test_loader)
    df = pd.DataFrame(data=metrics)
    df.to_csv(os.path.join(args.output_path, 'test_metrics.csv'), float_format='%.4f', index=False)
    print('Test metrics saved')


@torch.no_grad()
def predict(model: nn.Module, case_loader: DataLoader):   
    # Predict
    print('\n[Predict]')
    bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
    states = load_checkpoint(bestparams_path, args.device)
    model.load_state_dict(states['model'])
    model.eval()
    for i, (tensor, timestamp) in enumerate(case_loader):
        time_str = datetime.datetime.utcfromtimestampstamp(int(timestamp[0, i]))
        time_str = time_str.strftime('%Y-%m-%d %H:%M:%S')
        print('\nCase {} at {}'.format(i, time_str))
        
        # Forward propagation
        tensor = tensor.to(args.device)
        input_ = tensor[:, :args.input_steps]
        truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
        input_norm = transform.minmax_norm(input_, args.vmax, args.vmin)
        truth_norm = transform.minmax_norm(truth, args.vmax, args.vmin)
        pred_norm = model(input_norm)
        pred = transform.reverse_minmax_norm(pred_norm, args.vmax, args.vmin)
        tensors = (pred, truth, input_)
        
        # Save metrics
        metrics = {}
        metrics['Step'] = np.linspace(1, args.forecast_steps)
        # Evaluation
        for threshold in args.thresholds:
            pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
            metrics['POD_{:.1f}'.format(threshold)] = pod
            metrics['FAR_{:.1f}'.format(threshold)] = far
            metrics['CSI_{:.1f}'.format(threshold)] = csi
        metrics['MBE'] = evaluation.evaluate_mbe(pred, truth)
        metrics['MAE'] = evaluation.evaluate_mae(pred, truth)
        metrics['RMSE'] = evaluation.evaluate_rmse(pred, truth)
        metrics['SSIM'] = evaluation.evaluate_ssim(pred_norm, truth_norm)
        metrics['KLD'] += evaluation.evaluate_kld(pred, truth)
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(args.output_path, 'case_{}_metrics.csv'.format(i)), float_format='%.4f', index=False)
        print('Case {} metrics saved'.format(i))

        # Save tensors and figures
        visualizer.save_tensors(tensors, timestamp, args.output_path, 'case_{}'.format(i))
        print('Tensors saved')
        visualizer.plot_maps(tensors, timestamp, args.output_path, 'case_{}'.format(i))
        print('Figures saved')
    
    print('\nPrediction complete')


if __name__ == '__main__':
    main(args)