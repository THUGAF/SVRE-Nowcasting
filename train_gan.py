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
from torch.nn.utils import clip_grad_norm_
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
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--valid-ratio', type=float, default=0.1)
parser.add_argument('--case-indices', type=int, nargs='+', default=[0])

# model settings
parser.add_argument('--num-ensembles', type=int, default=6)

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
    generator = models.AN(args.input_steps, args.forecast_steps, add_noise=True).to(args.device)
    discriminator = models.Discriminator(args.input_steps + args.forecast_steps).to(args.device)
    count_params(generator, discriminator)
    optimizer_g = optim.Adam(generator.parameters(), args.learning_rate,
                             betas=(args.beta1, args.beta2),
                             weight_decay=args.weight_decay)
    optimizer_d = optim.Adam(discriminator.parameters(), args.learning_rate / 2,
                             betas=(args.beta1, args.beta2),
                             weight_decay=args.weight_decay)
    
    # Make dir
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Train, test, and predict
    print('\n### Start tasks ###')
    if args.train or args.test:
        train_loader, val_loader, test_loader = dataloader.load_data(args.data_path, 
            args.input_steps, args.forecast_steps, args.batch_size, args.num_workers, 
            args.train_ratio, args.valid_ratio, args.x_range, args.y_range)
    if args.train:
        train(generator, discriminator, optimizer_g, optimizer_d, train_loader, val_loader)
    if args.test:
        test(generator, test_loader)
    if args.predict:
        case_loader = dataloader.load_case(args.data_path, args.case_indices, args.input_steps,
                                           args.forecast_steps, args.x_range, args.y_range)
        predict(generator, case_loader)

    print('\n### All tasks complete ###')


def count_params(generator: nn.Module, discriminator: nn.Module):
    G_params = filter(lambda p: p.requires_grad, generator.parameters())
    D_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    num_G_params = sum([p.numel() for p in G_params])
    num_D_params = sum([p.numel() for p in D_params])
    print('\nModel name: {}'.format(type(generator).__name__))
    print('G params: {}'.format(num_G_params))
    print('D params: {}'.format(num_D_params))
    print('Total params: {}'.format(num_G_params + num_D_params))


def save_checkpoint(filename: str, current_iteration: int, train_loss_g: list, train_loss_d: list, 
                    val_loss_g: list, val_loss_d: list, val_score: list,
                    generator: nn.Module, discriminator: nn.Module, 
                    optimizer_g: optim.Optimizer, optimizer_d: optim.Optimizer):
    states = {
        'iteration': current_iteration,
        'train_loss_g': train_loss_g,
        'train_loss_d': train_loss_d,
        'val_loss_g': val_loss_g,
        'val_loss_d': val_loss_d,
        'val_score': val_score,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict()
    }
    torch.save(states, filename)


def load_checkpoint(filename: str, device: str):
    states = torch.load(filename, map_location=device)
    return states


def early_stopping(score: list, patience: int = 10):
    early_stopping_flag = False
    counter = 0
    current_epoch = len(score)
    if current_epoch == 1:
        min_score = np.inf
    else:
        min_score = min(score[:-1])
    if min_score > score[-1]:
        print('Metric decreased: {:.4f} --> {:.4f}'.format(min_score, score[-1]))
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
        shutil.copyfile(checkpoint_path, bestparams_path)
    else:
        min_score_epoch = score.index(min(score))
        if current_epoch > min_score_epoch:
            counter = current_epoch - min_score_epoch
            print('EarlyStopping counter: {} out of {}'.format(counter, patience))
            if counter == patience:
                early_stopping_flag = True
    return early_stopping_flag


def weighted_l1_loss(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    points = torch.tensor([10.0, 20.0, 30.0, 40.0])
    points = transform.minmax_norm(points)
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


def d_loss(fake_score: torch.Tensor, real_score: torch.Tensor, 
           loss_func: nn.Module = nn.MSELoss()) -> torch.Tensor:
    label = torch.ones_like(fake_score).type_as(fake_score)
    loss_pred = loss_func(fake_score, label * 0.0)
    loss_truth = loss_func(real_score, label * 1.0)
    d_loss = loss_pred + loss_truth
    return d_loss


def g_loss(fake_score: torch.Tensor, loss_func: nn.Module = nn.MSELoss()) -> torch.Tensor:
    label = torch.ones_like(fake_score).type_as(fake_score)
    g_loss = loss_func(fake_score, label * 1.0)
    return g_loss


def clip_weight(model: nn.Module, bound: float):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param = torch.clip(param, max=bound, min=-bound)
            setattr(model, name, param)


def train(generator: nn.Module, discriminator: nn.Module, optimizer_g: optim.Optimizer, 
          optimizer_d: optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader):
    # Pretrain
    if args.pretrain:
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        states = load_checkpoint(checkpoint_path, args.device)
        current_iteration = states['iteration']
        train_loss_g = states['train_loss_g']
        train_loss_d = states['train_loss_d']
        val_loss_g = states['val_loss_g']
        val_loss_d = states['val_loss_d']
        val_score = states['val_score']
        generator.load_state_dict(states['generator'])
        discriminator.load_state_dict(states['discriminator'])
        optimizer_g.load_state_dict(states['optimizer_g'])
        optimizer_d.load_state_dict(states['optimizer_d'])
        start_epoch = int(np.floor(current_iteration / len(train_loader)))
    else:
        current_iteration = 0
        train_loss_g = []
        train_loss_d = []
        val_loss_g = []
        val_loss_d = []
        val_score = []
        start_epoch = 0

    # Train and validation
    total_epochs = int(np.ceil((args.max_iterations - current_iteration) / len(train_loader)))
    print('\nMax iterations:', args.max_iterations)
    print('Total epochs:', total_epochs)

    for epoch in range(start_epoch, total_epochs):
        train_loss_g_epoch = 0
        train_loss_d_epoch = 0
        val_loss_g_epoch = 0
        val_loss_d_epoch = 0
        val_score_epoch = 0

        # Train
        print('\n[Train]')
        print('Epoch: [{}][{}]'.format(epoch + 1, total_epochs))
        generator.train()
        discriminator.train()

        # Timers
        train_epoch_timer = time.time()
        train_batch_timer = time.time()

        for i, (tensor, _) in enumerate(train_loader):
            # Check max iterations
            current_iteration += 1
            if current_iteration > args.max_iterations:
                print('Max iterations reached. Exit!')
                break
            
            # Forward progation
            tensor = tensor.to(args.device)
            input_ = tensor[:, :args.input_steps]
            truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
            input_norm = transform.minmax_norm(input_)
            truth_norm = transform.minmax_norm(truth)
            preds_norm = [generator(input_norm) for _ in range(args.num_ensembles)]
            
            # Discriminator backward propagation
            real_score = discriminator(tensor)
            fake_scores = [discriminator(torch.cat([input_norm, pred_norm.detach()], dim=1)) 
                           for pred_norm in preds_norm]
            fake_score = torch.mean(torch.stack(fake_scores), dim=0)
            loss_d = d_loss(fake_score, real_score)
            optimizer_d.zero_grad()
            loss_d.backward()
            clip_grad_norm_(discriminator.parameters(), 0.005)
            optimizer_d.step()
            clip_weight(discriminator, 0.005)

            # Generator backward propagation
            fake_scores = [discriminator(torch.cat([input_norm, pred_norm], dim=1)) 
                           for pred_norm in preds_norm]
            fake_score = torch.mean(torch.stack(fake_scores), dim=0)
            pred_norm = torch.mean(torch.stack(preds_norm), dim=0)
            loss_g = g_loss(fake_score) + \
                args.weight_svre * svre_loss(pred_norm, truth_norm) + \
                args.weight_recon * weighted_l1_loss(pred_norm, truth_norm)
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            # Record and print loss
            train_loss_g_epoch += loss_g.item()
            train_loss_d_epoch += loss_d.item()
            if (i + 1) % args.display_interval == 0:
                print('Epoch: [{}][{}]\tBatch: [{}][{}]\tLoss G: {:.4f}\tLoss D: {:.4f}\tTime: {:.4f}'.format(
                    epoch + 1, total_epochs, i + 1, len(train_loader), loss_g.item(), loss_d.item(),
                    time.time() - train_batch_timer))
                train_batch_timer = time.time()
            
        # Save train loss
        train_loss_g_epoch = train_loss_g_epoch / len(train_loader)
        train_loss_d_epoch = train_loss_d_epoch / len(train_loader)
        print('Epoch: [{}][{}]\tLoss G: {:.4f}\tLoss D: {:.4f}\tTime: {:.4f}'.format(
            epoch + 1, total_epochs, train_loss_g_epoch, train_loss_d_epoch, time.time() - train_epoch_timer))
        train_epoch_timer = time.time()
        train_loss_g.append(train_loss_g_epoch)
        train_loss_d.append(train_loss_d_epoch)
        np.savetxt(os.path.join(args.output_path, 'train_loss_g.txt'), train_loss_g)
        np.savetxt(os.path.join(args.output_path, 'train_loss_d.txt'), train_loss_d)
        print('Train loss saved')

        # Validate
        print('\n[Validate]')
        print('Epoch: [{}][{}]'.format(epoch + 1, total_epochs))
        generator.eval()
        discriminator.eval()

        # Timers
        val_epoch_timer = time.time()
        val_batch_timer = time.time()

        with torch.no_grad():
            for i, (tensor, _) in enumerate(val_loader):
                # Forward progation
                tensor = tensor.to(args.device)
                input_ = tensor[:, :args.input_steps]
                truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
                input_norm = transform.minmax_norm(input_)
                truth_norm = transform.minmax_norm(truth)
                preds_norm = [generator(input_norm) for _ in range(args.num_ensembles)]
                real_score = discriminator(tensor)
                fake_scores = [discriminator(torch.cat([input_norm, pred_norm], dim=1)) 
                               for pred_norm in preds_norm]
                fake_score = torch.mean(torch.stack(fake_scores), dim=0)
                loss_d = d_loss(fake_score, real_score)
                pred_norm = torch.mean(torch.stack(preds_norm), dim=0)
                loss_g = g_loss(fake_score) + \
                    args.weight_svre * svre_loss(pred_norm, truth_norm) + \
                    args.weight_recon * weighted_l1_loss(pred_norm, truth_norm)
                pred = transform.inverse_minmax_norm(pred_norm)
                score = evaluation.evaluate_forecast(pred, truth, args.thresholds[-1])[2]

                # Record and print loss
                val_loss_g_epoch += loss_g.item()
                val_loss_d_epoch += loss_d.item()
                val_score_epoch += score
                if (i + 1) % args.display_interval == 0:
                    print('Epoch: [{}][{}]\tBatch: [{}][{}]\tLoss G: {:.4f}\tLoss D: {:.4f}\tTime: {:.4f}'.format(
                        epoch + 1, total_epochs, i + 1, len(val_loader), loss_g.item(), loss_d.item(),
                        time.time() - val_batch_timer))
                    val_batch_timer = time.time()
            
        # Save val loss
        val_loss_g_epoch = val_loss_g_epoch / len(val_loader)
        val_loss_d_epoch = val_loss_d_epoch / len(val_loader)
        val_score_epoch = val_score_epoch / len(val_loader)
        print('Epoch: [{}][{}]\tLoss G: {:.4f}\tLoss D: {:.4f}\tTime: {:.4f}'.format(
            epoch + 1, total_epochs, val_loss_g_epoch, val_loss_d_epoch, 
            time.time() - val_epoch_timer))
        val_epoch_timer = time.time()
        val_loss_g.append(val_loss_g_epoch)
        val_loss_d.append(val_loss_d_epoch)
        val_score.append(val_score_epoch)
        np.savetxt(os.path.join(args.output_path, 'val_loss_g.txt'), val_loss_g)
        np.savetxt(os.path.join(args.output_path, 'val_loss_d.txt'), val_loss_d)
        np.savetxt(os.path.join(args.output_path, 'val_score.txt'), val_score)
        print('Val loss saved')

        # Plot loss
        visualizer.plot_loss(train_loss_g, val_loss_g, os.path.join(args.output_path, 'loss_g.png'))
        visualizer.plot_loss(train_loss_d, val_loss_d, os.path.join(args.output_path, 'loss_d.png'))
        print('Loss figure saved')

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        save_checkpoint(checkpoint_path, current_iteration, train_loss_g, train_loss_d, val_loss_g, val_loss_d, 
                        val_score, generator, discriminator, optimizer_g, optimizer_d)
        if args.early_stopping:
            early_stopping_flag = early_stopping([-v for v in val_score])
            if early_stopping_flag:
                print('Early stopped')
                break


@torch.no_grad()
def test(generator: nn.Module, test_loader: DataLoader):
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

    # Test
    print('\n[Test]')
    bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
    states = load_checkpoint(bestparams_path, args.device)
    generator.load_state_dict(states['generator'])
    generator.eval()

    # Timer
    test_timer = time.time()
    test_batch_timer = time.time()

    for i, (tensor, _) in enumerate(test_loader):
        # Forward propagation
        tensor = tensor.to(args.device)
        input_ = tensor[:, :args.input_steps]
        truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
        input_norm = transform.minmax_norm(input_)
        truth_norm = transform.minmax_norm(truth)
        truth_R = transform.ref_to_R(truth)
        preds_norm = [generator(input_norm) for _ in range(args.num_ensembles)]
        pred_norm = torch.mean(torch.stack(preds_norm), dim=0)
        pred = transform.inverse_minmax_norm(pred_norm)
        truth_R = transform.ref_to_R(truth)
        pred_R = transform.ref_to_R(pred)

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
        
        # Record and print time
        if (i + 1) % args.display_interval == 0:
            print('Batch: [{}][{}]\tTime: {:.4f}'.format(
                i + 1, len(test_loader), time.time() - test_batch_timer))
            test_batch_timer = time.time()
    
    # Print time
    print('Time: {:.4f}'.format(time.time() - test_timer))

    # Save metrics
    for key in metrics.keys():
        metrics[key] = metrics[key] / len(test_loader)
    df = pd.DataFrame(data=metrics, index=[0])
    df.to_csv(os.path.join(args.output_path, 'test_metrics.csv'), 
              float_format='%.6f', index=False)
    print('Test metrics saved')


@torch.no_grad()
def predict(generator: nn.Module, case_loader: DataLoader):
    # Init metric dict
    metrics = {}

    # Predict
    print('\n[Predict]')
    bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
    states = load_checkpoint(bestparams_path, args.device)
    generator.load_state_dict(states['generator'])
    generator.eval()

    for i, (tensor, timestamp) in enumerate(case_loader):
        time_str = datetime.datetime.utcfromtimestamp(int(timestamp[0, i]))
        time_str = time_str.strftime('%Y-%m-%d %H:%M:%S')
        print('\nCase {} at {}'.format(i, time_str))
        
        # Forward propagation
        tensor = tensor.to(args.device)
        input_ = tensor[:, :args.input_steps]
        truth = tensor[:, args.input_steps: args.input_steps + args.forecast_steps]
        input_norm = transform.minmax_norm(input_)
        truth_norm = transform.minmax_norm(truth)
        preds_norm = [generator(input_norm) for _ in range(args.num_ensembles)]
        pred_norm = torch.mean(torch.stack(preds_norm), dim=0)
        pred = transform.inverse_minmax_norm(pred_norm)
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
        metrics['SSIM'] = evaluation.evaluate_ssim(pred_norm, truth_norm)
        metrics['JSD'] = evaluation.evaluate_jsd(pred, truth)
            
        # Save metrics
        for key in metrics.keys():
            metrics[key] = metrics[key]
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
