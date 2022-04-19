import os

import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
import pytorch_lightning as pl

from models import EncoderForecaster, SmaAt_UNet, AttnUNet
from .discriminator import Discriminator
import utils.scaler as scaler
import models.losses as losses
import utils.evaluation as evaluation
import utils.visualizer as visualizer
import utils.saver as saver


class GAN(pl.LightningModule):
    r"""Deep Generative Adversarial Network.

    Args:
        args (args): Necessary arguments.
    """

    def __init__(self, args):
        super(GAN, self).__init__()
        if args.generator == 'EncoderForecaster':
            self.generator = EncoderForecaster(
                forecast_steps=args.forecast_steps,
                in_channels=args.in_channels, out_channels=args.out_channels,
                hidden_channels=args.hidden_channels
            )
        elif args.generator == 'SmaAt_UNet':
            self.generator = SmaAt_UNet(
                n_channels=args.input_steps, n_classes=args.forecast_steps
            )
        elif args.generator == 'AttnUNet':
            self.generator = AttnUNet(
                input_steps=args.input_steps, forecast_steps=args.forecast_steps, add_noise=True
            )
        
        self.discriminator = Discriminator(
            total_steps= args.input_steps + args.forecast_steps
        )
        
        self.args = args
        self.save_hyperparameters()
        self.init_logger_params()
        self.automatic_optimization = False

    def init_logger_params(self):
        self.train_loss = []
        self.val_loss = []
        self.train_batch_num = round((self.args.end_point - self.args.start_point) * self.args.train_ratio) // self.args.batch_size
        self.val_batch_num = round((self.args.end_point - self.args.start_point) * self.args.valid_ratio) // self.args.batch_size
        self.test_batch_num = ((self.args.end_point - self.args.start_point) - round(
            (self.args.end_point - self.args.start_point) * self.args.train_ratio) - round(
            (self.args.end_point - self.args.start_point) * self.args.valid_ratio)) // self.args.batch_size
        self.total_epochs = np.ceil(self.args.max_iterations / self.train_batch_num).astype(int)
        if not os.path.exists(self.args.output_path):
            os.mkdir(self.args.output_path)

    def show_grad(self):
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(name, '\tmax grad:', torch.max(torch.abs(param.grad)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def configure_optimizers(self):
        g_optimizer = AdamW(self.generator.parameters(), lr=self.args.gen_lr,
                            betas=(self.args.beta1, self.args.beta2), 
                            weight_decay=self.args.weight_decay)
        d_optimizer = AdamW(self.discriminator.parameters(), lr=self.args.disc_lr,
                            betas=(self.args.beta1, self.args.beta2), 
                            weight_decay=self.args.weight_decay)
        return g_optimizer, d_optimizer

    def on_train_epoch_start(self):
        print('\nTraining...')

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # Get input and truth
        data, seconds = batch
        data = data.transpose(1, 0).contiguous()
        seconds = seconds.transpose(1, 0).contiguous()
        input_ = data[:self.args.input_steps]
        truth = data[self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
        input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
        truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)
        g_optimizer, d_optimizer = self.optimizers()

        # Forward step
        pred = self.generator(input_)

        # Discriminator step
        fake_sequence = torch.cat([input_, pred.detach()])
        real_sequence = torch.cat([input_, truth])
        fake_score = self.discriminator(fake_sequence)
        real_score = self.discriminator(real_sequence)
        d_loss = losses.cal_d_loss(fake_score, real_score) * self.args.recon_reg

        # Discriminator backprop
        d_optimizer.zero_grad()
        self.manual_backward(d_loss)
        d_optimizer.step()

        # Generator step
        fake_scores = [self.discriminator(fake_sequence)]
        predictions = [pred]
        for _ in range(self.args.num_sampling - 1):
            pred = self.generator(input_)
            fake_sequence = torch.cat([input_, pred.detach()])
            fake_score = self.discriminator(fake_sequence)
            fake_scores.append(fake_score)
            predictions.append(pred)
        
        fake_score = torch.mean(torch.stack(fake_scores))
        pred = torch.mean(torch.stack(predictions), dim=0)
        g_loss = losses.cal_g_loss(fake_score) \
            + losses.recon_loss(pred, truth) * self.args.recon_reg \
            + losses.cv_loss(pred, truth) * self.args.global_var_reg \
            + losses.ssd_loss(pred, truth) * self.args.local_var_reg
        
        # Generator backprop
        g_optimizer.zero_grad()
        self.manual_backward(g_loss)
        g_optimizer.step()

        results = {
            'train/g_loss': g_loss,
            'train/recon_loss': losses.recon_loss(pred, truth) * self.args.recon_reg, 
            'train/cv_loss': losses.cv_loss(pred, truth) * self.args.global_var_reg, 
            'train/ssd_loss': losses.ssd_loss(pred, truth) * self.args.local_var_reg,
            'train/d_loss': d_loss,
        }
        self.log_dict(results, on_step=True, on_epoch=True, logger=True)

        # print training performance on a batch
        if (batch_idx + 1) % self.args.log_interval == 0:
            print('Epoch: [{}][{}]  Batch: [{}][{}]  G Loss: {:.4f}  D Loss: {:.4f}'
                       .format(self.current_epoch + 1, self.total_epochs, batch_idx + 1, self.train_batch_num, g_loss, d_loss))

        if batch_idx + 1 == self.train_batch_num:
            input_ = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
            pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
            truth = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)
            visualizer.plot_map(input_, pred.detach(), truth, seconds, self.args.output_path,
                                stage='train', lon_range=self.args.lon_range, lat_range=self.args.lat_range)
            saver.save_tensors(input_, pred.detach(), truth, self.args.output_path, stage='train')
        
        loss = torch.tensor([g_loss, d_loss])
        return loss
 
    def on_validation_epoch_start(self):
        print('\nValidate...')

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        # Get input and truth
        data, seconds = batch
        data = data.transpose(1, 0).contiguous()
        seconds = seconds.transpose(1, 0).contiguous()
        input_ = data[:self.args.input_steps]
        truth = data[self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
        input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
        truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)
        
        # Forward step
        pred = self.generator(input_)

        # Discriminator step
        fake_sequence = torch.cat([input_, pred.detach()])
        real_sequence = torch.cat([input_, truth])
        fake_score = self.discriminator(fake_sequence)
        real_score = self.discriminator(real_sequence)
        d_loss = losses.cal_d_loss(fake_score, real_score) * self.args.recon_reg
        
        # Generator step
        g_loss = losses.cal_g_loss(fake_score) \
            + losses.recon_loss(pred, truth) * self.args.recon_reg \
            + losses.cv_loss(pred, truth) * self.args.global_var_reg \
            + losses.ssd_loss(pred, truth) * self.args.local_var_reg

        results = {
            'val/g_loss': g_loss,
            'val/recon_loss': losses.recon_loss(pred, truth) * self.args.recon_reg,
            'val/cv_loss': losses.cv_loss(pred, truth) * self.args.global_var_reg,
            'val/ssd_loss': losses.ssd_loss(pred, truth) * self.args.local_var_reg,
            'val/d_loss': d_loss
        }
        self.log_dict(results, on_epoch=True, logger=True)

        # print training performance on a batch
        if (batch_idx + 1) % self.args.log_interval == 0:
            print('Epoch: [{}][{}]  Batch: [{}][{}]  G Loss: {:.4f}  D Loss: {:.4f}'
                       .format(self.current_epoch + 1, self.total_epochs, batch_idx + 1, self.val_batch_num, g_loss, d_loss))

        if batch_idx + 1 == self.val_batch_num:
            input_ = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
            pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
            truth = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)
            visualizer.plot_map(input_, pred.detach(), truth, seconds, self.args.output_path,
                                stage='val', lon_range=self.args.lon_range, lat_range=self.args.lat_range)
            saver.save_tensors(input_, pred.detach(), truth, self.args.output_path, stage='val')
        
        loss = torch.tensor([g_loss, d_loss])
        return loss

    def validation_epoch_end(self, outputs):
        loss = outputs
        loss = torch.stack(loss).numpy()
        val_loss = np.mean(loss, axis=0).tolist()
        self.val_loss.append(val_loss)
        if not os.path.exists(os.path.join(self.args.output_path, 'loss')):
            os.mkdir(os.path.join(self.args.output_path, 'loss'))
        np.savetxt(os.path.join(self.args.output_path, 'loss', 'val_loss.csv'),
                   np.array(self.val_loss), fmt='%.8f', delimiter=',')

    def training_epoch_end(self, outputs):
        loss = [output['loss'] for output in outputs]
        loss = torch.stack(loss).numpy()
        train_loss = np.mean(loss, axis=0).tolist()
        self.train_loss.append(train_loss)
        if not os.path.exists(os.path.join(self.args.output_path, 'loss')):
            os.mkdir(os.path.join(self.args.output_path, 'loss'))
        np.savetxt(os.path.join(self.args.output_path, 'loss', 'train_loss.csv'),
                   np.array(self.train_loss), fmt='%.8f', delimiter=',')
        
        visualizer.plot_loss(np.array(self.train_loss)[:, 0], np.array(self.val_loss)[:, 0],
                             root=self.args.output_path, title='Generator Loss', filename='generator_loss.png')
        visualizer.plot_loss(np.array(self.train_loss)[:, 1], np.array(self.val_loss)[:, 1],
                             root=self.args.output_path, title='Discriminator Loss', filename='discriminator_loss.png')

    def on_test_epoch_start(self):
        print('\nTest...')

    def test_step(self, batch, batch_idx):
        # Get input and truth
        data, seconds = batch
        data = data.transpose(1, 0).contiguous()
        seconds = seconds.transpose(1, 0).contiguous()
        input_ = data[:self.args.input_steps]
        truth = data[self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
        input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
        truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)
        
        # Forward step
        pred = self.generator(input_)

        # Calculate metrics
        pred_rev = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
        truth_rev = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)
        metrics = {}
        for threshold in self.args.thresholds:
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
        
        # print training performance on a batch
        if (batch_idx + 1) % self.args.log_interval == 0:
            print('Epoch: [{}][{}]  Batch: [{}][{}]'
                  .format(1, 1, batch_idx + 1, self.test_batch_num))
        
        if batch_idx + 1 == self.test_batch_num:
            input_ = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
            pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
            truth = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)
            visualizer.plot_map(input_, pred.detach(), truth, seconds, self.args.output_path,
                                stage='test', lon_range=self.args.lon_range, lat_range=self.args.lat_range)
            saver.save_tensors(input_, pred.detach(), truth, self.args.output_path, stage='test')

        return metrics

    def test_epoch_end(self, outputs):
        test_metrics = {}
        for key in outputs[0].keys():
            test_metrics[key] = []
        for key in test_metrics.keys():
            for metrics in outputs:
                test_metrics[key].append(metrics[key])
        for key in test_metrics.keys():
            test_metrics[key] = np.mean(test_metrics[key], axis=0)
        
        if not os.path.exists(os.path.join(self.args.output_path, 'metrics')):
            os.mkdir(os.path.join(self.args.output_path, 'metrics'))
                       
        df = pd.DataFrame(data=test_metrics)
        df.to_csv(os.path.join(self.args.output_path, 'metrics', 'test_metrics.csv'), float_format='%.8f')
        
    def on_predict_start(self):
        print('\nInference...')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Get input and truth
        data, seconds = batch
        data = data.transpose(1, 0).contiguous()
        seconds = seconds.transpose(1, 0).contiguous()
        input_ = data[:self.args.input_steps]
        truth = data[self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
        input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
        truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)
        
        # Forward step
        pred = self.generator(input_)

        # Calculate metrics
        pred_rev = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
        truth_rev = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)
        metrics = {}
        for threshold in self.args.thresholds:
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

        if not os.path.exists(os.path.join(self.args.output_path, 'metrics')):
            os.mkdir(os.path.join(self.args.output_path, 'metrics'))
        
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'metrics', 'sample_metrics.csv'), float_format='%.8f')
        
        input_ = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
        pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
        truth = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)
        visualizer.plot_map(input_, pred.detach(), truth, seconds, self.args.output_path,
                            stage='sample', lon_range=self.args.lon_range, lat_range=self.args.lat_range)
        saver.save_tensors(input_, pred.detach(), truth, self.args.output_path, stage='sample')


    def on_predict_end(self):
        print('Inference done.')
