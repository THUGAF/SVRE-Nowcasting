import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.transform as transform
import utils.losses as losses


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# refers to https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'       
        """
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, trainer):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, trainer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, trainer)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, trainer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        trainer.save_checkpoint(self.path)
        self.val_loss_min = val_loss


class NNTrainer:
    def __init__(self, args):
        self.args = args

    def fit(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.model.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if self.args.pretrain:
            start_iterations = self.load_checkpoint()['iteration']
        else:
            start_iterations = 0
        self.total_epochs = int(math.ceil((self.args.max_iterations - start_iterations) / len(train_loader)))
        
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr,
                              betas=(self.args.beta1, self.args.beta2),
                              weight_decay=self.args.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.5)

        if self.args.train:
            self.train()
        if self.args.test:
            self.test()

    def train(self):
        self.count_params()
        # Pretrain: Load model and optimizer
        if self.args.pretrain:
            states = self.load_checkpoint()
            self.model.load_state_dict(states['model'])
            self.optimizer.load_state_dict(states['optimizer'])
            self.current_iterations = states['iteration']
            self.train_loss = states['train_loss']
            self.val_loss = states['val_loss']
            start_epoch = int(math.floor(self.current_iterations / len(self.train_loader)))
        else:
            self.current_iterations = 0
            self.train_loss = []
            self.val_loss = []
            start_epoch = 0
        
        early_stopping = EarlyStopping(verbose=True, path='bestmodel.pt')

        for epoch in range(start_epoch, self.total_epochs):
            print('\n[Train]')
            print('Epoch: [{}][{}]'.format(epoch + 1, self.total_epochs))
            train_loss = []
            val_loss = []

            # Train
            self.model.train()
            for i, (tensor, timestamp) in enumerate(self.train_loader):
                # Check max iterations
                self.current_iterations += 1
                if self.current_iterations > self.args.max_iterations:
                    print('Max interations %d reached.' % self.args.max_iterations)
                    break

                tensor = tensor.to(self.args.device)
                timestamp = timestamp.to(self.args.device)
                input_ = tensor[:, :self.args.input_steps]
                truth = tensor[:, self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
                input_norm = transform.minmax_norm(input_, self.args.vmax, self.args.vmin)
                truth_norm = transform.minmax_norm(truth, self.args.vmax, self.args.vmin)
                pred_norm = self.model(input_norm)

                # Backward propagation
                loss = self.args.lambda_rec * losses.biased_mae_loss(pred_norm, truth_norm, self.args.vmax, self.args.vmin) \
                    + self.args.lambda_var * losses.cv_loss(pred_norm, truth_norm)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_loss.append(loss.item())
                if (i + 1) % self.args.display_interval == 0:
                    print('Epoch: [{}][{}] Batch: [{}][{}] Loss: {:.4f}'.format(
                        epoch + 1, self.total_epochs, i + 1, len(self.train_loader), loss.item()
                    ))
            
            self.train_loss.append(np.mean(train_loss))
            print('Epoch: [{}][{}] Loss: {:.4f}'.format(epoch + 1, self.total_epochs, self.train_loss[-1]))
            np.savetxt(os.path.join(self.args.output_path, 'train_loss.txt'), self.train_loss)

            print('\nVisualizing...')
            pred = transform.reverse_minmax_norm(pred_norm, self.args.vmax, self.args.vmin)
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'train')
            print('Visualization complete')

            # Validate
            print('\n[Val]')
            self.model.eval()
            with torch.no_grad():
                for i, (tensor, timestamp) in enumerate(self.val_loader):
                    tensor = tensor.to(self.args.device)
                    timestamp = timestamp.to(self.args.device)
                    input_ = tensor[:, :self.args.input_steps]
                    truth = tensor[:, self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
                    input_norm = transform.minmax_norm(input_, self.args.vmax, self.args.vmin)
                    truth_norm = transform.minmax_norm(truth, self.args.vmax, self.args.vmin)
                    pred_norm = self.model(input_norm)
                    loss = self.args.lambda_rec * losses.biased_mae_loss(pred_norm, truth_norm, self.args.vmax, self.args.vmin) \
                        + self.args.lambda_var * losses.cv_loss(pred_norm, truth_norm)
                    val_loss.append(loss.item())
                    if (i + 1) % self.args.display_interval == 0:
                        print('Epoch: [{}][{}] Batch: [{}][{}] Loss: {:.4f}'.format(
                            epoch + 1, self.total_epochs, i + 1, len(self.val_loader), loss.item()
                        ))
                            
            self.val_loss.append(np.mean(val_loss))    
            print('Epoch: [{}][{}] Loss: {:.4f}'.format(epoch + 1, self.total_epochs, self.val_loss[-1]))        
            np.savetxt(os.path.join(self.args.output_path, 'val_loss.txt'), self.val_loss)

            print('\nVisualizing...')
            pred = transform.reverse_minmax_norm(pred_norm, self.args.vmax, self.args.vmin)
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'val')
            visualizer.plot_loss(self.train_loss, self.val_loss, self.args.output_path)
            print('Visualization complete')

            # Save checkpoint
            self.save_checkpoint()
            if self.args.early_stopping:
                early_stopping(self.val_loss[-1], self)
            if early_stopping.early_stop:
                break
    
    @torch.no_grad()
    def test(self):
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        for threshold in self.args.thresholds:
            metrics['POD-%ddBZ' % threshold] = []
            metrics['FAR-%ddBZ' % threshold] = []
            metrics['CSI-%ddBZ' % threshold] = []
        metrics['ME'] = []
        metrics['MAE'] = []
        metrics['SSIM'] = []
        metrics['KLD'] = []
        test_loss = []
        
        print('\n[Test]')
        self.model.load_state_dict(self.load_checkpoint('bestmodel.pt')['model'])
        self.model.eval()
        self.count_params()

        for i, (tensor, timestamp) in enumerate(self.test_loader):
            tensor = tensor.to(self.args.device)
            timestamp = timestamp.to(self.args.device)
            input_ = tensor[:, :self.args.input_steps]
            truth = tensor[:, self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
            input_norm = transform.minmax_norm(input_, self.args.vmax, self.args.vmin)
            truth_norm = transform.minmax_norm(truth, self.args.vmax, self.args.vmin)
            pred_norm = self.model(input_norm)
            loss = self.args.lambda_rec * losses.biased_mae_loss(pred_norm, truth_norm, self.args.vmax, self.args.vmin) \
                + self.args.lambda_var * losses.cv_loss(pred_norm, truth_norm)
            test_loss.append(loss.item())
            if (i + 1) % self.args.display_interval == 0:
                print('Batch: [{}][{}] Loss: {:.4f}'.format(i + 1, len(self.test_loader), loss.item()))

            pred = transform.reverse_minmax_norm(pred_norm, self.args.vmax, self.args.vmin)
            for threshold in self.args.thresholds:
                pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
                metrics['POD-%ddBZ' % threshold].append(pod)
                metrics['FAR-%ddBZ' % threshold].append(far)
                metrics['CSI-%ddBZ' % threshold].append(csi)
            metrics['ME'].append(evaluation.evaluate_me(pred, truth))
            metrics['MAE'].append(evaluation.evaluate_mae(pred, truth))
            metrics['SSIM'].append(evaluation.evaluate_ssim(pred_norm, truth_norm))
            metrics['KLD'].append(evaluation.evaluate_kld(pred, truth))
        
        print('Loss: {:.4f}'.format(np.mean(test_loss)))
        
        print('\nEvaluating...')
        for key in metrics.keys():
            if key != 'Time':
                metrics[key] = np.mean(metrics[key], axis=0)
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'test_metrics.csv'), float_format='%.4g', index=False)
        print('Evaluation complete')
        
        print('\nVisualizing...')
        visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'test')
        print('Visualization complete')
        print('\nTest complete')

    @torch.no_grad()
    def predict(self, model, sample_loader):
        print('\n[Predict]')
        self.model = model
        self.model.to(self.args.device)
        self.model.load_state_dict(self.load_checkpoint('bestmodel.pt')['model'])
        self.model.eval()
        self.count_params()

        for i, (tensor, timestamp) in enumerate(sample_loader):
            print('\nSample {}'.format(i))
            metrics = {}
            metrics['Time'] = np.linspace(6, 60, 10)
            tensor = tensor.to(self.args.device)
            timestamp = timestamp.to(self.args.device)
            input_ = tensor[:, :self.args.input_steps]
            truth = tensor[:, self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
            input_norm = transform.minmax_norm(input_, self.args.vmax, self.args.vmin)
            truth_norm = transform.minmax_norm(truth, self.args.vmax, self.args.vmin)
            pred_norm = self.model(input_norm)
            pred = transform.reverse_minmax_norm(pred_norm, self.args.vmax, self.args.vmin)

            print('\nEvaluating...')
            for threshold in self.args.thresholds:
                pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
                metrics['POD-%ddBZ' % threshold] = pod
                metrics['FAR-%ddBZ' % threshold] = far
                metrics['CSI-%ddBZ' % threshold] = csi
            metrics['ME'] = evaluation.evaluate_me(pred, truth)
            metrics['MAE'] = evaluation.evaluate_mae(pred, truth)
            metrics['SSIM'] = evaluation.evaluate_ssim(pred_norm, truth_norm)
            metrics['KLD'] = evaluation.evaluate_kld(pred, truth)
                    
            df = pd.DataFrame(data=metrics)
            df.to_csv(os.path.join(self.args.output_path, 'sample_{}_metrics.csv'.format(i)), float_format='%.4g', index=False)
            print('Evaluation complete')

            print('\nVisualizing...')
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'sample_{}'.format(i))
            visualizer.plot_psd(pred, truth, self.args.output_path, 'sample_{}'.format(i))
            print('Visualization complete')
        
        print('\nPrediction complete')

    def save_checkpoint(self, filename='checkpoint.pt'):
        states = {
            'iteration': self.current_iterations,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(states, os.path.join(self.args.output_path, filename))

    def load_checkpoint(self, filename='checkpoint.pt'):
        states = torch.load(os.path.join(self.args.output_path, filename), map_location=self.args.device)
        return states
    
    def count_params(self):
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        num_params = sum([p.numel() for p in model_params])
        print('\nModel name: {}'.format(type(self.model).__name__))
        print('Total params: {}'.format(num_params))


class GANTrainer:
    def __init__(self, args):
        self.args = args

    def fit(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.model.generator.to(self.args.device)
        self.model.discriminator.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if self.args.pretrain:
            start_iterations = self.load_checkpoint()['iteration']
        else:
            start_iterations = 0
        self.total_epochs = int(math.ceil((self.args.max_iterations - start_iterations) / len(train_loader)))
        
        self.optimizer_g = Adam(self.model.generator.parameters(), lr=self.args.lr,
                                betas=(self.args.beta1, self.args.beta2),
                                weight_decay=self.args.weight_decay)
        self.optimizer_d = Adam(self.model.discriminator.parameters(), lr=self.args.lr / 2,
                                betas=(self.args.beta1, self.args.beta2),
                                weight_decay=1)
        self.scheduler_g = StepLR(self.optimizer_g, step_size=2000, gamma=0.5)
        self.scheduler_d = StepLR(self.optimizer_d, step_size=2000, gamma=0.5)

        if self.args.train:
            self.train()
        if self.args.test:
            self.test()
    
    def train(self):
        self.count_params()
        # Pretrain: Load model and optimizer
        if self.args.pretrain:
            states = self.load_checkpoint()
            self.model.generator.load_state_dict(states['model'])
            self.model.discriminator.load_state_dict(states['discriminator'])
            self.optimizer_g.load_state_dict(states['optimizer_g'])
            self.optimizer_d.load_state_dict(states['optimizer_d'])
            self.current_iterations = states['iteration']
            self.train_loss_g = states['train_loss_g']
            self.train_loss_d = states['train_loss_d']
            self.val_loss_g = states['val_loss_g']
            self.val_loss_d = states['val_loss_d']
            start_epoch = int(math.floor(self.current_iterations / len(self.train_loader)))
        else:
            self.current_iterations = 0
            self.train_loss_g = []
            self.train_loss_d = []
            self.val_loss_g = []
            self.val_loss_d = []
            start_epoch = 0
        
        early_stopping = EarlyStopping(verbose=True, path='bestmodel.pt')

        for epoch in range(start_epoch, self.total_epochs):
            print('\n[Train]')
            print('Epoch: [{}][{}]'.format(epoch + 1, self.total_epochs))
            train_loss_g = []
            train_loss_d = []
            val_loss_g = []
            val_loss_d = []

            # Train
            self.model.train()
            for i, (tensor, timestamp) in enumerate(self.train_loader):
                # Check max iterations
                self.current_iterations += 1
                if self.current_iterations > self.args.max_iterations:
                    print('Max interations %d reached.' % self.args.max_iterations)
                    break

                tensor = tensor.to(self.args.device)
                timestamp = timestamp.to(self.args.device)
                input_ = tensor[:, :self.args.input_steps]
                truth = tensor[:, self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
                input_norm = transform.minmax_norm(input_, self.args.vmax, self.args.vmin)
                truth_norm = transform.minmax_norm(truth, self.args.vmax, self.args.vmin)
                preds_norm = [self.model(input_norm) for _ in range(self.args.ensemble_members)]
                real_score = self.model.discriminator(tensor)
                fake_scores = [self.model.discriminator(torch.cat([input_norm, pred_norm.detach()], dim=1)) for pred_norm in preds_norm]
                
                # Backward propagation
                fake_score = torch.mean(torch.stack(fake_scores), dim=0)
                loss_d = losses.cal_d_loss(fake_score, real_score)
                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()
                self.scheduler_d.step()

                fake_scores = [self.model.discriminator(torch.cat([input_norm, pred_norm], dim=1)) for pred_norm in preds_norm]
                fake_score = torch.mean(torch.stack(fake_scores), dim=0)
                pred_norm = torch.mean(torch.stack(preds_norm), dim=0)
                loss_g = losses.cal_g_loss(fake_score) + \
                    self.args.lambda_var * losses.cv_loss(pred_norm, truth_norm) + \
                    self.args.lambda_rec * losses.biased_mae_loss(pred_norm, truth_norm, self.args.vmax, self.args.vmin)  
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()
                self.scheduler_g.step()

                train_loss_g.append(loss_g.item())
                train_loss_d.append(loss_d.item())
                if (i + 1) % self.args.display_interval == 0:
                    print('Epoch: [{}][{}] Batch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                        epoch + 1, self.total_epochs, i + 1, len(self.train_loader), loss_g.item(), loss_d.item()
                    ))
            
            self.train_loss_g.append(np.mean(train_loss_g))
            self.train_loss_d.append(np.mean(train_loss_d))
            print('Epoch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                epoch + 1, self.total_epochs, self.train_loss_g[-1], self.train_loss_d[-1]))
            
            np.savetxt(os.path.join(self.args.output_path, 'train_loss_g.txt'), self.train_loss_g)
            np.savetxt(os.path.join(self.args.output_path, 'train_loss_d.txt'), self.train_loss_d)

            print('\nVisualizing...')
            pred = transform.reverse_minmax_norm(pred_norm, self.args.vmax, self.args.vmin)
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'train')
            print('Visualization complete')
           
            # Validate
            print('\n[Val]')
            self.model.eval()
            with torch.no_grad():
                for i, (tensor, timestamp) in enumerate(self.val_loader):
                    tensor = tensor.to(self.args.device)
                    timestamp = timestamp.to(self.args.device)
                    input_ = tensor[:, :self.args.input_steps]
                    truth = tensor[:, self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
                    input_norm = transform.minmax_norm(input_, self.args.vmax, self.args.vmin)
                    truth_norm = transform.minmax_norm(truth, self.args.vmax, self.args.vmin)
                    
                    preds_norm = [self.model(input_norm) for _ in range(self.args.ensemble_members)]
                    real_score = self.model.discriminator(tensor)
                    fake_scores = [self.model.discriminator(torch.cat([input_norm, pred_norm], dim=1)) for pred_norm in preds_norm]
                    fake_score = torch.mean(torch.stack(fake_scores), dim=0)
                    loss_d = losses.cal_d_loss(fake_score, real_score)

                    pred_norm = torch.mean(torch.stack(preds_norm), dim=0)
                    loss_g = losses.cal_g_loss(fake_score) + \
                        self.args.lambda_var * losses.cv_loss(pred_norm, truth_norm) + \
                        self.args.lambda_rec * losses.biased_mae_loss(pred_norm, truth_norm, self.args.vmax, self.args.vmin)

                    val_loss_g.append(loss_g.item())
                    val_loss_d.append(loss_d.item())
                    if (i + 1) % self.args.display_interval == 0:
                        print('Epoch: [{}][{}] Batch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                            epoch + 1, self.total_epochs, i + 1, len(self.val_loader), loss_g.item(), loss_d.item()
                        ))
                            
            self.val_loss_g.append(np.mean(val_loss_g))
            self.val_loss_d.append(np.mean(val_loss_d))
            print('Epoch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                epoch + 1, self.total_epochs, self.val_loss_g[-1], self.val_loss_d[-1]))

            np.savetxt(os.path.join(self.args.output_path, 'val_loss_g.txt'), self.val_loss_g)
            np.savetxt(os.path.join(self.args.output_path, 'val_loss_d.txt'), self.val_loss_d)

            print('\nVisualizing...')
            pred = transform.reverse_minmax_norm(pred_norm, self.args.vmax, self.args.vmin)
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'val')
            visualizer.plot_loss(self.train_loss_g, self.val_loss_g, self.args.output_path, 'loss_g.png')
            visualizer.plot_loss(self.train_loss_d, self.val_loss_d, self.args.output_path, 'loss_d.png')
            print('Visualization complete')

            # Save checkpoint
            self.save_checkpoint()

            if self.args.early_stopping:
                early_stopping(self.val_loss_g[-1], self)
            if early_stopping.early_stop:
                break

    @torch.no_grad()
    def test(self):
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        for threshold in self.args.thresholds:
            metrics['POD-%ddBZ' % threshold] = []
            metrics['FAR-%ddBZ' % threshold] = []
            metrics['CSI-%ddBZ' % threshold] = []
        metrics['ME'] = []
        metrics['MAE'] = []
        metrics['SSIM'] = []
        metrics['KLD'] = []
        test_loss_g = []
        test_loss_d = []
        
        print('\n[Test]')
        self.model.generator.load_state_dict(self.load_checkpoint('bestmodel.pt')['model'])
        self.model.eval()
        self.count_params()

        for i, (tensor, timestamp) in enumerate(self.test_loader):
            tensor = tensor.to(self.args.device)
            timestamp = timestamp.to(self.args.device)
            input_ = tensor[:, :self.args.input_steps]
            truth = tensor[:, self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
            input_norm = transform.minmax_norm(input_, self.args.vmax, self.args.vmin)
            truth_norm = transform.minmax_norm(truth, self.args.vmax, self.args.vmin)

            preds_norm = [self.model(input_norm) for _ in range(self.args.ensemble_members)]
            real_score = self.model.discriminator(tensor)
            fake_scores = [self.model.discriminator(torch.cat([input_norm, pred_norm], dim=1)) for pred_norm in preds_norm]
            fake_score = torch.mean(torch.stack(fake_scores), dim=0)
            loss_d = losses.cal_d_loss(fake_score, real_score)
            
            pred_norm = torch.mean(torch.stack(preds_norm), dim=0)
            loss_g = losses.cal_g_loss(fake_score) + \
                self.args.lambda_var * losses.cv_loss(pred_norm, truth_norm) + \
                self.args.lambda_rec * losses.biased_mae_loss(pred_norm, truth_norm, self.args.vmax, self.args.vmin)

            test_loss_g.append(loss_g.item())
            test_loss_d.append(loss_d.item())
            if (i + 1) % self.args.display_interval == 0:
                print('Batch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                    i + 1, len(self.test_loader), loss_g.item(), loss_d.item()))
            
            pred = transform.reverse_minmax_norm(pred_norm, self.args.vmax, self.args.vmin)
            for threshold in self.args.thresholds:
                pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
                metrics['POD-%ddBZ' % threshold].append(pod)
                metrics['FAR-%ddBZ' % threshold].append(far)
                metrics['CSI-%ddBZ' % threshold].append(csi)
            metrics['ME'].append(evaluation.evaluate_me(pred, truth))
            metrics['MAE'].append(evaluation.evaluate_mae(pred, truth))
            metrics['SSIM'].append(evaluation.evaluate_ssim(pred_norm, truth_norm))
            metrics['KLD'].append(evaluation.evaluate_kld(pred, truth))
                    
        print('Loss G: {:.4f} Loss D: {:.4f}'.format(np.mean(test_loss_g), np.mean(test_loss_d)))

        print('\nEvaluating...')
        for key in metrics.keys():
            if key != 'Time':
                metrics[key] = np.mean(metrics[key], axis=0)
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'test_metrics.csv'), float_format='%.4g', index=False)
        print('Evaluation complete')

        print('\nVisualizing...')
        visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'test')
        print('Visualization complete')
        print('\nTest complete')

    @torch.no_grad()
    def predict(self, model, sample_loader):
        print('\n[Predict]')
        self.model = model
        self.model.generator.to(self.args.device)
        self.model.generator.load_state_dict(self.load_checkpoint('bestmodel.pt')['model'])
        self.model.eval()
        self.count_params()
        
        for i, (tensor, timestamp) in enumerate(sample_loader):
            print('\nSample {}'.format(i))
            metrics = {}
            metrics['Time'] = np.linspace(6, 60, 10)
            tensor = tensor.to(self.args.device)
            timestamp = timestamp.to(self.args.device)
            input_ = tensor[:, :self.args.input_steps]
            truth = tensor[:, self.args.input_steps: self.args.input_steps + self.args.forecast_steps]
            input_norm = transform.minmax_norm(input_, self.args.vmax, self.args.vmin)
            truth_norm = transform.minmax_norm(truth, self.args.vmax, self.args.vmin)
            preds_norm = [self.model(input_norm) for _ in range(self.args.ensemble_members)]
            pred_norm = torch.mean(torch.stack(preds_norm), dim=0)
            pred = transform.reverse_minmax_norm(pred_norm, self.args.vmax, self.args.vmin)

            print('\nEvaluating...')
            for threshold in self.args.thresholds:
                pod, far, csi = evaluation.evaluate_forecast(pred, truth, threshold)
                metrics['POD-%ddBZ' % threshold] = pod
                metrics['FAR-%ddBZ' % threshold] = far
                metrics['CSI-%ddBZ' % threshold] = csi
            metrics['ME'] = evaluation.evaluate_me(pred, truth)
            metrics['MAE'] = evaluation.evaluate_mae(pred, truth)
            metrics['SSIM'] = evaluation.evaluate_ssim(pred_norm, truth_norm)
            metrics['KLD'] = evaluation.evaluate_kld(pred, truth)
                    
            df = pd.DataFrame(data=metrics)
            df.to_csv(os.path.join(self.args.output_path, 'sample_{}_metrics.csv'.format(i)), float_format='%.4g', index=False)
            print('Evaluation complete')

            print('\nVisualizing...')
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'sample_{}'.format(i))
            visualizer.plot_psd(pred, truth, self.args.output_path, 'sample_{}'.format(i))
            print('Visualization complete')
        
        print('\nPrediction complete')

    def save_checkpoint(self, filename='checkpoint.pt'):
        states = {
            'iteration': self.current_iterations,
            'train_loss_g': self.train_loss_g,
            'train_loss_d': self.train_loss_d,
            'val_loss_g': self.val_loss_g,
            'val_loss_d': self.val_loss_d,
            'model': self.model.generator.state_dict(),
            'discriminator': self.model.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict()
        }
        torch.save(states, os.path.join(self.args.output_path, filename))

    def load_checkpoint(self, filename='checkpoint.pt'):
        states = torch.load(os.path.join(self.args.output_path, filename), map_location=self.args.device)
        return states

    def count_params(self):
        G_params = filter(lambda p: p.requires_grad, self.model.generator.parameters())
        D_params = filter(lambda p: p.requires_grad, self.model.discriminator.parameters())
        num_G_params = sum([p.numel() for p in G_params])
        num_D_params = sum([p.numel() for p in D_params])
        print('\nModel name: {}'.format(type(self.model).__name__))
        print('G params: {}'.format(num_G_params))
        print('D params: {}'.format(num_D_params))
        print('Total params: {}'.format(num_G_params + num_D_params))
