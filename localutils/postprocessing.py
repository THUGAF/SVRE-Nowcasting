import os
import torch
import numpy as np
import pandas as pd
import utils.evaluation as evaluation

import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.scaler as scaler


class KF:
    def __init__(self, num_dims, M, x0=None):
        self.num_dims = num_dims
        self.M = M
        self.H = torch.eye(num_dims)
        self.R = torch.eye(num_dims)
        self.Q = torch.eye(num_dims)
        if x0 is None:
            self.x0 = torch.zeros(num_dims)
        else:
            assert num_dims == x0.size(0)
            self.x0 = x0
        self.x0.requires_grad = True
        self.P0 = torch.eye(num_dims, requires_grad=True)

    def forward(self, num_steps, ys):
        X = torch.zeros(num_steps, self.num_dims)
        P = torch.zeros(num_steps, self.num_dims, self.num_dims)
        x_for, P_for = self.x0, self.P0
        for i in range(num_steps):
            x_ana, P_ana = self.analysis(x_for, P_for, ys[i])
            x_for, P_for = self.forecast(x_ana, P_ana)
            X[i], P[i] = x_for, P_for
        return X, P

    def analysis(self, x_for, P_for, y):
        K = P_for @ self.H @ np.linalg.inv(self.H @ P_for @ self.H.T + self.R)
        x_ana = x_for + K @ (y - self.H @ x_for)
        P_ana = (torch.eye(self.num_dims) - K @ self.H) @ P_for
        return x_ana, P_ana

    def forecast(self, x_ana, P_ana):
        x_for = self.M(x_ana)
        # Calculate the jacobian matrix
        M_jacob = torch.zeros(self.num_dims, self.num_dims)
        for n in range(self.num_dims):
            w = torch.zeros(self.num_dims)
            w[n] = 1.0
            x_ana.backward(w)
            M_jacob[:, n] = x_for.grad
            x_for.grad.zero_()
        P_for = M_jacob @ P_ana @ M_jacob.T + self.Q
        return x_for, P_for


class EnKF:
    def __init__(self, num_dims, num_ensemble, M, x0=None):
        self.num_dims = num_dims
        self.M = M
        self.H = torch.eye(num_dims)
        self.R = torch.eye(num_dims)
        if x0 is None:
            self.x0 = torch.zeros(num_ensemble, num_dims)
        else:
            assert num_dims == x0.size(1)
            assert num_ensemble == x0.size(0)
            self.x0 = x0
        self.x0.requires_grad = True
        self.P0 = torch.eye(num_dims, requires_grad=True)
        self.num_ensemble = num_ensemble
    
    def forward(self, num_steps, ys):
        X = torch.zeros(num_steps, self.num_ensemble, self.num_dims)
        P = torch.zeros(num_steps, self.num_dims, self.num_dims)
        X_for, P_for = self.x0, self.P0
        for i in range(num_steps):
            Y, Ru = self.perturb(ys[i])
            X_ana, P_ana = self.analysis(X_for, P_for, Y, Ru)
            X_for, P_for = self.forecast(X_ana, P_ana)
            X[i], P[i] = X_for, P_for
        return X, P

    def perturb(self, y):
        mu = np.zeros(self.num_dims)
        U = np.random.multivariate_normal(mu, self.R, self.num_ensemble - 1)
        u_last = mu - np.sum(U, axis=0)
        U = np.concatenate((U, np.expand_dims(u_last, axis=0)))
        U = torch.from_numpy(U)
        Y = y.unsqueeze(0) + U
        Ru = (U.T @ U) / (self.num_ensemble - 1)
        return Y, Ru

    def analysis(self, X_for, P_for, Y, Ru):
        Ku = P_for @ self.H @ np.linalg.inv(self.H @ P_for @ self.H.T + Ru)
        X_ana = torch.stack([X_for[m] + Ku @ (Y[m] - self.H @ X_for[m]) for m in range(self.num_ensemble)])
        x_ana_bar = torch.mean(X_ana, dim=0)
        X_ana_anomly = X_ana - x_ana_bar.unsqueeze(0)
        P_ana = (X_ana_anomly.T @ X_ana_anomly) / (self.num_ensemble - 1)
        return X_ana, P_ana

    def forecast(self, X_ana, P_ana):
        X_for = torch.stack([self.M(X_ana[m]) for m in range(self.num_ensemble)])
        x_for_bar = torch.mean(X_for, dim=0)
        X_for_anomly = X_for - x_for_bar.unsqueeze(0)
        P_for = (X_for_anomly.T @ X_for_anomly) / (self.num_ensemble - 1)
        return X_for, P_for



class Assimilation:
    def __init__(self, args):
        self.args = args

    def load_checkpoint(self, filename='checkpoint.pt'):
        states = torch.load(os.path.join(self.args.output_path, filename))
        return states

    def evaluate(self, pred, truth, metrics):
        for threshold in self.args.thresholds:
            pod, far, csi, hss = evaluation.evaluate_forecast(pred, truth, threshold)
            metrics['POD-%ddBZ' % threshold] = pod
            metrics['FAR-%ddBZ' % threshold] = far
            metrics['CSI-%ddBZ' % threshold] = csi
            metrics['HSS-%ddBZ' % threshold] = hss

        metrics['CC'] = evaluation.evaluate_cc(pred, truth)
        metrics['ME'] = evaluation.evaluate_me(pred, truth)
        metrics['MAE'] = evaluation.evaluate_mae(pred, truth)
        metrics['RMSE'] = evaluation.evaluate_rmse(pred, truth)
        return metrics
    
    def da_forward(self, x):
        x = x[-1].flatten().expand()
        y = self.model(x)
        y = y[0].flatten()
        return y[0]

    def predict(self, model, sample_loader):
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        self.model = model
        self.model.load_state_dict(self.load_checkpoint('bestmodel.pt')['model'])
        self.model.eval()
        print('\n[Predict]')
        with torch.no_grad():
            # direct forecast with no observation
            for tensor, timestamp in sample_loader:
                tensor = tensor.transpose(1, 0)
                timestamp = timestamp.transpose(1, 0)
                input_ = tensor[:self.args.input_steps]
                truth = tensor[self.args.input_steps:]
                input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)
                
                pred = self.model(input_)
                input_rev = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
                pred_rev = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                truth_rev = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)

            metrics = self.evaluate(pred_rev, truth_rev, metrics)
            df = pd.DataFrame(data=metrics)
            df.to_csv(os.path.join(self.args.output_path, 'nonobs_metrics.csv'), float_format='%.8f', index=False)
            visualizer.plot_map(input_rev, pred_rev, truth_rev, timestamp, self.args.output_path, 'nonobs')
            print('No obs done.')

            # forecast with pure observation
            pred_obs = []
            for tensor, timestamp in sample_loader:
                tensor = tensor.transpose(1, 0)
                timestamp = timestamp.transpose(1, 0)
                truth = tensor[self.args.input_steps:]
                for s in range(self.args.forecast_steps):
                    input_ = tensor[s: s + self.args.input_steps]
                    input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)

                    pred = self.model(input_)
                    pred_rev = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                    pred_obs.append(pred_rev[0])
            
            input_ = tensor[:self.args.input_steps]
            pred_obs = torch.stack(pred_obs)
            metrics = self.evaluate(pred_obs, truth, metrics)
            df = pd.DataFrame(data=metrics)
            df.to_csv(os.path.join(self.args.output_path, 'obs_metrics.csv'), float_format='%.8f', index=False)
            visualizer.plot_map(input_, pred_obs, truth, timestamp, self.args.output_path, 'obs')
            print('Pure obs done.')

            # forecast with data assimilation
            # width = self.args.lon_range[1] - self.args.lon_range[0]
            # height = self.args.lat_range[1] - self.args.lat_range[0]
            # pred_da = []
            # for tensor, timestamp in sample_loader:
            #     tensor = tensor.transpose(1, 0)
            #     timestamp = timestamp.transpose(1, 0)
            #     input_ = tensor[:self.args.input_steps]
            #     input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
            #     ys = torch.stack([truth[s].flatten() for s in range(self.args.forecast_steps)])
                
            #     da_model = EnKF(width * height * self.args.forecast_steps, 5, self.da_forward, x0=torch.flatten(input_))
            #     pred, _ = da_model.forward(self.args.forecast_steps, ys)
            #     pred = torch.mean(pred, dim=1).view(self.args.forecast_steps, self.args.forecast_steps, -1)
            #     pred_rev = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
            #     for s in range(self.args.forecast_steps):
            #         pred_da.append(pred_rev[s, s].view(1, 1, width, height))
            
            # pred_da = torch.stack(pred_da)
            # metrics = self.evaluate(pred_da, truth_rev, metrics)
            # df = pd.DataFrame(data=metrics)
            # df.to_csv(os.path.join(self.args.output_path, 'enkf_metrics.csv'), float_format='%.8f', index=False)
            # visualizer.plot_map(input_rev, pred_rev, truth_rev, timestamp, self.args.output_path, 'enkf')
