import torch
import numpy as np


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
