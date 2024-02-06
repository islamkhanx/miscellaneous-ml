import numpy as np
import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        shape = (1, num_features)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        # torch.is_grad_enabled() возвращает True, если расчёт градиентов включен, 
        # то есть модель находится в состоянии обучения (train)
        if torch.is_grad_enabled():
            mu = torch.mean(x, dim=0, keepdim=True)
            var_unc = torch.var(x, dim=0, keepdim=True, unbiased=False)
            var_cor = torch.var(x, dim=0, keepdim=True, unbiased=True)

            self.moving_mean =  (1 - self.momentum) * self.moving_mean + self.momentum * mu
            self.moving_var =  (1 - self.momentum) * self.moving_var + self.momentum * var_cor

            x = x - mu
            x = x /(var_unc + self.eps)**0.5

        else:
            x = x - self.moving_mean
            x = x / (self.moving_var + self.eps)**0.5

        x = x * self.gamma + self.beta
        return x

def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)]).astype(float)

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

def solution():
    batch_size, num_features = map(int, input().split())
    eps, momentum = map(float, input().split())
    train_vectors = read_matrix(batch_size)
    test_vectors = read_matrix(batch_size)

    train_vectors = torch.from_numpy(train_vectors).float()
    test_vectors = torch.from_numpy(test_vectors).float()

    batch_norm_1d = BatchNorm1d(num_features, eps, momentum)
    output_train = batch_norm_1d.forward(train_vectors).detach().numpy().round(2)
    with torch.no_grad():
        output_eval = batch_norm_1d.forward(test_vectors).detach().numpy().round(2)

    print_matrix(output_train)
    print()
    print_matrix(output_eval)

if __name__ == '__main__':
    solution()
    # input:
        # 2 4
        # 0.01 0.1
        # 0.5 1.0 0.3 1.3
        # 0.9 0.2 0.4 1.0
        # 0.2 1.1 0.5 1.2
        # 0.4 0.7 0.6 1.1
    # output:
        # -0.89 0.97 -0.45 0.83
        # 0.89 -0.97 0.45 -0.83

        # 0.14 1.07 0.49 1.13
        # 0.34 0.66 0.59 1.03
