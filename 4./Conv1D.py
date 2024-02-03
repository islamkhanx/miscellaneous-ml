import numpy as np


class Conv1d:

    def __init__(self, in_channels, out_channels, kernel_size, padding='same', activation='relu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation

        self.W, self.biases = self.init_weight_matrix()

    def init_weight_matrix(self,):
        np.random.seed(1)
        W = np.random.uniform(size=(self.in_channels, self.kernel_size, self.out_channels))
        biases = np.random.uniform(size=(1, self.out_channels))
        return W, biases

    def forward(self, x):
        pad_size = self.kernel_size // 2
        x_padded = np.pad(x, ((0, 0), (pad_size, pad_size)), mode ='constant')

        T = x.shape[1]
        output = np.zeros((self.out_channels, T))

        for t in range(T):
            for k in range(self.out_channels):
                output[k, t] = np.sum(x_padded[:, t:t + self.kernel_size] * self.W[:, :, k]) + self.biases[0, k]
                output[k, t] = max(0, output[k, t])
        return output

def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

def solution():
    in_channels, out_channels, kernel_size = map(int, input().split())
    input_vectors = read_matrix(in_channels)

    conv = Conv1d(in_channels, out_channels, kernel_size)
    output = conv.forward(input_vectors).round(3)
    print_matrix(output)


if __name__ == '__main__':
    solution()
    # input:
        # 2 2 3
        # 10.0 -10.0 10.0
        # 5.0 15.0 -5.0
    # output:
        # 7.01 10.628 0.0
        # 15.95 12.365 0.0