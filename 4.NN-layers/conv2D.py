import numpy as np

class Conv2d:

    def __init__(
        self, in_channels, out_channels, kernel_size_h, kernel_size_w, padding=0, stride=1
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.padding = padding
        self.stride = stride

        self.W, self.biases = self.init_weight_matrix()

    def init_weight_matrix(self,):
        np.random.seed(1)
        W = np.random.uniform(size=(
            self.in_channels, self.kernel_size_h, self.kernel_size_w, self.out_channels
        ))
        biases = np.random.uniform(size=(1, self.out_channels))
        return W, biases

    def forward(self, x):
        _, h, w = x.shape
        padded_h = h + 2 * self.padding
        padded_w = w + 2 * self.padding

        padded_input = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        out_h = (padded_h - self.kernel_size_h) // self.stride + 1
        out_w = (padded_w - self.kernel_size_w) // self.stride + 1
        output = np.zeros((self.out_channels, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                for k in range(self.out_channels):
                    output[k, i, j] = self.biases[0,k]+np.sum(padded_input[:,i*self.stride:i*self.stride+                     self.kernel_size_h,j*self.stride:j*self.stride+self.kernel_size_w] * self.W[:, :, :, k])
        return output

def read_matrix(in_channels, h, w, dtype=float):
    return np.array([list(map(dtype, input().split())) 
                     for _ in range(in_channels * h)]).reshape(in_channels, h, w)

def print_matrix(matrix):
    for channel in matrix:
        for row in channel:
            print(' '.join(map(str, row)))

def solution():
    in_channels, out_channels, kernel_size_h, kernel_size_w, h, w, padding, stride = map(int, input().split())
    input_image = read_matrix(in_channels, h, w)

    conv = Conv2d(in_channels, out_channels, kernel_size_h, kernel_size_w, padding, stride)
    output = conv.forward(input_image).round(3)
    print_matrix(output)

if __name__ =='__main__':
    solution()
    # input:
        # 2 3 2 2 3 4 0 1
        # 1.0 1.0 1.0 1.0
        # 1.0 0.0 2.0 1.0
        # 1.0 2.0 0.0 1.0
        # 5.0 7.0 4.0 5.0
        # 3.0 6.0 5.0 5.0
        # 8.0 4.0 6.0 3.0
    # output:
        # 13.728 12.47 12.22
        # 12.19 12.804 10.673
        # 11.894 13.172 11.029
        # 10.776 11.907 11.507
        # 11.178 12.24 12.025
        # 14.462 11.378 10.675
