import numpy as np

class RNN:

    def __init__(self, in_features, hidden_size, n_classes, activation='tanh'):
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.activation = np.tanh

        init_weight_matrix = self.init_weight_matrix
        self.W_ax = init_weight_matrix(size=(hidden_size, in_features))
        self.W_aa = init_weight_matrix(size=(hidden_size, hidden_size))
        self.W_ay = init_weight_matrix(size=(n_classes, hidden_size))
        self.ba = init_weight_matrix(size=(hidden_size, 1))
        self.by = init_weight_matrix(size=(n_classes, 1))

    def init_weight_matrix(self, size):
        np.random.seed(1)
        W = np.random.uniform(size=size)
        return W

    def forward(self, x):
        in_features, T = x.shape

        a = np.zeros(shape=(self.hidden_size, T+1))
        for t in range(T):
            a[:, t+1] = self.activation(
                np.dot(self.W_aa, a[:, t])
                + np.dot(self.W_ax, x[:, t])
                + self.ba.T
                )

        y = np.zeros(shape=(self.n_classes, T))
        for t in range(T):
            y[:, t] = self.softmax(
                np.dot(self.W_ay, a[:, t+1])
                + self.by.T
            )

        return y

    def softmax(self, x):
        denom = np.sum(np.exp(x))
        return np.exp(x) / denom


def read_matrix(n_rows, dtype=float):
    return np.array([list(map(dtype, input().split())) for _ in range(n_rows)])

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

def solution():
    in_features, hidden_size, n_classes = map(int, input().split())
    input_vectors = read_matrix(in_features)

    rnn = RNN(in_features, hidden_size, n_classes)
    output = rnn.forward(input_vectors).round(3)
    print_matrix(output)


if __name__ == '__main__':
    solution()
    # input:
    # 2 2 3
        # 0.0 -1.0 2.0
        # -3.0 0.0 1.0
    # output:
        # 0.243 0.355 0.516
        # 0.526 0.461 0.329
        # 0.232 0.185 0.155
