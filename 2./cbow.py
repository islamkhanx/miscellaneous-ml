import numpy as np


class CBOW:

    def __init__(self, vocab_size: int, embedding_dim: int, random_state: int = 1):
        np.random.seed(random_state)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = self.init_weight_matrix()
        self.contexts = self.init_weight_matrix().T

    def init_weight_matrix(self, ):
        W = np.random.uniform(size=(self.vocab_size, self.embedding_dim))
        return W

    def forward(self, x):
        z1 = np.array([self.embeddings[i] for i in  x])
        z1 = np.sum(z1, axis=0)
        z2 = np.dot(z1, self.contexts)

        return self.softmax(z2)

    
    def softmax(self, vector):
        denom = np.sum(np.exp(vector))
        return np.exp(vector) / denom
    
    
def read_vector(dtype=int):
    return np.array(list(map(dtype, input().split())))

def solution():
    vocab_size, embedding_dim = read_vector()
    input_vector = read_vector()

    cbow = CBOW(vocab_size, embedding_dim)
    output = cbow.forward(input_vector).round(3)
    print(' '.join(map(str, output)))

if __name__ =='__main__':
    solution()
    # input:
        # 9 19
        # 7 8 4 5 7 4 4 0 5 1 2 7 1 4 5 3 8 1 4
    # output:
        # 0.0 0.001 0.0 0.0 0.999 0.0 0.0 0.0 0.0