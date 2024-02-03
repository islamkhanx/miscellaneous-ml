import numpy as np

class LogisticRegression:

    def __init__(self, max_iter=5e3, lr=0.04, tol=0.001, l1_coef=0.1):

        '''
        max_iter – максимальное количеств
        '''

        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.l1 = l1_coef
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        '''
        Обучение модели.

        X_train – матрица объектов для обучения
        y_train – ответы на объектах для обучения

        X_val – матрица объектов для валидации
        y_val – ответы на объектах для валидации
        '''

        n, m = X_train.shape

        self.weights = np.zeros((1, m))
        self.bias = np.mean(y_train)

        n_iter = 0
        gradient_norm = np.inf

        while n_iter < self.max_iter and gradient_norm > self.tol:

            dJdw, dJdb = self.grads(X_train, y_train)
            gradient_norm = np.linalg.norm(np.hstack([dJdw.flatten(), [dJdb]]))

            self.weights -= self.lr * dJdw
            self.bias -= self.lr * dJdb

            n_iter += 1

        return self

    def predict(self, X):

        '''
        Метод возвращает предсказанную метку класса на объектах X
        '''

        predictions = (self.predict_proba(X) > 0.5).astype(int)
        return predictions


    def predict_proba(self, X):

        '''
        Метод возвращает вероятность класса 1 на объектах X
        '''
        logits = np.dot(X, self.weights.T) + self.bias
        return self.sigmoid(logits)

    def grads(self, X, y):

        '''
        Рассчёт градиентов
        '''
        def w_der(w):
            if w > 0:
                return 1
            elif w == 0:
                return 0
            else:
                return -1
        y_hat = self.predict_proba(X)
        w_r = self.l1 * np.apply_along_axis(w_der, axis=0, arr=self.weights)
        dJdw = np.mean(X * (y_hat - y), axis=0, keepdims=True) + w_r
        dJdb = np.mean(y_hat - y)

        return dJdw, dJdb

    @staticmethod
    def sigmoid(x):
        '''
        Сигмоида от x
        '''
        return 1 / (1 + np.exp(-x))


def read_input():
    n, m = map(int, input().split())
    x_train = np.array([input().split() for _ in range(n)]).astype(float)
    y_train = np.array([input().split() for _ in range(n)]).astype(float)
    return x_train, y_train


def solution():
    x_train, y_train = read_input()

    model = LogisticRegression(max_iter=5e3, lr=0.04, l1_coef=0.1)
    model.fit(x_train, y_train)

    all_weights = [model.bias] + list(model.weights.flatten())
    result = ' '.join(map(lambda x: str(float(x)), all_weights))
    print(result)

if __name__ == '__main__':
    solution()
    # input: 
        # 6 2
        # -2 2
        # 2 -2
        # -2 2
        # 1 -1
        # -1 1
        # 2 -1
        # 0
        # 1
        # 0
        # 1
        # 0
        # 1
    # output: 
        # 0.004437 1.63406495 -0.0031905