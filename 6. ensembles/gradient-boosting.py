import numpy as np


class MyDecisionTreeRegressor:

    def __init__(self, max_depth=None, max_features=None, min_leaf_samples=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_leaf_samples = min_leaf_samples
        self._node = {
                        'left': None,
                        'right': None,
                        'feature': None,
                        'threshold': None,
                        'depth': 0,
                        'mean_val': None
                    }
        self.tree = None  # словарь в котором будет храниться построенное дерево


    def fit(self, X, y):
        self.tree = {'root': self._node.copy()}  # создаём первую узел в дереве
        self._build_tree(self.tree['root'], X, y)  # запускаем рекурсивную функцию для построения дерева
        return self


    def predict(self, X):
        preds = []
        for x in X:
            preds_for_x = self._get_predict(self.tree['root'], x)
            preds.append(preds_for_x)
        
        return np.array(preds)
    
    def get_best_split(self, X, y):
        best_t = None
        best_j, best_Q = None, -np.inf
        rand_features = np.random.choice(range(len(X[0])), size=min(self.max_features,len(X[0])),replace=False)
        for j in rand_features:
            Q,t = self.best_split(y, X[:,j])
            if Q > best_Q:
                best_Q = Q
                best_j = j
                best_t = t
        best_left_ids = X[:,best_j] <= best_t 
        best_right_ids = X[:,best_j] > best_t 
        return best_j, best_t, best_left_ids, best_right_ids
    
    def best_split(self,y, x_i):
        x_u = np.unique(x_i)
        t_potential = []

        for i in range(len(x_u) - 1):
            t_potential.append((x_u[i+1] + x_u[i]) / 2)
        t_best , Q_best = None, -np.inf

        for t in t_potential:
            left = y[x_i > t]
            right = y[x_i <= t]
            Q  = self.calc_Q(left, right)
            if Q > Q_best:
                Q_best = Q
                t_best = t
        return Q_best,t_best
    
    
    def calc_Q(self, left, right):
        MSE = self.mse
        y = np.concatenate((left, right))
        return MSE(y) - (len(left) * MSE(left) + len(right) * MSE(right)) / len(y)

    def mse(self, y):
        y_av = np.mean(y)
        return np.mean((y_av - y)**2)
        

    def _build_tree(self, curr_node, X, y):

        if curr_node['depth'] == self.max_depth:  # выход из рекурсии если построили до максимальной глубины
            curr_node['mean_val'] = np.mean(y)  # сохраняем предсказания листьев дерева перед выходом из рекурсии
            return
        
        if len(np.unique(y)) == 1:  # выход из рекурсии значения если "y" одинковы для все объектов
            curr_node['mean_val'] = np.mean(y)
            return

        j, t, left_ids, right_ids = self.get_best_split(X, y)  # нахождение лучшего разбиения

        curr_node['feature'] = j  # признак по которому производится разбиение в текущем узле
        curr_node['threshold'] = t  # порог по которому производится разбиение в текущем узле

        left = self._node.copy()  # создаём узел для левого поддерева
        right = self._node.copy()  # создаём узел для правого поддерева

        left['depth'] = curr_node['depth'] + 1  # увеличиваем значение глубины в узлах поддеревьев
        right['depth'] = curr_node['depth'] + 1

        curr_node['left'] = left
        curr_node['right'] = right

        self._build_tree(left, X[left_ids], y[left_ids])  # продолжаем построение дерева
        self._build_tree(right, X[right_ids], y[right_ids])

    def _get_predict(self, node, x):
        if node['threshold'] is None:  # если в узле нет порога, значит это лист, выходим из рекурсии
            return node['mean_val']

        if x[node['feature']] <= node['threshold']:  # уходим в правое или левое поддерево в зависимости от порога и признака
            return self._get_predict(node['left'], x)
        else:
            return self._get_predict(node['right'], x)

class MyGradientBoostingRegressor:

    def __init__(self, learning_rate, max_depth, max_features, n_estimators):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.estimators = []
        
    def fit(self, X, y):
        for i in range(self.n_estimators):
            estimator = MyDecisionTreeRegressor()

    def predict(self, X):
        pass


def read_matrix(n, dtype=float):
    matrix = np.array([list(map(dtype, input().split())) for _ in range(n)])
    return matrix

def read_input_matriсes(n, m, k):
    X_train, y_train, X_test = read_matrix(n), read_matrix(n), read_matrix(k)
    return X_train, y_train, X_test

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

def solution():
    n, m, k = map(int, input().split())
    X_train, y_train, X_test = read_input_matriсes(n, m, k)

    gb = MyGradientBoostingRegressor()
    gb.fit(X_train, y_train)

    predictions = gb.predict(X_test)
    print_matrix(predictions)


if __name__ == '__main__':
    solution()
    # input:
        # 3 4 2
        # 1.0 0.0 0.0 0.0
        # 2.0 0.0 0.0 0.0
        # 3.0 0.0 0.0 0.0
        # 1.0
        # 2.0
        # 3.0
        # 1.0 0.0 0.0 0.0
        # 2.0 0.0 0.0 0.0
    # output:
        # 1.0
        # 2.0
