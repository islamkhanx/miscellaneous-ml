import numpy as np


class MyDecisionTreeClassifier:

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
                        'classes_proba': None
                    }
        self.tree = None  # словарь в котором будет храниться построенное дерево
        self.classes = None  # список меток классов

    def fit(self, X, y):
        self.classes = np.unique(y)  
        self.tree = {'root': self._node.copy()}  # создаём первую узел в дереве
        self._build_tree(self.tree['root'], X, y)  # запускаем рекурсивную функцию для построения дерева
        return self

    def predict_proba(self, X):
        proba_preds = []
        for x in X:
            preds_for_x = self._get_predict(self.tree['root'], x)
            proba_preds.append(preds_for_x)
        return np.array(proba_preds)

    def predict(self, X):
        proba_preds = self.predict_proba(X)
        preds = proba_preds.argmax(axis=1).reshape(-1, 1)
        return preds
    
    def get_best_split(self, X, y):
        best_t = None
        best_j, best_Q = None, -np.inf
        for j in range(len(X[0])):
            Q,t = self.best_split(y, X[:,j])
            if Q > best_Q:
                best_Q = Q
                best_j = j
                best_t = t
        best_left_ids = X[:,best_j] < best_t 
        best_right_ids = X[:,best_j] >= best_t 
        return best_j, best_t, best_left_ids, best_right_ids
    
    def best_split(self,y, x_i):
        x_u = np.unique(x_i)
        t_potential = []

        for i in range(len(x_u) - 1):
            t_potential.append((x_u[i+1] + x_u[i]) / 2)
        t_best , Q_best = None, 0

        for t in t_potential:
            left = y[x_i >= t]
            right = y[x_i < t]
            Q  = self.calc_Q(left, right)
            if Q > Q_best:
                Q_best = Q
                t_best = t
        return Q_best,t_best
    
    
    def calc_Q(self, left, right):
        MSE = self.gini
        y = np.concatenate((left, right))
        return MSE(y) - (len(left) * MSE(left) + len(right) * MSE(right)) / len(y)

    def gini(self, y):
        values, counts = np.unique(y, return_counts=True)
        n =  len(y)
        return 1 - np.sum(counts**2) / n**2
        

    def _build_tree(self, curr_node, X, y):

        if curr_node['depth'] == self.max_depth:  # выход из рекурсии если построили до максимальной глубины
            curr_node['classes_proba'] = {c: (y == c).mean() for c in self.classes}  # сохраняем предсказания листьев дерева перед выходом из рекурсии
            return

        if len(np.unique(y)) == 1:  # выход из рекурсии значения если "y" одинковы для все объектов
            curr_node['classes_proba'] = {c: (y == c).mean() for c in self.classes}
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
            return [node['classes_proba'][c] for c in self.classes]

        if x[node['feature']] <= node['threshold']:  # уходим в правое или левое поддерево в зависимости от порога и признака
            return self._get_predict(node['left'], x)
        else:
            return self._get_predict(node['right'], x)


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

    clf = MyDecisionTreeClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba_preds = clf.predict_proba(X_test).round(4)

    print_matrix(preds)
    print_matrix(proba_preds)


if __name__ == '__main__':
    solution()
    # input:
        # 5 3 2
        # 7.5 8.1 20.0
        # 1.0 -4.7 6.0
        # 48.0 32.0 -0.5
        # 24.0 1.0 3.4
        # 3.0 2.5 6.0
        # 0
        # 2
        # 2
        # 1
        # 2
        # 0.6 5.1 3.2
        # 21.9 18.0 5.4
    # output:
        # 2
        # 1
        # 0.0 0.0 1.0
        # 0.0 1.0 0.0