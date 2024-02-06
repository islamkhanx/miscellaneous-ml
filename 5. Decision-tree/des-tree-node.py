import numpy as np


def MSE(y):
    mean = np.mean(y)
    return np.mean((y - mean)**2) 

def Q_func(left, right):
    y = np.concatenate((left, right))
    return MSE(y) - (len(left) * MSE(left) + len(right) * MSE(right)) / len(y)

def best_split(y, x_i):
    x_u = np.unique(x_i)
    t_potential = []
    
    for i in range(len(x_u) - 1):
        t_potential.append((x_u[i+1] + x_u[i]) / 2)
    t_best , Q_best = None, -np.inf
    
    for t in t_potential:
        left = y[x_i >= t]
        right = y[x_i < t]
        Q  = Q_func(left, right)
        if Q > Q_best:
            Q_best = Q
            t_best = t
            
            
    return Q_best,t_best

def decision_stump(X, y):
    best_t = None
    best_j, best_Q = None, -np.inf
    for j in range(len(X[0])):
        Q,t = best_split(y, X[:,j])
        if Q > best_Q:
            best_Q = Q
            best_j = j
            best_t = t
 
     
    # индекс признака по которому производился лучший сплит best_j = ...
    # порог с котором сравнивается признак best_t = ...
    best_left_ids = X[:,best_j] < best_t # вектор со значениями True для объектов в левом поддереве, 
    best_right_ids = X[:,best_j] >= best_t # вектор со значениями True для объектов в правом поддереве, 
    y_preds_left = np.mean(y[best_left_ids]) # предсказание в левом поддерева
    y_preds_right = np.mean(y[best_right_ids]) # предсказание в правом поддерева

    result = [
        best_Q,
        best_j,
        best_t,
        best_left_ids.sum(),
        best_right_ids.sum(),
        y_preds_left,
        y_preds_right
    ]
    return result

def read_input():
    n, m = map(int, input().split())
    x_train = np.array([input().split() for _ in range(n)]).astype(float)
    y_train = np.array([input().split() for _ in range(n)]).astype(float)
    return x_train, y_train

def solution():
    X, y = read_input()
    result = decision_stump(X, y)
    result = np.round(result, 2)
    output = ' '.join(map(str, result))
    print(output)

if __name__ == '__main__':
    solution()
    # input:
        # 5 3
        # 7.5 8.1 20.0
        # 1.0 -4.7 6.0
        # 48.0 32.0 -0.5
        # 24.0 1.0 3.4
        # 3.0 2.5 6.0
        # 7
        # 2
        # 12
        # 1
        # 3
    # output:
        # 13.5 1.0 5.3 3.0 2.0 2.0 9.5
