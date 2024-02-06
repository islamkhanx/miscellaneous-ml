import numpy as np


def k_plus_plus(X: np.ndarray, k: int, random_state: int = 27) -> np.ndarray:
    """Инициализация центроидов алгоритмом k-means++.

    :param X: исходная выборка
    :param k: количество кластеров
    :return: набор центроидов в одном np.array
    """
    n_samples, n_features = X.shape
    centers = np.full(shape=(k, n_features), fill_value=np.inf)
    centers[0] = X[np.random.choice(n_samples, 1, replace=True)]
    for c in range(k-1):
        distances = np.min(np.sum((X[:,  np.newaxis, :] - centers) ** 2, axis=2), axis=1)
        sum_dist = np.sum(distances)
        probabilities = distances/sum_dist
        centers[c+1] = X[np.random.choice(n_samples, 1, replace=True, p=probabilities)]

    return centers


class KMeans:
    def __init__(self, n_clusters=8, tol=0.0001, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # инициализируем центры кластеров
        # centers.shape = (n_clusters, n_features)
        centers = k_plus_plus(X, self.n_clusters)

        for n_iter in range(self.max_iter):
            # считаем расстояние от точек из X до центроидов
            distances = np.sum((X[:,  np.newaxis, :] - centers) ** 2, axis=2)
            # определяем метки как индекс ближайшего для каждой точки центроида
            labels = np.argmin(distances, axis=1)

            old_centers = centers.copy()
            for c in range(self.n_clusters):
                # пересчитываем центроид
                # новый центроид есть среднее точек X с меткой рассматриваемого центроида
                if len(X[labels == c]) > 0:
                    centers[c, :] = np.mean(X[labels == c], axis=0)

            # записываем условие сходимости
            # норма Фробениуса разности центров кластеров двух последовательных итераций < tol
            if np.linalg.norm(centers - old_centers) < self.tol:
                break

        # cчитаем инерцию
        # сумма квадратов расстояний от точек до их ближайших центров кластеров
        inertia = np.min(
                    np.sum((X[:,  np.newaxis, :]-centers) ** 2, axis=2), axis=1
                ).sum()

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        return self

    def predict(self, X):
        # определяем метку для каждого элемента X на основании обученных центров кластеров 
        distances = np.sum(
            (X[:,  np.newaxis, :] - self.cluster_centers_) ** 2,
            axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def fit_predict(self, X):
        return self.fit(X).labels_


def read_input():
    n1, n2, k = map(int, input().split())

    read_line = lambda x: list(map(float, x.split()))
    X_train = np.array([read_line(input()) for _ in range(n1)])
    X_test = np.array([read_line(input()) for _ in range(n2)])

    return X_train, X_test, k


def solution():
    X_train, X_test, k = read_input()
    kmeans = KMeans(n_clusters=k, tol=1e-8, random_state=27)
    kmeans.fit(X_train)
    train_labels = kmeans.labels_
    test_labels = kmeans.predict(X_test)

    print(' '.join(map(str, train_labels)))
    print(' '.join(map(str, test_labels)))


if __name__ == '__main__':
    solution()
    # input:
        # 6 2 3
        # 1 2 3.5 4
        # 1 4 7 5.5
        # 1 0 5 9
        # 10 2 11 11
        # 10 4 -3 0
        # 10 0 2 2.5
        # 0 0 -1.5 1
        # 12 3 4 4
    # output:
        # 2 2 0 0 1 2
        # 2 2