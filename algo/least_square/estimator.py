import numpy as np
import matplotlib.pyplot as plt
from algo.least_square.utils import qr

class Estimator():

    def __init__(self, method="full QR"):
        self.method = method

    def fit(self, X:np.ndarray, Y:np.ndarray):
        '''
        Implement the Least Square using standard normal equation method
        :param X:
        :return:
        '''
        m = X.shape[0]
        X = np.concatenate([X, np.ones((m, 1))], axis=1) # add additional column for the offset
        n = X.shape[1]
        if self.method == "normal equation":
            normal_matrix = X.T@X
            w = np.linalg.solve(normal_matrix, X.T@Y)
            self.w = w
        elif self.method == "full QR":
            # Ay = b => QRx = y => Rx = Q^Ty

            # full QR decomposition for A
            Q, R = qr(X)

            # y = Q^Ty
            y = Q.T@Y

            # solve Rx = Q^Ty with backward substitution
            w = np.zeros(n)
            w[-1] = y[n - 1] / R[n - 1, n - 1]
            for i in range(n - 2, -1, -1):
                w[i] = (y[i - i] - np.sum(w[-(i+1):] * R[i, -(i+1):])) / (R[i, i])
            self.w = w

            e = (R@w - y.T).squeeze()
            self.squared_error = np.sum(e[n:]**2)

    def predict(self, X: np.ndarray):
        m, n = X.shape
        X = np.concatenate([X, np.ones((m, 1))], axis=1) # add additional column for the offset
        return X@self.w

    def get_squared_error(self):
        return float(self.squared_error)

if __name__ == "__main__":

    data_points = np.array([
        [-2, -13],
        [-1, -7],
        [0, 1],
        [2, 2],
        [3, 4],
        [4, 9]
    ])

    estimator = Estimator(method="full QR")
    X = np.reshape(data_points[:, 0], (6, 1))
    Y = np.reshape(data_points[:, 1], (6, 1))
    estimator.fit(X, Y)

    Y_prime = estimator.predict(X)
    squared_error = float(np.sum((Y.T-Y_prime)**2))
    print(f"Squared error from evaluation = {round(squared_error, 2)}")
    print(f"Squared error from QR decomposition = {round(estimator.get_squared_error(), 2)}")


    x = np.linspace(np.min(data_points[:, 0]), np.max(data_points[:, 0]), 200)
    x = np.reshape(x, (200, 1))
    y = estimator.predict(x)

    plt.figure(dpi=300)
    plt.plot(data_points[:, 0], data_points[:, 1], ' ', marker='d', markerfacecolor='r', label='data points')
    plt.plot(x, y, lw=2, color="b")
    plt.legend(["Least Square", "Data Points"])
    plt.grid(axis="y")
    plt.show()




