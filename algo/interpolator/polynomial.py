import numpy as np
import matplotlib.pyplot as plt

class PolynomialInterpolator():

    def __init__(self, data_points: np.ndarray):
        self.n = len(data_points)
        self.data_points = data_points
        self.B = self._get_newton_coefficients()


    def _get_newton_coefficients(self):
        t = self.data_points[:, 0]
        B = np.zeros((self.n, self.n))
        B[:, 0] = self.data_points[:, 1]
        for i in range(1, self.n):
            for j in range(i, self.n):
                B[j, i] = (B[j, i-1] - B[j-1, i-1]) / (t[j] - t[j-i])
        return np.diag(B)

    def lagrange(self, t: float):
        '''
        Lagrange Interpolation
        :param t:
        :return:
        '''
        x = self.data_points[:, 0]
        y = self.data_points[:, 1]
        val = 0
        for k in range(self.n):

            xk = x[k]
            yk = y[k]

            x_prime = np.delete(x, k)

            nominator = np.prod(t - x_prime) # (t-t1)(t-t2)...(t-t_{k-1})(t-t_{k+1})...(t-tn)
            denominator = np.prod(xk - x_prime) # (tk-t1)(tk-t2)...(tk-t_{k-1})(tk-t_{k+1})...(tk-tn)

            Lk = nominator / denominator

            val += Lk * yk

        return val

    def newton(self, x):
        '''
        Newtons divided difference
        y = \sum_{i=1}^{n} B_i N_i(x)
        where N_i(x) = (x - x1)(x - x2)...(x - x_{i-1})
        :param x:
        :return:
        '''
        y = self.B[-1]
        t = self.data_points[:, 0]
        # Newton's Divided Difference can be implemented in a recursive manner
        for i in range(self.n - 2, -1, -1):
            y = (x - t[i])*y + self.B[i]
        return y

if __name__ == "__main__":

    data_points = np.array([
        [-2, -13],
        [-1, -7],
        [0, 1],
        [2, 2],
        [3, 4],
        [4, 9]
    ])

    poly_interpolator = PolynomialInterpolator(data_points)

    ts = np.linspace(np.min(data_points[:, 0]), np.max(data_points[:, 0]), 200)
    ys = [poly_interpolator.lagrange(t) for t in ts]

    plt.figure(dpi=300)
    plt.plot(ts, ys, lw=2, color="b")
    plt.plot(data_points[:, 0], data_points[:, 1], ' ', marker='d', markerfacecolor='r', label='data points')
    plt.legend(["Lagrange", "Data Points"])
    plt.grid(axis="y")
    plt.show()


    ts = np.linspace(np.min(data_points[:, 0]), np.max(data_points[:, 0]), 200)
    ys = [poly_interpolator.newton(t) for t in ts]

    plt.figure(dpi=300)
    plt.plot(ts, ys, lw=2, color="b")
    plt.plot(data_points[:, 0], data_points[:, 1], ' ', marker='d', markerfacecolor='r', label='data points')
    plt.legend(["Newton's Divided Difference", "Data Points"])
    plt.grid(axis="y")
    plt.show()