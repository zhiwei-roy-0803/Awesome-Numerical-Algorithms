import numpy as np
from copy import deepcopy
from scipy.linalg import lu
from typing import List, Union


class LinearSolver:

    def __init__(self, max_iter: int = 1000, tol: float = 1e-3):
        self.max_iter = max_iter
        self.tol = tol
        pass

    def _check_square(self, A):
        '''
        Check whether input matrix A is a square matrix
        :param A:
        :return:
        '''
        r, c = A.shape
        if r != c:
            raise RuntimeError("Input matrix A should be a square matrix")

    def _check_diagonal_dominance(self, A: np.ndarray):
        '''
        Check whether input matrix A is suitable for using iterative method (diagonal dominant)
        :param A:
        :return:
        '''
        r = A.shape[0]
        for i in range(r):
            aii = A[i, i]
            sum_abs_row_i = np.sum(np.abs(A[i, :])) - np.abs(aii)
            if aii < sum_abs_row_i:
                raise RuntimeWarning("Input matrix A is not diagonal-dominant. Jacobian iteration or "
                                     "Gauss-Seidal method may not converge globally. Try Gauss Elimination "
                                     "or Gradient-Based (CG, GD) method instead.")


    def jac_iter(self, A: np.ndarray, b: np.ndarray, x0: np.ndarray) -> Union[np.ndarray, int]:
        self._check_square(A)
        self._check_diagonal_dominance(A)
        x = np.zeros_like(x0)
        r, c = A.shape
        for iter in range(1, self.max_iter):
            for i in range(r):
                tmp = 0
                aii = A[i, i]
                for j in range(c):
                    if j == i:
                        continue
                    tmp += x0[j] * A[i, j] # use previous result
                tmp = (b[i] - tmp) / aii
                x[i] = tmp
            if np.mean(np.abs(x - x0)) < self.tol:
                break
            x0 = deepcopy(x)
        return x, iter


    def gauss_seidel_iter(self, A: np.ndarray, b: np.ndarray, x0: np.ndarray) -> Union[np.ndarray, int]:
        self._check_square(A)
        self._check_diagonal_dominance(A)
        x = deepcopy(x0)
        r, c = A.shape
        for iter in range(1, self.max_iter):
            for i in range(r):
                tmp = 0
                aii = A[i, i]
                for j in range(c):
                    if j == i:
                        continue
                    tmp += x[j] * A[i, j] # use latest result
                tmp = (b[i] - tmp) / aii
                x[i] = tmp
            if np.mean(np.abs(x - x0)) < self.tol:
                break
            x0 = deepcopy(x)
        return x, iter


    def gauss_elimination(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        self._check_square(A)
        n = A.shape[0]

        # perform LU decomposition first: PA = LU
        P, L, U = lu(A) # we do not implement the tedious Gauss Elimination manually here :)
        b_prime = P@b

        # Ax = b <=> PAx = Pb (b') <=> LUx = b'
        # solve Lc = b' by forward substitution
        c = np.zeros(n)
        c[0] = b_prime[0] / L[0, 0]
        for i in range(1, n):
            tmp = 0
            for j in range(i):
                tmp += L[i, j] * c[j]
            c[i] = (b_prime[i] - tmp) / L[i, i]

        # solve Ux = c by backward substitution
        x = np.zeros(n)
        x[-1] = c[-1] / U[-1, -1]
        for i in range(n - 2, -1, -1):
            tmp = 0
            for j in range(n - 1, i, -1):
                tmp += U[i, j] * x[j]
            x[i] = (c[i] - tmp) / U[i, i]

        return x


    def tridiagonal(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        self._check_square(A)

        n = len(b)
        c = np.zeros(n - 1)
        d = np.zeros(n)

        # prepare vector c and d
        c[0] = A[0, 1] / A[0, 0]
        for i in range(1, n - 1):
            c[i] = A[i, i + 1] / (A[i, i] - A[i, i - 1] * c[i - 1])

        d[0] = b[0] / A[0, 0]
        for i in range(1, n):
            d[i] = (b[i] - A[i, i - 1] * d[i - 1]) / (A[i, i] - A[i, i - 1] * c[i - 1])

        # backward substitution for solving x
        x = np.zeros(n)
        x[-1] = d[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]

        return x




    def conjugate_grad(self, A: np.ndarray, b: np.ndarray):
        # TODO: implement conjugate gradient method for solving a linear system
        raise NotImplementedError



    def grad_descent(self, A:np.ndarray, b:np.ndarray, x0: np.ndarray) -> Union[np.ndarray, int]:
        '''
        Solve Ax = b by finding the optimal of the following quadratic programming

                            x = argmin 1/2 x'Ax + bx

        This optimization problem can be solved with gradient descent with line search
        :param A:
        :param b:
        :param x0:
        :return:
        '''
        self._check_square(A)
        x = deepcopy(x0)
        n = A.shape[0]
        for iter in range(self.max_iter):
            # gradient is equal to the residual
            d = r = A@x - b
            r = np.reshape(r, (n, 1))
            # linear search
            alpha = (r.T@r)/(r.T@A@r) # this alpha value will make the objective descent most, and it is optimal
            alpha = alpha[0, 0]
            # gradient descent
            x -= alpha * d
            if np.mean(np.abs(x - x0)) < self.tol:
                break
            x0 = deepcopy(x)
        return x, iter



if __name__ == "__main__":
    n = 10
    A = np.random.normal(loc=0.0, scale=1.0, size=(n, n)) + 20 * np.identity(n)
    b = np.random.randn(n)
    solver = LinearSolver()

    # reference solution
    x_gt = np.linalg.solve(A, b)

    # Jacob iteration
    x0 = np.zeros(n)
    x, iter = solver.jac_iter(A, b, x0)
    if np.mean(np.abs(x_gt - x)) > 1e-3:
        raise RuntimeError("Incorrect solution with Jacob Iteration")
    print(f"Solve Ax = b @ Iter = {iter} with Jacob Iteration")

    # Gauss-Seidal iteration
    x0 = np.zeros(n)
    x, iter = solver.gauss_seidel_iter(A, b, x0)
    if np.mean(np.abs(x_gt - x)) > 1e-3:
        raise RuntimeError("Incorrect solution with Gauss Seidel Iteration")
    print(f"Solve Ax = b @ Iter = {iter} with Gauss Seidel Iteration")

    # Gradient Descent
    x0 = np.zeros(n)
    x, iter = solver.grad_descent(A, b, x0)
    if np.mean(np.abs(x_gt - x)) > 1e-3:
        raise RuntimeError("Incorrect solution with Gradient Descent")
    print(f"Solve Ax = b @ Iter = {iter} with Gradient Descent ")

    # Gauss Elimination
    x = solver.gauss_elimination(A, b)
    if np.mean(np.abs(x_gt - x)) > 1e-3:
        raise RuntimeError("Incorrect solution with Gauss Elimination")
    print(f"Solve Ax = b with Gauss Elimination")