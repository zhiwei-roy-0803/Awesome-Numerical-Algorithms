import numpy as np
from copy import deepcopy

import scipy.linalg

from algo.utils.qr import qr

def power_iteration(A: np.ndarray, x0: np.ndarray, max_iter: int = 100):
    '''
    Power iteration for finding the largest eigen value of A and the corresponding eigenvector
    :param A:
    :param x0:
    :return:
    '''
    m, n = A.shape
    if m != n:
        raise RuntimeError("Only square matrices have eigen value and eigen vectors!")
    u = deepcopy(x0)
    lamb = 0
    for i in range(1, max_iter + 1):
        u  /= np.linalg.norm(u) # normalization before computing the next estimation of lambda
        lamb = u.T@A@u # least square estimation of lambda: Au = lu => u^TAu = l u^Tu => l = u^T A u since u shas unit length
        u = A@u
    u /= np.linalg.norm(u) # final normalization to make it unit length
    return lamb[0, 0], u

def inverse_power_iteration(A: np.ndarray, x0: np.ndarray, s: float, max_iter: int = 100):
    '''
    Inverse Power Iteration for finding the eigen value of A that is closest to s
    :param A:
    :param x0:
    :param s:
    :param max_iter:
    :return:
    '''
    m, n = A.shape
    if m != n:
        raise RuntimeError("Only square matrices have eigen value and eigen vectors!")
    u = deepcopy(x0)
    lamb = 0
    for i in range(1, max_iter + 1):
        x = u / np.linalg.norm(u)
        # u, _ = solver.grad_descent(A - s*np.identity, x, u) # (A-sI)u_{k+1} = u_{k} => u_{k+1} = (A - sI)^(-1) u_{k}
        # u = np.linalg.solve(A - s*np.identity(m), x)
        u = np.linalg.solve(A - s*np.identity(m), x)
        lamb = np.vdot(u, x)
        # print(lamb)
    u /= np.linalg.norm(u)
    return lamb, u

def rayleigh_quotient_iteration(A: np.ndarray, x0: np.ndarray, max_iter: int = 5):
    '''
    Implement Rayleigh Quotient Iteration to find the smallest eigen value of a square matrix A
    The key difference of RQI compared with Inverse Power Iteration is that the "s" in Inverse Power Iteration
    is now replaced with the latest estimation of the smallest eigen value \lambda. That means that RQI is a
    dynamic version of Inverse Power Iteration and thus it is much faster than the Inverse Power Iteration
    :param A:
    :param x0:
    :param max_iter:
    :return:
    '''
    m, n = A.shape
    if m != n:
        raise RuntimeError("Only square matrices have eigen value and eigen vectors!")
    u = deepcopy(x0)
    lamb = 0
    for i in range(1, max_iter + 1):
        x = u / np.linalg.norm(u)
        lamb = x.T@A@x
        u = np.linalg.solve(A - lamb*np.identity(m), x)
        print(lamb)
    u /= np.linalg.norm(u)
    return lamb[0, 0], u

def normalized_simultaneous_iteration(A: np.ndarray, max_iter: int = 10):
    '''
    Implement Normalized Simultaneous Iteration for find all eigen vectors and eigen values in parallel

    For an orthogonal matrix Q = [q1, q2, ..., qn], AQ = [Aq1, Aq2, ..., Aqn]. According to the idea of
    power iteration, Aq_i will be the estimation of the i-th eigen vector. However, in general Aq_i is not
    orthogonal to Aq_j, so after we obtain AQ, we should compute its QR decomposition to obtain another set of
    orthogonal basis. Finally, after several iteration, Q will be an orthogonal matrix containing eigen vectors of A.
    R will be a diagonal matrix which diagonal elements will be the eigen values.
    :param A:
    :return:
    '''
    m, n = A.shape
    if m != n:
        raise RuntimeError("Only square matrices have eigen value and eigen vectors!")

    # NSI main iteration
    Q, R = scipy.linalg.qr(A)
    for iter in range(1, max_iter):
        Q, R = scipy.linalg.qr(A@Q)

    lamb = np.diag(R)
    return lamb, Q

def unshifted_qr(A: np.ndarray, max_iter: int = 10):
    '''
    Implement Normalized Simultaneous Iteration for find all eigen vectors and eigen values in parallel

    For an orthogonal matrix Q = [q1, q2, ..., qn], AQ = [Aq1, Aq2, ..., Aqn]. According to the idea of
    power iteration, Aq_i will be the estimation of the i-th eigen vector. However, in general Aq_i is not
    orthogonal to Aq_j, so after we obtain AQ, we should compute its QR decomposition to obtain another set of
    orthogonal basis. Finally, after several iteration, Q will be an orthogonal matrix containing eigen vectors of A.
    R will be a diagonal matrix which diagonal elements will be the eigen values.
    :param A:
    :return:
    '''
    m, n = A.shape
    if m != n:
        raise RuntimeError("Only square matrices have eigen value and eigen vectors!")

    # NSI main iteration
    Q, R = scipy.linalg.qr(A)
    for iter in range(1, max_iter):
        Ai = R@Q
        Q, R = scipy.linalg.qr(Ai)

    lamb = np.diag(R)
    return lamb, Q


if __name__ == "__main__":
    A = np.random.normal(0, 1, (100, 5))
    A = A@A.T
    x0 = np.zeros((100, 1))
    x0[0] = 1
    lamb, _ = power_iteration(A, x0)
    print(f"largest lambda = {lamb} from Power Iteration")
    lamb, _ = inverse_power_iteration(A, x0, s=0.5)
    print(f"Smallest lambda = {lamb} from Power Iteration")

    lamb_gt = np.linalg.eigvals(A).real
    lamb_largest_gt = lamb_gt[0]
    lamb_smallest_gt = lamb_gt[-1]

    print(f"largest lambda = {lamb_largest_gt} from Numpy")
    print(f"smallest lambda = {lamb_smallest_gt} from Numpy")

    lamb, _ = rayleigh_quotient_iteration(A, x0)
    print(f"lambda = {lamb} from RQI")

    lamb, Q = normalized_simultaneous_iteration(A, max_iter=200)
    if np.mean(np.abs(lamb - lamb_gt)) < 1e-3:
        print(f"NSI correct")

    lamb, Q = unshifted_qr(A, max_iter=500)
    if np.mean(np.abs(lamb - lamb_gt)) < 1e-3:
        print(f"Unshifted QR correct")

