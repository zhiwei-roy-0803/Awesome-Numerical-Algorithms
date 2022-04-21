import numpy as np

def qr(A: np.ndarray):
    '''
    Implement full QR decomposition for a matrix A with arbitrary size
    :param A:
    :return:
    '''
    m, n = A.shape
    if m < n:
        raise RuntimeError("A should have more raws than columns")
    Q = np.zeros((m, m))
    R = np.zeros((m, n))

    Q[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0])
    R[0, 0] = np.linalg.norm(A[:, 0])
    for i in range(1, n):
        tmp = None
        for j in range(i):
            R[j, i] = np.vdot(A[:, i], Q[:, j]) # project the i-th column of A into the j-th orthogonal basis in R^{m}
            tmp = R[j, i] * Q[:, j]
        q = A[:, i] - tmp
        R[i, i] = np.linalg.norm(q)
        q /= np.linalg.norm(q) # normalize vector q so that it will be unit length
        Q[:, i] = q

    for i in range(n, m):
        a = np.zeros(m)
        a[i-n] = 1
        tmp = None
        for j in range(i):
            r = np.vdot(a, Q[:, j]) # project the i-th column of A into the j-th orthogonal basis in R^{m}
            tmp = r * Q[:, j]
        q = a - tmp
        Q[:, i] = q/np.linalg.norm(q)

    return Q, R
