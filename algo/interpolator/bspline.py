import numpy as np
from scipy.misc import derivative
from ..linear_solver.solver import LinearSolver


def _Nik(u, knots, i, k):
    if k == 0:
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0
    if knots[i + k] == knots[i]:
        c1 = 0.0
    else:
        c1 = (u - knots[i]) / (knots[i + k] - knots[i]) * _Nik(u, knots, i, k - 1)
    if knots[i + k + 1] == knots[i + 1]:
        c2 = 0.0
    else:
        c2 = (knots[i + k + 1] - u) / (knots[i + k + 1] - knots[i + 1]) * _Nik(u, knots, i + 1, k - 1)
    return c1 + c2


def interpolate(u, knots, control_pts, degree):
    n = len(control_pts)
    val = 0.0
    for i in range(n):
        val += control_pts[i] * _Nik(u, knots, i, degree)
    return val

def basis_func(T, x):

    func_dict = {
        "func1": lambda x: (x - T[0]) * (x - T[0]) * (x - T[0]) / (T[1] - T[0]) / (T[2] - T[0]) / (T[3] - T[0]),

        "func2": lambda x: (x - T[0]) * (x - T[0]) * (T[2] - x) / (T[2] - T[1]) / (T[3] - T[0]) / (T[2] - T[0]) +
                           (T[3] - x) * (x - T[0]) * (x - T[1]) / (T[2] - T[1]) / (T[3] - T[1]) / (T[3] - T[0]) +
                           (T[4] - x) * (x - T[1]) * (x - T[1]) / (T[2] - T[1]) / (T[4] - T[1]) / (T[3] - T[1]),

        "func3": lambda x: (x - T[0]) * (T[3] - x) * (T[3] - x) / (T[3] - T[2]) / (T[3] - T[1]) / (T[3] - T[0]) +
                           (T[4] - x) * (T[3] - x) * (x - T[1]) / (T[3] - T[2]) / (T[4] - T[1]) / (T[3] - T[1]) +
                           (T[4] - x) * (T[4] - x) * (x - T[2]) / (T[3] - T[2]) / (T[4] - T[2]) / (T[4] - T[1]),

        "func4": lambda x: (T[4] - x) * (T[4] - x) * (T[4] - x) / (T[4] - T[3]) / (T[4] - T[2]) / (T[4] - T[1])
    }

    if T[0] <= x < T[1]:
        return func_dict["func1"](x)

    elif T[1] <= x < T[2]:
        return func_dict["func2"](x)

    elif T[2] <= x < T[3]:
        return func_dict["func3"](x)

    elif T[3] <= x < T[4]:
        return func_dict["func4"](x)
    else:
        raise RuntimeError

def basis_derivative2(T, x):

    func_dict = {
        "func1": lambda x: (x - T[0]) * (x - T[0]) * (x - T[0]) / (T[1] - T[0]) / (T[2] - T[0]) / (T[3] - T[0]),

        "func2": lambda x: (x - T[0]) * (x - T[0]) * (T[2] - x) / (T[2] - T[1]) / (T[3] - T[0]) / (T[2] - T[0]) +
                           (T[3] - x) * (x - T[0]) * (x - T[1]) / (T[2] - T[1]) / (T[3] - T[1]) / (T[3] - T[0]) +
                           (T[4] - x) * (x - T[1]) * (x - T[1]) / (T[2] - T[1]) / (T[4] - T[1]) / (T[3] - T[1]),

        "func3": lambda x: (x - T[0]) * (T[3] - x) * (T[3] - x) / (T[3] - T[2]) / (T[3] - T[1]) / (T[3] - T[0]) +
                           (T[4] - x) * (T[3] - x) * (x - T[1]) / (T[3] - T[2]) / (T[4] - T[1]) / (T[3] - T[1]) +
                           (T[4] - x) * (T[4] - x) * (x - T[2]) / (T[3] - T[2]) / (T[4] - T[2]) / (T[4] - T[1]),

        "func4": lambda x: (T[4] - x) * (T[4] - x) * (T[4] - x) / (T[4] - T[3]) / (T[4] - T[2]) / (T[4] - T[1])
    }

    if T[0] <= x < T[1]:
        return derivative(func_dict["func1"], x0=x, dx=1e-3, n=2)

    elif T[1] <= x < T[2]:
        return derivative(func_dict["func2"], x0=x, dx=1e-3, n=2)

    elif T[2] <= x < T[3]:
        return derivative(func_dict["func3"], x0=x, dx=1e-3, n=2)

    elif T[3] <= x < T[4]:
        return derivative(func_dict["func4"], x0=x, dx=1e-3, n=2)
    else:
        raise RuntimeError

class CubicBSpline():

    def __init__(self, points: np.ndarray, ndim: int = 2, epislon: float = 1e-3):
        self.points = points
        self.ndim = ndim
        self.num_points = len(points)
        self.n = self.num_points - 1
        self.eps = epislon
        self.deboor_pts, self.knots = self._find_control_points()

    def _parameterize(self, method="chord"):
        func_dict = {
            "uniform": self._uniform,
            "chord": self._chord
        }
        return func_dict[method]()


    def _uniform(self):
        '''
        parameterize the curve uniformly
        :return:
        '''
        return np.linspace(0, 1, self.num_points + 1)


    def _chord(self):
        '''
        parameterize the curve using chord length
        :return:
        '''
        t = np.zeros(self.num_points)
        dist = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)
        t[1:-1] = np.cumsum(dist)[:-1]/np.sum(dist)
        t[-1] = 1.0
        return t

    def _get_knots(self, t):
        t0 = t[0]
        tn = t[-1]
        knots = np.zeros(4 + 4 + self.num_points - 2)
        knots[:4] = [t0-3*self.eps, t0-2*self.eps, t0-self.eps, t0]
        knots[-4:] = [tn, tn+self.eps, tn+2*self.eps, tn+3*self.eps]
        knots[4:-4] = t[1:-1]
        i = 4
        while i < len(knots) - 4:
            j = 1
            while i + j < len(knots) and knots[i + j] - knots[i] < self.eps:
                knots[i + j] = knots[i] + j * self.eps
                j += 1
            i = i + j
        return knots

    def _build_linear_system(self, coordinates, knot):
        A = np.zeros((self.n + 3, self.n + 3))
        b = np.zeros(self.n + 3)

        # Step 1: set the first and last diagonal component in matrix A as 1
        A[0, 0] = basis_func(T=knot[:5], x=0.0)
        A[-1, -1] = basis_func(T=knot[-5:], x=1.0)

        # Step 2: for the i-th input data point except for the first and the last one, eval corresponding cubic basis
        # func (3 terms) and place the eval results into the (i + 1)-th row
        for i in range(2, self.n + 1):
            x = knot[i + 2]
            a1 = basis_func(T=knot[i - 1:i + 4], x=x)
            a2 = basis_func(T=knot[i:i + 5], x=x)
            a3 = basis_func(T=knot[i + 1:i + 6], x=x)
            A[i, i - 1] = a1
            A[i, i] = a2
            A[i, i + 1] = a3


        # Step 3: establish C2 continuity equation for the two boundary points, put the corresponding coefficients in
        # the second the penultimate row
        A[1, 0] = basis_derivative2(T=knot[0:5], x=knot[3])
        A[1, 1] = basis_derivative2(T=knot[1:6], x=knot[3])
        A[1, 2] = basis_derivative2(T=knot[2:7], x=knot[3])
        A[-2, -3] = basis_derivative2(T=knot[self.n:self.n+5], x=knot[self.n+3])
        A[-2, -2] = basis_derivative2(T=knot[self.n+1:self.n+6], x=knot[self.n+3])
        A[-2, -1] = basis_derivative2(T=knot[self.n+2:self.n+7], x=knot[self.n+3])

        # Step 4: place data point at the suitable place in vector b
        b[0] = coordinates[0]
        b[-1] = coordinates[-1]
        b[2:-2] = coordinates[1:-1]

        return [A, b]

    def _find_control_points(self):

        # Step 1: parameterize the curve
        t = self._parameterize()

        # Step 2: set knot vector
        knots = self._get_knots(t)

        # Step 3: build matrix and solve matrix with a linear system solver to find the required de Boor points
        deBoor_points = np.zeros((self.n + 3, self.ndim))
        solver = LinearSolver()
        for dim in range(self.ndim):
            A, b = self._build_linear_system(self.points[:, dim], knots)
            deBoor_points[:, dim] = solver.tridiagonal(A, b)

        return deBoor_points, knots

    def eval(self, ts: np.ndarray):
        x = [interpolate(t, self.knots, self.deboor_pts[:, 0], degree=3) for t in ts]
        y = [interpolate(t, self.knots, self.deboor_pts[:, 1], degree=3) for t in ts]
        return np.concatenate([x, y], axis=1)