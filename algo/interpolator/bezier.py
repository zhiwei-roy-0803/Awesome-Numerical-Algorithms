import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class Bezier():

    def __init__(self, control_points: np.ndarray, degree: int):

        self.control_points = control_points
        self.dergre = degree
        self.n = len(control_points) - 1

    def eval(self, t):
        '''
        Evaluate a Bézier curve at t with de Casteljau algorithm
        :param t:
        :return:
        '''
        P0 = [pt for pt in self.control_points]
        for k in range(self.n):
            P = []
            m = len(P0) - 1
            for i in range(m):
                P.append((1 - t) * P0[i] + t * P0[i + 1]) # successive linear interpolation
            P0 = deepcopy(P)
        return P[0]


    def degree_elevation(self, new_degree):
        '''
        Compute the control points of a Bézier curve with the same shape as the current one but with higher degree
        :return:
        '''
        raise NotImplementedError

class CubicBezierInterpolator():

    def __init__(self, data_points, end_condition="natural"):
        self.data_points = data_points
        self.n = len(data_points) - 1 # number line segments
        self.curves = self._find_control_points(end_condition)

    def _find_control_points(self, end_condition):
        '''
        Find cubic curves that can interpolate each line segment between two adjacent data points.
        (1) Step One: determine the tangent vector for each intermediate data points using Catmull-Rom's rule
        (2) For each line segment, there are four control points that are to be determined. Using the interpolation
            requirement, we can obtain two equations (the Bezier curve should interpolate the start and end point).
            Using two tangent vector, we can obtain another two equations. Solve these four equations will yield the
            four control points of the Bezier curve for a specific line segment
        (3) After finding the control points for each line segment, evaluating the Bezier curve for each line segment
            using de Casteljau algorithm to obtain the final interpolation result.

        :param end_condition:
        :return:
        '''

        # determine the tangent vector of each intermediate point using Catmull-Rom's rule. This will guarantee the C1
        # continuity in the connection point of two Bézier curve
        T = []
        for i in range(1, self.n):
            T.append(0.5 * (self.data_points[i + 1] - self.data_points[i - 1]))
        curves = []
        for i in range(self.n):
            if i == 0:
                Pi0 = self.data_points[0]
                Pi2 = self.data_points[1] - 1/3 * T[0]
                Pi3 = self.data_points[1]
                if end_condition == "natural":
                    Pi1 = (self.data_points[0] + Pi2) / 2
                elif end_condition == "simple":
                    Pi1 = Pi0
                else:
                    raise NotImplementedError

            elif i == self.n - 1:
                Pi0 = self.data_points[-2]
                Pi1 = self.data_points[-2] + 1/3 * T[-1]
                Pi3 = self.data_points[-1]
                if end_condition == "natural":
                    Pi2 = (self.data_points[-1] + Pi1) / 2
                elif end_condition == "simple":
                    Pi2 = Pi3
                else:
                    raise NotImplementedError

            else:
                Pi0 = self.data_points[i]
                Pi1 = self.data_points[i] + 1/3 * T[i - 1]
                Pi2 = self.data_points[i + 1] - 1/3 * T[i]
                Pi3 = self.data_points[i + 1]

            control_points = np.concatenate([[Pi0], [Pi1], [Pi2], [Pi3]], axis=0)
            curves.append(Bezier(control_points, 3))

        return curves


    def eval(self):
        pts = []
        for curve in self.curves:
            ts = np.linspace(0, 1, 100)
            pts.append(np.array([curve.eval(t) for t in ts]))
        return np.concatenate(pts, axis=0)

if __name__ == "__main__":
    control_points = np.array([
        [0, 0],
        [2, 0],
        [2, 2],
        [4, 0]
    ])
    degree = 3
    B = Bezier(control_points, degree)
    ts = np.linspace(0, 1, 100)
    pts = np.array([B.eval(t) for t in ts])

    plt.figure(dpi=300)
    plt.plot(pts[:, 0], pts[:, 1], lw=2, color="b")
    plt.plot(control_points[:, 0], control_points[:, 1], ' ', marker='d', markerfacecolor='r', label='control points')
    plt.legend(["Bezier", "Control Points"])
    plt.grid(axis="y")
    plt.show()

    data_points = np.array([
        [0, 0],
        [2, 0],
        [2, 2],
        [4, 0]
    ])

    bezier_spline_interpolator = CubicBezierInterpolator(data_points, end_condition="simple")
    pts = bezier_spline_interpolator.eval()

    plt.figure(dpi=300)
    plt.plot(pts[:, 0], pts[:, 1], lw=2, color="b")
    plt.plot(data_points[:, 0], data_points[:, 1], ' ', marker='d', markerfacecolor='r', label='data points')
    plt.legend(["Bezier Spline", "Data Points"])
    plt.grid(axis="y")
    plt.show()