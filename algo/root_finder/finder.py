import numpy as np
from sympy import Symbol, lambdify
from scipy.misc import derivative
from typing import Callable

class RootFinder():


    def __init__(self, tol: float = 1e-3, max_iter: int = 1000):
        self.tol = tol
        self.max_iter = max_iter


    def bisect(self, func: Callable, a: float, b: float):
        # check whether f(min)f(max) < 0
        if func(a) * func(b) > 0:
            raise ValueError("You should make sure f(min)f(max) <= 0")

        # bisection search
        iter = 0
        while (b - a) / 2 > self.tol and iter < self.max_iter:
            c = (b + a) / 2
            if np.isclose(np.abs(func(c)), 0):
                break
            if func(a) * func(c) < 0:
                b = c
            else:
                a = c
            iter += 1
        return (b + a) / 2, iter


    def netwon(self, func: Callable, x0: float = None, J:Callable = None):
        '''
        Newton method: use the tangent line to approximate the func locally and the next guess will be the intersection
        of the tangent line and the x-axis
        :param func:
        :param x0:
        :param J:
        :return:
        '''
        iter = 0
        x = x0
        while iter < self.max_iter:
            if np.abs(func(x)) < self.tol:
                break
            if isinstance(J, Callable):
                x -= func(x) / J(x)
            else:
                J = derivative(func, x) # TODO: replace scipy numerical derivative with the method in this library
                x -= func(x) / J
            iter += 1
        return x, iter


    def scant(self, func:Callable, x0: float, x1: float):
        '''
        A derivative-free version of the Gauss Newton method. Each iteration using a line to approximate the tangent
        :param func:
        :param x0:
        :param x1:
        :return:
        '''
        iter = 0
        x = x1
        while iter < self.max_iter:
            x = x0 - func(x0) * (x1- x0) / (func(x1) - func(x0))
            if np.abs(func(x)) < self.tol:
                break
            x0 = x1
            x1 = x
            iter += 1

        return x, iter


    def false_positive(self, func:Callable, b: float, a: float):
        '''
        Combination of Bisection and Secant. Use the point computed by Secant method as the midpoint in Bisection method
        :param func:
        :param b:
        :param a:
        :return:
        '''
        iter = 0
        while (b - a) / 2 > self.tol and iter < self.max_iter:
            c = a - func(a) * (b- a) / (func(b) - func(a)) # use the point obtained from Secant method as the midpoint in Bisection method
            if np.isclose(np.abs(func(c)), 0):
                break
            if func(a) * func(c) < 0:
                b = c
            else:
                a = c
            iter += 1

        return x, iter


    def IQI(self, func: Callable, x0: float, x1: float, x2: float):
        '''
        Use a polynomial instead of a line to approximate the func locally. Use (y0, x0), (y1, x1), (y2, x2)
        to fit a quadratic polynomial x = p(y) and the next guess will be x = p(0)
        :param func:
        :param x0:
        :param x1:
        :param x2:
        :return:
        '''
        iter = 0
        x = x2
        while iter < self.max_iter:
            y0 = func(x0)
            y1 = func(x1)
            y2 = func(x2)
            # fit an inverse polynomial x = p(y)
            p = np.polyfit(x=[y0, y1, y2], y=[x0, x1, x2], deg=2)
            x = np.polyval(p, x=0)
            if np.isclose(np.abs(func(x)), 0):
                break
            x0 = x1
            x1 = x2
            x2 = x
            iter += 1
        return x, iter


if __name__ == "__main__":

    x = Symbol("x")

    y = x**2 - 3
    y_prime = y.diff()

    func = lambdify(x, y, 'numpy')
    J = lambdify(x, y_prime, 'numpy')

    root_finder = RootFinder()

    # Bisection
    x, iter = root_finder.bisect(func, a=0, b=2)
    print(f"x* = {round(x, 2)} by bisection @ iter = {iter}")

    # Newton
    x, iter = root_finder.netwon(func, x0=2.0, J=None)
    print(f"x* = {round(x, 2)} by Newton @ iter = {iter}")

    # Scant
    x, iter = root_finder.scant(func, x0=0.0, x1=2.0)
    print(f"x* = {round(x, 2)} by Secant @ iter = {iter}")

    # false positive
    x, iter = root_finder.false_positive(func, a=0.0, b=2.0)
    print(f"x* = {round(x, 2)} by False Positive @ iter = {iter}")

    # IQI
    x, iter = root_finder.IQI(func, x0=0, x1=1.0, x2=2.0)
    print(f"x* = {round(x, 2)} by IQI @ iter = {iter}")
