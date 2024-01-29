"""
Code from Cubic spline planner
Author: Atsushi Sakai(@Atsushi_twi)
"""
import math
import numpy as np
import bisect


class CubicSpline1D:
    """
    1D Cubic Spline class
    Parameters
    ----------
    x : list
        x coordinates for data points. This x coordinates must be
        sorted in ascending order.
    y : list
        y coordinates for data points
    """

    def __init__(self, x, y):
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) - h[i] / 3.0 * (
                2.0 * self.c[i] + self.c[i + 1]
            )
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x):
        """
        Calc `y` position for given `x`.
        if `x` is outside the data point's `x` range, return None.
        Returns
        -------
        y : float
            y position for given x.
        """
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = (
            self.a[i] + self.b[i] * dx + self.c[i] * dx**2.0 + self.d[i] * dx**3.0
        )

        return position

    def calc_first_derivative(self, x):
        """
        Calc first derivative at given x.
        if x is outside the input x, return None
        Returns
        -------
        dy : float
            first derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2.0
        return dy

    def calc_second_derivative(self, x):
        """
        Calc second derivative at given x.
        if x is outside the input x, return None
        Returns
        -------
        ddy : float
            second derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x[:-1], x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h, a):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = (
                3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
            )
        return B


class CubicSpline2D:
    """
    Cubic CubicSpline2D class
    Parameters
    ----------
    x : list
        x coordinates for data points.
    y : list
        y coordinates for data points.
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        Returns
        -------
        x : float
            x position for given s.
        y : float
            y position for given s.
        """
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        Returns
        -------
        k : float
            curvature for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.
        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        if s < self.s[0]:
            s+= self.s[-1]
        if s >= self.s[-1]:
            s-= self.s[-1]
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        yaw = math.atan2(dy, dx)
        return yaw

    def calc_smooth_curvature(self, s, window=1.0):
        """
        calc curvature and smooth it with average filter
        """
        ss = np.arange(s - window / 2.0, s + window / 2.0, 0.1)
        ks = [self.calc_curvature(s % self.s[-1]) for s in ss]
        return np.mean(ks)

    def calc_smooth_yaw(self, s, window=1.0):
        """
        calc curvature and smooth it with average filter
        """
        ss = np.arange(s - window / 2.0, s + window / 2.0, 0.1)
        ss = [s % self.s[-1] for s in ss]
        yaws = [self.calc_yaw(si) for si in ss]
        return np.mean(yaws)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    #y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    centerline = np.load("centerline_extend.npy")
    x = list(centerline[:,1])
    #print(len(x))
    y = list(centerline[:,2])
    ds = 0.1  # [m] distance of each interpolated points

    sp = CubicSpline2D(x, y)
    s = centerline[:, 0] # np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk, srk = [], [], [], [], []
    my_yaw = []
    my_rk = []
    for num in range(s.shape[0]):
        i_s = s[num]
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        my_yaw.append(centerline[num][5])
        rk.append(sp.calc_curvature(i_s))
        srk.append(sp.calc_smooth_curvature(i_s,1.0))
        my_rk.append(centerline[num][4])


    plt.subplots(1)
    plt.plot(x, y, "xb", label="Data points")
    plt.plot(rx, ry, "-r", label="Cubic spline path")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.subplots(1)
    plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    plt.subplots(1)
    plt.plot(s, [np.rad2deg(iyaw) for iyaw in my_yaw], "-r", label="myyaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.subplots(1)
    plt.plot(s, srk, "-r", label="sm_curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.subplots(1)
    plt.plot(s, my_rk, "-r", label="my_curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()
