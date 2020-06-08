from numpy import array, concatenate, dot
from numpy.linalg import inv


class Trajectory:
    def __init__(self, t0, tf, q0, qf, n=1):
        self.n = n
        self.t0 = t0
        self.tf = tf
        self.q0 = q0
        self.qf = qf

        try:
            if n == 1:
                self.a = self.linear()
                self.ap = self.a[1]
                self.app = 0.0
            elif n == 3:
                self.a = self.cubic()
                self.ap = array([self.a[1], 2.0 * self.a[2], 3.0 * self.a[3]])
                self.app = array([2.0 * self.a[2], 6.0 * self.a[3]])
            elif n == 5:
                self.a = self.quintic()
                self.ap = array([self.a[1], 2.0 * self.a[2], 3.0 * self.a[3], 4.0 * self.a[4], 5.0 * self.a[5]])
                self.app = array([2.0 * self.a[2], 6.0 * self.a[3], 12.0 * self.a[4], 20.0 * self.a[5]])
        except ValueError:
            print("The order for the trajectory should be 1, 3, or 5")

    def linear(self):
        A = array([[1.0, self.t0], [1.0, self.tf]])

        return inv(A) @ array([self.q0, self.qf])

    def cubic(self):
        A = array([[1.0, self.t0, self.t0**2, self.t0**3],
                   [0.0, 1.0, 2.0 * self.t0, 3.0 * self.t0**2],
                   [1.0, self.tf, self.tf**2, self.tf**3],
                   [0.0, 1.0, 2.0 * self.tf, 3.0 * self.tf**2]])

        return inv(A) @ concatenate((self.q0, self.qf))

    def quintic(self):
        A = array([[1.0, self.t0, self.t0 ** 2, self.t0 ** 3, self.t0 ** 4, self.t0 ** 5],
                   [0.0, 1.0, 2.0 * self.t0, 3.0 * self.t0 ** 2, 4.0 * self.t0**3, 5.0 * self.t0**4],
                   [0.0, 0.0, 2.0, 6.0 * self.t0, 12.0 * self.t0**2, 20.0 * self.t0**3],
                   [1.0, self.tf, self.tf ** 2, self.tf ** 3, self.tf ** 4, self.tf ** 5],
                   [0.0, 1.0, 2.0 * self.tf, 3.0 * self.tf ** 2, 4.0 * self.tf**3, 5.0 * self.tf**4],
                   [0.0, 0.0, 2.0, 6.0 * self.tf, 12.0 * self.tf**2, 20.0 * self.tf**3]])

        return inv(A) @ concatenate((self.q0, self.qf))

    def q(self, t):
        if self.n == 1:
            return dot(self.a, array([1.0, t]))
        elif self.n == 3:
            return dot(self.a, array([1.0, t, t**2, t**3]))
        elif self.n == 5:
            return dot(self.a, array([1.0, t, t**2, t**3, t**4, t**5]))

    def qp(self, t):
        if self.n == 1:
            return self.ap * t
        elif self.n == 3:
            return dot(self.ap, array([1.0, t, t**2]))
        elif self.n == 5:
            return dot(self.ap, array([1.0, t, t**2, t**3, t**4]))

    def qpp(self, t):
        if self.n == 1:
            return self.app * t
        elif self.n == 3:
            return dot(self.app, array([1.0, t]))
        elif self.n == 5:
            return dot(self.app, array([1.0, t, t ** 2, t ** 3]))
