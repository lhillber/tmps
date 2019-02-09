#! user/bin/python3
import numpy as np

class Bfunc:
    def __init__(self, sol):
        self.func = getattr(self, sol)

    def __call__(self, r, r0, *args, **kwargs):
        return self.transform(r, r0, self.func, *args, **kwargs)

    def transform(self, r, r0, func, *args, **kwargs):
        r = np.array(r)
        r0 = np.array(r0)
        r = r - r0
        return func(r, *args, **kwargs)

    @staticmethod
    def line(r, L, I=1):
        return I * L * r

    @staticmethod
    def loop(r, R, I=1):
        return I * R*R * r

if __name__ == '__main__':
    r = [1,1,1]
    r0 = [0.5, 0.5, 0.5]
    L = 2
    R = 2
    B = Bfunc('loop')(r, r0, R, I=1)
    print(B)
    B = Bfunc('line')(r, r0, L=L, I=1)
    print(B)
