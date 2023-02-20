from scipy.optimize import fsolve
import numpy as np

def func(x):
    return (x - 1.2) * 2 * np.exp(1.2 * 2 * x) + (x + 1.2) * np.exp(-1.2 * 2 *x)

sol_fsolve = fsolve(func, [-1, 1])
print(sol_fsolve[0])