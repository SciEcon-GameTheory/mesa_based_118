import random
import numpy as np

cp = np.array([0.25, 0.75])
print(cp.ravel())
print(np.random.choice([1], p = cp.ravel()))