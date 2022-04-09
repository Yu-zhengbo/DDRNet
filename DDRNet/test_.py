import numpy as np
import matplotlib.pyplot as plt
T1 = 29  # 波动周期
T2 = 20  # 半周期
lr_lambda = lambda x: ((np.cos(2 * np.pi * x / T1) + 1) * (1 - 1E-3) / 2 + 1e-3) * np.exp(-x / T2 * 0.693147)

y = []
for i in range(1,100):
    y.append(lr_lambda(i))
plt.plot(y)
print(y)
plt.show()