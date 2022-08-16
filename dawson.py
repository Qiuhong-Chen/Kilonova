
import numpy as np
from scipy.special import dawsn
import matplotlib.pyplot as plt

# 区间[-10,10]，512分割
x = np.linspace(-10, 10, 10000)

# 积分元
Dx = dawsn(x)

# Figure&Axes
fig = plt.figure(figsize=(6.5, 5))
ax = fig.add_subplot(111)
ax.grid()
ax.set_title("Dawson integral", fontsize=16)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("D(x)", fontsize=15)
ax.set_xlim(0, 10)
ax.set_ylim(-0.6, 0.6)

ax.plot(x, Dx, color = "blue")

plt.show()
