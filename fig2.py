import math 
import numpy as np 
from numba import jit
from scipy.special import dawsn
import matplotlib.pyplot as plt
#导入计算需要的库

#常数导入
from numpy import pi
M0 = 1.989e33   #mass / g
M = 10**(-2)*M0     #mass / g
ka = 0.2 # cm^2 / g
f1 = 10**(-3)
f2 = 10**(-4)
f3 = 10**(-5)
tc = 0.975e5 # s
V = 1e10     # cm / s
c = 3e10   # light velocity    unit:cm/s
beta = V/c

L01 = (3*f1*M*c**2)/(4*beta*tc)
L02 = (3*f2*M*c**2)/(4*beta*tc)
L03 = (3*f3*M*c**2)/(4*beta*tc)

h = 1000000 #设置步长
t = np.linspace(0.001, 172800, h)
x = t/tc
T = np.linspace(0.0000001, 2, h)

Dx = dawsn(x)

#f1 = 
L1 = L01*np.sqrt(8*beta/3)*Dx
logL1 = []
for i in L1:
    logL1.append(math.log(i, 10))
    
#f2
L2 = L02*np.sqrt(8*beta/3)*Dx
logL2 = []
for i in L2:
    logL2.append(math.log(i, 10))
    
#f3
L3 = L03*np.sqrt(8*beta/3)*Dx
logL3 = []
for i in L3:
    logL3.append(math.log(i, 10))


# Figure&Axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
#ax.grid()
ax.set_title(r"Fig2", fontsize=20)
ax.set_xlabel(r"t [days]", fontsize=15)
ax.set_ylabel(r"logL [erg $s^{-1}$]", fontsize=15)
ax.set_xlim(0, 2)
ax.set_ylim(41, 45)

ax.plot(T, logL1, label = r"logf = -3")
ax.plot(T, logL2, label = r"logf = -4")
ax.plot(T, logL3, label = r"logf = -5")

plt.legend()
plt.show()

