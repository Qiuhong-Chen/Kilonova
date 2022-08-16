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
f = 10**(-3)
tc = 0.975e5 # s
V = 1e10     # cm / s
c = 3e10   # light velocity    unit:cm/s
beta = V/c
L0 = (3*f*M*c**2)/(4*beta*tc)
T1 = 2.80e4


h = 1000000 #设置步长
t = np.linspace(0.000001, 97500, h)
tau = t/tc

x = np.sqrt(3/(8*beta))*tau
Dx = dawsn(x)

U = np.sqrt(8*beta/3)*(1/tau**4)*Dx

#logL
L = L0*tau**4*U
logL = []
for i in L:
    logL.append(math.log(i, 10))
    
#logT_eff
T_eff = T1*tau**(1/2)*U**(1/4)
logT_eff = []
for i in T_eff:
    logT_eff.append(math.log(i, 10))


# Figure&Axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
#ax.grid()
ax.set_title(r"Fig3", fontsize=20)
ax.set_xlabel(r"log $T_{eff}$ [K]", fontsize=15)
ax.set_ylabel(r"log L [erg $s^{-1}$]", fontsize=15)
ax.set_xlim(4.1, 4.8)
ax.set_ylim(41, 45)

ax.invert_xaxis()

ax.plot(logT_eff, logL)


#plt.legend()
plt.show()




