import math 
import numpy as np 
from numba import jit
from scipy.special import dawsn
import matplotlib.pyplot as plt
#导入计算需要的库

#常数导入
M0 = 1.989e33   #mass / g
M = 10**(-2)*M0     #mass / g
f = 10**(-3)
tc = 0.975e5 # s
V = 1e10     # cm / s
c = 3e10   # light velocity    unit:cm/s
beta = V/c
L0 = (3*f*M*c**2)/(4*beta*tc)
a1 = 1000
a2 = 100
a3 = 10
a4 = 1
a5 = 0.1
a6 = 0.01


h = 1000000 #设置步长
t0 = 0.01
t = np.linspace(t0, 172800, h)
T = t/86400
tau = t/tc


#############################################################
###a1 = 1000
x01 = a1*np.sqrt((2*beta)/3)
x1 = np.sqrt(3/(8*beta))*(tau-(4/3)*a1*beta)

D01 = dawsn(x01)
Dx1 = dawsn(x1)

C1 = (4/3)*a1*beta*(-1 + a1*np.sqrt((8*beta)/3)*D01)
#C11 = (4/3)*a1*beta*(-1+np.exp((-2/3)*a1**2*beta)+a1*np.sqrt((8*beta)/3)*D01)
U1 = C1/(tau**4*np.exp((3*tau**2)/(8*beta))) + 4*a1*beta*np.exp(-a1*tau)/(3*tau**4)*(1 + a1*np.sqrt(8*beta/3)*Dx1)

L1 = L0*tau**4*U1
logL1 = []
for i in L1:
    logL1.append(math.log(i, 10))

###########################################################
####a1 = 100
x02 = a2*np.sqrt((2*beta)/3)
x2 = np.sqrt(3/(8*beta))*(tau-(4/3)*a2*beta)

D02 = dawsn(x02)
Dx2 = dawsn(x2)

C2 = (4/3)*a2*beta*(-1 + a2*np.sqrt((8*beta)/3)*D02)
#C11 = (4/3)*a1*beta*(-1+np.exp((-2/3)*a1**2*beta)+a1*np.sqrt((8*beta)/3)*D01)
U2 = C2/(tau**4*np.exp((3*tau**2)/(8*beta))) + 4*a2*beta*np.exp(-a2*tau)/(3*tau**4)*(1 + a2*np.sqrt(8*beta/3)*Dx2)

L2 = L0*tau**4*U2
logL2 = []
for i in L2:
    logL2.append(math.log(i, 10))

#######################################################
###a1 = 10
x03 = a3*np.sqrt((2*beta)/3)
x3 = np.sqrt(3/(8*beta))*(tau-(4/3)*a3*beta)

D03 = dawsn(x03)
Dx3 = dawsn(x3)

C3 = (4/3)*a3*beta*(-1 + a3*np.sqrt((8*beta)/3)*D03)
#C11 = (4/3)*a1*beta*(-1+np.exp((-2/3)*a1**2*beta)+a1*np.sqrt((8*beta)/3)*D01)
U3 = C3/(tau**4*np.exp((3*tau**2)/(8*beta))) + 4*a3*beta*np.exp(-a3*tau)/(3*tau**4)*(1 + a3*np.sqrt(8*beta/3)*Dx3)

L3 = L0*tau**4*U3
logL3 = []
for i in L3:
    logL3.append(math.log(i, 10))

###################################################
###a1 = 1
x04 = a4*np.sqrt((2*beta)/3)
x4 = np.sqrt(3/(8*beta))*(tau-(4/3)*a4*beta)

D04 = dawsn(x04)
Dx4 = dawsn(x4)

C4 = (4/3)*a4*beta*(-1 + a4*np.sqrt((8*beta)/3)*D04)
#C11 = (4/3)*a1*beta*(-1+np.exp((-2/3)*a1**2*beta)+a1*np.sqrt((8*beta)/3)*D01)
U4 = C4/(tau**4*np.exp((3*tau**2)/(8*beta))) + 4*a4*beta*np.exp(-a4*tau)/(3*tau**4)*(1 + a4*np.sqrt(8*beta/3)*Dx4)

L4 = L0*tau**4*U4
logL4 = []
for i in L4:
    logL4.append(math.log(i, 10))

#########################################################
###a1 = 0.1
x05 = a5*np.sqrt((2*beta)/3)
x5 = np.sqrt(3/(8*beta))*(tau-(4/3)*a5*beta)

D05 = dawsn(x05)
Dx5 = dawsn(x5)

C5 = (4/3)*a5*beta*(-1 + a5*np.sqrt((8*beta)/3)*D05)
#C11 = (4/3)*a1*beta*(-1+np.exp((-2/3)*a1**2*beta)+a1*np.sqrt((8*beta)/3)*D01)
U5 = C5/(tau**4*np.exp((3*tau**2)/(8*beta))) + 4*a5*beta*np.exp(-a5*tau)/(3*tau**4)*(1 + a5*np.sqrt(8*beta/3)*Dx5)

L5 = L0*tau**4*U5
logL5 = []
for i in L5:
    logL5.append(math.log(i, 10))

################################################################
###a1 = 0.01
x06 = a6*np.sqrt((2*beta)/3)
x6 = np.sqrt(3/(8*beta))*(tau-(4/3)*a6*beta)

D06 = dawsn(x06)
Dx6 = dawsn(x6)

C6 = (4/3)*a6*beta*(-1 + a6*np.sqrt((8*beta)/3)*D06)
#C11 = (4/3)*a1*beta*(-1+np.exp((-2/3)*a1**2*beta)+a1*np.sqrt((8*beta)/3)*D01)
U6 = C6/(tau**4*np.exp((3*tau**2)/(8*beta))) + 4*a6*beta*np.exp(-a6*tau)/(3*tau**4)*(1 + a6*np.sqrt(8*beta/3)*Dx6)

L6 = L0*tau**4*U6
logL6 = []
for i in L6:
    logL6.append(math.log(i, 10))


######绘制图形
# Figure&Axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
#ax.grid()
ax.set_title(r"Fig1", fontsize=20)
ax.set_xlabel(r"t [days]", fontsize=15)
ax.set_ylabel(r"logL [erg $s^{-1}$]", fontsize=15)
ax.set_xlim(0, 2)
ax.set_ylim(41, 45)

#ax.plot(t, L)
#ax.plot(t, L1, label=r"$log(t_{c}/t_{rad}) = 3$")
ax.plot(T, logL1, label=r"$log(t_{c}/t_{rad}) = 3$")
ax.plot(T, logL2, label=r"$log(t_{c}/t_{rad}) = 2$")
ax.plot(T, logL3, label=r"$log(t_{c}/t_{rad}) = 1$")
ax.plot(T, logL4, label=r"$log(t_{c}/t_{rad}) = 0$")
ax.plot(T, logL5, label=r"$log(t_{c}/t_{rad}) = -1$")
ax.plot(T, logL6, label=r"$log(t_{c}/t_{rad}) = -2$")

plt.legend()
plt.show()