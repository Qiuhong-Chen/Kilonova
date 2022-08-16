import math 
import numpy as np 
from numba import jit
from scipy.special import dawsn
import matplotlib.pyplot as plt

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
###########main
###a1 = 1000
t01 = 49991
t1 = np.linspace(t01, 172800, h)
T1 = np.linspace(t01/86400, 2, h)
tau1 = t1/tc


x01 = a1*np.sqrt((2*beta)/3)
x1 = np.sqrt(3/(8*beta))*(tau1-(4/3)*a1*beta)

D01 = dawsn(x01)
C11 = (4/3)*a1*beta*(-1+np.exp((-2/3)*a1**2*beta)+a1*np.sqrt((8*beta)/3)*D01)

Dx1 = dawsn(x1)
L1 = L0*C11*np.exp((-3*tau1**2)/(8*beta)) + (4/3)*L0*a1*beta*np.exp(-a1*tau1)*(1-np.exp((-3/(8*beta))*(tau1-(4/3)*a1*beta))+a1*np.sqrt(8*beta/3)*Dx1)

logL1 = []
for i in L1:
    logL1.append(math.log(i, 10))


#####a2 = 100
t02 = 56672
t2 = np.linspace(t02, 172800, h)
T2 = np.linspace(t02/86400, 2, h)
tau2 = t2/tc

x02 = a2*np.sqrt((2*beta)/3)
x2 = np.sqrt(3/(8*beta))*(tau2-(4/3)*a2*beta)

D02 = dawsn(x02)
C12 = (4/3)*a2*beta*(-1+np.exp((-2/3)*a2**2*beta)+a2*np.sqrt((8*beta)/3)*D02)

Dx2 = dawsn(x2)
L2 = L0*C12*np.exp((-3*tau2**2)/(8*beta)) + (4/3)*L0*a2*beta*np.exp(-a2*tau2)*(1-np.exp((-3/(8*beta))*(tau2-(4/3)*a2*beta))+a2*np.sqrt(8*beta/3)*Dx2)

logL2 = []
for i in L2:
    logL2.append(math.log(i, 10))


###a3 = 10
t03 = 83745
t3 = np.linspace(t03, 172800, h)
T3 = np.linspace(t03/86400, 2, h)
tau3 = t3/tc

x03 = a3*np.sqrt((2*beta)/3)
x3 = np.sqrt(3/(8*beta))*(tau3-(4/3)*a3*beta)

D03 = dawsn(x03)
C13 = (4/3)*a3*beta*(-1+np.exp((-2/3)*a3**2*beta)+a3*np.sqrt((8*beta)/3)*D03)

Dx3 = dawsn(x3)
L3 = L0*C13*np.exp((-3*tau3**2)/(8*beta)) + (4/3)*L0*a3*beta*np.exp(-a3*tau3)*(1-np.exp((-3/(8*beta))*(tau3-(4/3)*a3*beta))+a3*np.sqrt(8*beta/3)*Dx3)

logL3 = []
for i in L3:
    logL3.append(math.log(i, 10))


###a4 = 1
t04 = 33180
t4 = np.linspace(t04, 172800, h)
T4 = np.linspace(t04/86400, 2, h)
tau4 = t4/tc


x04 = a4*np.sqrt((2*beta)/3)
x4 = np.sqrt(3/(8*beta))*(tau4-(4/3)*a4*beta)


D04 = dawsn(x04)
C14 = (4/3)*a4*beta*(-1+np.exp((-2/3)*a4**2*beta)+a4*np.sqrt((8*beta)/3)*D04)

Dx4 = dawsn(x4)
L4 = L0*C14*np.exp((-3*tau4**2)/(8*beta)) + (4/3)*L0*a4*beta*np.exp(-a4*tau4)*(1-np.exp((-3/(8*beta))*(tau4-(4/3)*a4*beta))+a4*np.sqrt(8*beta/3)*Dx4)

logL4 = []
for i in L4:
    logL4.append(math.log(i, 10))



###a5 = 0.1
t05 = 4500
t5 = np.linspace(t05, 172800, h)
T5 = np.linspace(t05/86400, 2, h)
tau5 = t5/tc


x05 = a5*np.sqrt((2*beta)/3)
x5 = np.sqrt(3/(8*beta))*(tau5-(4/3)*a5*beta)


D05 = dawsn(x05)
C15 = (4/3)*a5*beta*(-1+np.exp((-2/3)*a5**2*beta)+a5*np.sqrt((8*beta)/3)*D05)

Dx5 = dawsn(x5)
L5 = L0*C15*np.exp((-3*tau5**2)/(8*beta)) + (4/3)*L0*a5*beta*np.exp(-a5*tau5)*(1-np.exp((-3/(8*beta))*(tau5-(4/3)*a5*beta))+a5*np.sqrt(8*beta/3)*Dx5)

logL5 = []
for i in L5:
    logL5.append(math.log(i, 10))


###a6 = 0.01
t06 = 500
t6 = np.linspace(t06, 172800, h)
T6 = np.linspace(t06/86400, 2, h)
tau6 = t6/tc


x06 = a6*np.sqrt((2*beta)/3)
x6 = np.sqrt(3/(8*beta))*(tau6-(4/3)*a6*beta)


D06 = dawsn(x06)
C16 = (4/3)*a6*beta*(-1+np.exp((-2/3)*a6**2*beta)+a6*np.sqrt((8*beta)/3)*D06)

Dx6 = dawsn(x6)
L6 = L0*C16*np.exp((-3*tau6**2)/(8*beta)) + (4/3)*L0*a6*beta*np.exp(-a6*tau6)*(1-np.exp((-3/(8*beta))*(tau6-(4/3)*a6*beta))+a6*np.sqrt(8*beta/3)*Dx6)

logL6 = []
for i in L6:
    logL6.append(math.log(i, 10))




# Figure&Axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
#ax.grid()
ax.set_title(r"Fig1", fontsize=20)
ax.set_xlabel(r"t [days]", fontsize=15)
ax.set_ylabel(r"logL [erg $s^{-1}$]", fontsize=15)
ax.set_xlim(0, 2)
ax.set_ylim(41, 45)

ax.plot(T1, logL1, label=r"$log(t_{c}/t_{rad}) = 3$")
ax.plot(T2, logL2, label=r"$log(t_{c}/t_{rad}) = 2$")
ax.plot(T3, logL3, label=r"$log(t_{c}/t_{rad}) = 1$")
ax.plot(T4, logL4, label=r"$log(t_{c}/t_{rad}) = 0$")
ax.plot(T5, logL5, label=r"$log(t_{c}/t_{rad}) = -1$")
ax.plot(T6, logL6, label=r"$log(t_{c}/t_{rad}) = -2$")

plt.legend()
plt.show()









