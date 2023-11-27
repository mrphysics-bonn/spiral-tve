import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

Gy = sym.Symbol('Gy')
g = sym.Symbol('G')
Gz = sym.Symbol('Gz')
x = sym.Symbol('x')
Delta = sym.Symbol('Delta')
delta = sym.Symbol('delta')
t180 = sym.Symbol('t180')
t = sym.Symbol('t')
tp = sym.Symbol('tp')
tv = sym.Symbol('tv')
tn = sym.Symbol('tn')
a = sym.Symbol('a')

#%% qMASmod

# Integral byy

I1 = sym.integrate((Gy/x*(t**2/2-delta*t)+Gy/x*delta**2/2)**2, (t, delta, delta+x))
I2 = (Gy*x/2)**2*tp/2+(Gy*tp/(2*sym.pi))**2*tp/4+(Gy*x)*(Gy*tp/(2*sym.pi))*(tp/(2*sym.pi))*2
I3 = sym.integrate((Gy*x/2-Gy*t+Gy/x*(t**2/2-(tv-x)*t)+Gy/x*(tv-x)**2/2+Gy*(tv-x))**2,(t,tv-x,tv))
I4 = sym.integrate((-Gy/x*(t**2/2-tn*t)-Gy/x*tn**2/2)**2, (t, tn, tn+x))
I5 = (Gy*x/2)**2*tp/2+(Gy*tp/(2*sym.pi))**2*tp/4+(Gy*x)*(Gy*tp/(2*sym.pi))*(tp/(2*sym.pi))*2
I6 = sym.integrate((-Gy*x/2+Gy*t-Gy/x*(t**2/2-(Delta-x)*t)-Gy/x*(Delta-x)**2/2-Gy*(Delta-x))**2,(t,Delta-x,Delta))

Iges = I1+I2+I3+I4+I5+I6
Iges = sym.simplify(Iges)
Iges

# Integral bzz - block gradients

alpha = sym.Symbol('alpha')
g = Gz*3/2*tp/(2*sym.pi)*1/delta

I1 = sym.integrate((g*t)**2, (t,0,delta))
I2 = g**2*delta**2*tp/2-g*delta*Gz*tp/(2*sym.pi)*tp+3/4*(Gz*tp/(2*sym.pi))**2*tp
I3 = sym.integrate((g*delta-Gz*tp/sym.pi)**2, (t,tv,tn))
I4 = g**2*delta**2*tp/2-g*delta*Gz*tp/(2*sym.pi)*tp+3/4*(Gz*tp/(2*sym.pi))**2*tp
I5 = sym.integrate((g*delta-g*(t-Delta))**2, (t,Delta,Delta+delta))

Iges = I1+I2+I3+I4+I5
Iges = sym.simplify(Iges)
Iges

# Integral bzz - trapezoidal gradients

rho = sym.Symbol('rho')
delta_T = sym.Symbol('delta_T')
g = Gz*3/2*tp/(2*sym.pi)*1/delta

I1 = sym.integrate((g/rho*t**2/2)**2, (t,0,rho))
I2 = sym.integrate((g*rho/2+g*t-g*rho)**2, (t,rho,delta_T))
I3 = sym.integrate((g*rho/2+g*t-g*rho-g/rho*(t**2/2-delta_T*t)-g/rho*delta_T**2/2)**2, (t,delta_T,delta))
I4 = sym.integrate((g*rho/2+g*delta-g*rho-g/rho*(delta**2/2-delta_T*delta)-g/rho*delta_T**2/2)**2, (t,delta,delta+x))
I5 = sym.integrate((g*rho/2+g*delta-g*rho-g/rho*(delta**2/2-delta_T*delta)-g/rho*delta_T**2/2+Gz*tp/(2*sym.pi)*sym.cos(2*sym.pi/tp*(t-(delta+x)))-Gz*tp/(2*sym.pi))**2, (t,delta+x,tv-x))
I6 = sym.integrate((g*rho/2+g*delta-g*rho-g/rho*(delta**2/2-delta_T*delta)-g/rho*delta_T**2/2-Gz*tp/(sym.pi))**2, (t,tv-x,tn+x))
I7 = sym.integrate((g*rho/2+g*delta-g*rho-g/rho*(delta**2/2-delta_T*delta)-g/rho*delta_T**2/2-Gz*tp/(2*sym.pi)*sym.cos(2*sym.pi/tp*(t-(tn+x)))-Gz*tp/(2*sym.pi))**2, (t,tn+x,Delta-x))
I8 = sym.integrate((g*rho/2+g*delta-g*rho-g/rho*(delta**2/2-delta_T*delta)-g/rho*delta_T**2/2)**2, (t,Delta-x,Delta))
I9 = sym.integrate((g*rho/2+g*delta-g*rho-g/rho*(delta**2/2-delta_T*delta)-g/rho*delta_T**2/2-g/rho*(t**2/2-Delta*t)-g/rho*Delta**2/2)**2, (t,Delta,Delta+rho))
I10 = sym.integrate((g*rho/2+g*delta-g*rho-g/rho*(delta**2/2-delta_T*delta)-g/rho*delta_T**2/2-g/rho*((Delta+rho)**2/2-Delta*(Delta+rho))-g/rho*Delta**2/2-g*(t-(Delta+rho)))**2, (t,Delta+rho,Delta+delta_T))
I11 = sym.integrate((g*rho/2+g*delta-g*rho-g/rho*(delta**2/2-delta_T*delta)-g/rho*delta_T**2/2-g/rho*((Delta+rho)**2/2-Delta*(Delta+rho))-g/rho*Delta**2/2-g*((Delta+delta_T)-(Delta+rho))+g/rho*((t)**2/2-(Delta+delta_T)*t)-g*t+g/rho*(Delta+delta_T)**2/2+g*(Delta+delta_T))**2, (t,Delta+delta_T,Delta+delta))

Iges = I1+I2+I3+I4+I5+I6+I7+I8+I9+I10+I11
Iges = sym.simplify(Iges)
Iges

#Iges = Gz**2*tp**2/(320*sym.pi**3*delta**2*rho**2)*(45*sym.pi*Delta*delta**4-180*sym.pi*Delta*delta**3*delta_T-60*sym.pi*Delta*delta**3*rho+270*sym.pi*Delta*delta**2*delta_T**2+120*sym.pi*Delta*delta**2*delta_T*rho+40*sym.pi*Delta*delta**2*rho**2+110*sym.pi*Delta*delta**2*rho**2-180*sym.pi*Delta*delta*delta_T**3-60*sym.pi*Delta*delta*delta_T**2*rho-180*sym.pi*Delta*delta*delta_T*rho**2-60*sym.pi*Delta*delta*rho**3+45*sym.pi*Delta*delta_T**4+90*sym.pi*Delta*delta_T**2*rho**2+45*sym.pi*Delta*rho**4-12*sym.pi*delta**5+60*sym.pi*delta**4*delta_T-60*sym.pi*delta**4*rho-120*sym.pi*delta**3*delta_T**2+120*sym.pi*delta**3*delta_T*rho+120*sym.pi*delta**3*rho*tn-120*sym.pi*delta**3*rho*tv+120*sym.pi*delta**2*delta_T**3-30*sym.pi*delta**2*delta_T**2*rho+90*sym.pi*delta**2*delta_T*rho**2-240*sym.pi*delta**2*delta_T*rho*tn+240*sym.pi*delta**2*delta_T*rho*tv-40*sym.pi*delta**2*rho**2*tn+40*sym.pi*delta**2*rho**2*tv-40*sym.pi*delta**2*rho**2*x-40*sym.pi*delta**2*rho**2*x+240*sym.pi*delta**2*rho**2*x-60*sym.pi*delta*delta_T**4-60*sym.pi*delta*delta_T**3*rho+120*sym.pi*delta*delta_T**2*rho*tn-120*sym.pi*delta*delta_T**2*rho*tv-60*sym.pi*delta*delta_T*rho**3-60*sym.pi*delta*rho**4+120*sym.pi*delta*rho**3*tn-120*sym.pi*delta*rho**3*tv+12*sym.pi*delta_T**5+30*sym.pi*delta_T**4*rho-30*sym.pi*delta_T**3*rho**2+30*sym.pi*delta_T**2*rho**3+18*sym.pi*rho**5)
Iges = Gz**2*tp**2*(45*sym.pi*Delta*delta**4 - 180*sym.pi*Delta*delta**3*delta_T - 60*sym.pi*Delta*delta**3*rho + 270*sym.pi*Delta*delta**2*delta_T**2 + 120*sym.pi*Delta*delta**2*delta_T*rho + 40*sym.pi*Delta*delta**2*rho**2 + 110*sym.pi*Delta*delta**2*rho**2 - 180*sym.pi*Delta*delta*delta_T**3 - 60*sym.pi*Delta*delta*delta_T**2*rho - 180*sym.pi*Delta*delta*delta_T*rho**2 - 60*sym.pi*Delta*delta*rho**3 + 45*sym.pi*Delta*delta_T**4 + 90*sym.pi*Delta*delta_T**2*rho**2 + 45*sym.pi*Delta*rho**4 - 12*sym.pi*delta**5 + 60*sym.pi*delta**4*delta_T - 60*sym.pi*delta**4*rho - 120*sym.pi*delta**3*delta_T**2 + 120*sym.pi*delta**3*delta_T*rho + 120*sym.pi*delta**3*rho*tn - 120*sym.pi*delta**3*rho*tv + 120*sym.pi*delta**2*delta_T**3 - 30*sym.pi*delta**2*delta_T**2*rho + 90*sym.pi*delta**2*delta_T*rho**2 - 240*sym.pi*delta**2*delta_T*rho*tn + 240*sym.pi*delta**2*delta_T*rho*tv - 40*sym.pi*delta**2*rho**2*tn + 40*sym.pi*delta**2*rho**2*tv - 40*sym.pi*delta**2*rho**2*x - 40*sym.pi*delta**2*rho**2*x + 240*sym.pi*delta**2*rho**2*x - 60*sym.pi*delta*delta_T**4 - 60*sym.pi*delta*delta_T**3*rho + 120*sym.pi*delta*delta_T**2*rho*tn - 120*sym.pi*delta*delta_T**2*rho*tv - 60*sym.pi*delta*delta_T*rho**3 - 60*sym.pi*delta*rho**4 + 120*sym.pi*delta*rho**3*tn - 120*sym.pi*delta*rho**3*tv + 12*sym.pi*delta_T**5 + 30*sym.pi*delta_T**4*rho - 30*sym.pi*delta_T**3*rho**2 + 30*sym.pi*delta_T**2*rho**3 + 18*sym.pi*rho**5)/(320*sym.pi**3*delta**2*rho**2)
Iges = sym.simplify(Iges)
Iges
str(Iges)
