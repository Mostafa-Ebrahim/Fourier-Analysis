import numpy as np
import matplotlib.pyplot as plt

#* Define domain 
dx = 0.01
L = 2*np.pi
x = np.arange(0,L+dx,dx)
n = len(x)
nquart = int(np.floor(n/4))

########################################################################

#* Define disceret function 
f = np.zeros_like(x)
f[nquart : 3*nquart] = 1
plt.plot(x, f, color='k', LineWidth = 2)

########################################################################

#* Compute Fourier series 
A0 = np.sum(f * np.ones_like(x)) * dx * 2 / L
fourierSeries = A0/2 * np.ones_like(f)

for k in range(1,101):
    Ak = np.sum(f * np.cos(2*np.pi*k*x/L)) * dx * 2 / L
    Bk = np.sum(f * np.sin(2*np.pi*k*x/L)) * dx * 2 / L
    fourierSeries += Ak*np.cos(2*k*np.pi*x/L) + Bk*np.sin(2*k*np.pi*x/L)
plt.plot(x, fourierSeries, LineWidth = 1.5, color = 'red')

# Customize the major grid
plt.grid(b=True, which='major', color='#666666', linestyle='-')

# Customize the minor grid
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.show()