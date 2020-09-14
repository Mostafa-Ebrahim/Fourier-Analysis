import numpy as np
import matplotlib.pyplot as plt

#* Define domain
dx = 0.001
L = (np.pi)
x = L * np.arange(-1+dx, 1+dx, dx)
n = len(x)
nquart = int(np.floor(n/4))

########################################################################

#* Define triangular hat function
f = np.zeros_like(x)    # make an array of zeros (takes x and make it zeros)
f[nquart : 2*nquart] = (4/n)*np.arange(1,nquart+1)
f[2*nquart : 3*nquart] = np.ones(nquart) - (4/n)*np.arange(0,nquart)

fig, ax = plt.subplots()

########################################################################

#* Compute Fourier series
A0 = np.sum(f * np.ones_like(x)) * dx
fourierSeries = A0/2

A = np.zeros(50)
B = np.zeros(50)
for k in range(50):
    A[k] = np.sum(f * np.cos(np.pi*(k+1)*x/L)) * dx # Inner product
    B[k] = np.sum(f * np.sin(np.pi*(k+1)*x/L)) * dx
    fourierSeries += A[k]*np.cos((k+1)*np.pi*x/L) + B[k]*np.sin((k+1)*np.pi*x/L)
    ax.plot(x, fourierSeries)

# Customize the major grid
plt.grid(b=True, which='major', color='#666666', linestyle='-')

# Customize the minor grid
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.show()