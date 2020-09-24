import numpy as np
import matplotlib.pyplot as plt
import datapane as dp


#* Create the function and its derivative 
n = 128
L = 30
dx = L/n
x = np.arange(-L/2, L/2, dx, dtype = complex)
f = np.cos(x) * np.exp(-np.power(x,2)/25) # Function
df = -(np.sin(x) * np.exp(-np.power(x,2)/25) + (2/25)*x*f) # Derivative


#* Computing derivative using FFT 
fhat = np.fft.fft(f)
kappa = 2*np.pi/L * np.arange(-n/2, n/2)
kappa = np.fft.fftshift(kappa)
dfhat = kappa * fhat * 1j
dfFFT = np.real(np.fft.ifft(dfhat))


#* Plots 
plt.plot(x,df.real,color='k',LineWidth=2,label='True Derivative')
plt.plot(x,dfFFT.real,'--',color='r',LineWidth=1.5,label='FFT Derivative')
plt.show()