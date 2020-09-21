import numpy as np
import matplotlib.pyplot as plt

n = 256
w = np.exp(-1j * 2 * np.pi / n)

DFT = np.zeros((n,n))

# Slow
for i in range(n):
    for k in range(n):
        DFT[i,k] = w**(i*k)
        
DFT = np.real(DFT)
        
plt.imshow(DFT)
plt.show()