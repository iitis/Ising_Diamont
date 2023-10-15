from correlation_functions import *
from wigner import *
import matplotlib.pylab as plt
from itertools import product
from multiprocessing import Pool
import time
pool = Pool(processes = 20)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)

#parameters  for the (1/2,1/2) model
# s='1/2'; beta=np.float128(300); h0=h=1e-4
# x = np.linspace(0.01,3*np.pi/2,300)
# y = np.linspace(0.01, 2*np.pi,300)
# X, Y = np.meshgrid(x, y)
# J=-np.sin(X)
# Jz=-np.sin(Y)
# Jp=4*np.cos(Y)
# t=f=0

#parameters for the (1/2,1) model
s='1'; beta=np.float128(300); h0=h=1e-4
x = np.linspace(1.7,4.7,300)
y = np.linspace(1.7, 4.7,300)
X, Y = np.meshgrid(x, y)
J=Jz=np.sin(X)
Jp=4*np.sin(Y)

params_list = [(s, beta, xi, h0, yi, h, zi) for (xi, yi, zi) in zip(J.flatten(), Jz.flatten(), Jp.flatten())]
with Pool(processes=20) as pool:
    results = pool.map(rho,params_list)
results_vals = np.array(results).reshape(X.shape)
np.savez("entanglement_negativity_full_spin_one",results_vals)

cmap = 'viridis'
plt.contourf(X, Y, results_vals, 100, cmap=cmap)
cb=plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$y$',fontsize=20)
plt.tight_layout()
plt.savefig("fig.pdf")
plt.show()
