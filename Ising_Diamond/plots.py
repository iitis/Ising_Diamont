from correlation_functions import *
from wigner import *
import matplotlib.pylab as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)

#parameters  for the (1/2,1/2) model
#s='1/2'; beta=np.float128(200); h0=h=1e-4
# x = np.linspace(0.,3*np.pi/2,200)
# y = np.linspace(0., 2*np.pi,200)
# X, Y = np.meshgrid(x, y)
# J=-np.sin(X)
# Jz=-np.sin(Y)
# Jp=4*np.cos(Y)

#parameters for the (1/2,1) model
s='1'; beta=np.float128(200); h0=h=1e-4
x = np.linspace(1.7,4.5,100)
y = np.linspace(1.7, 4.5,100)
X, Y = np.meshgrid(x, y)
J=Jz=np.sin(X)
Jp=4*np.sin(Y)

cmap = 'viridis'
W=np.vectorize(SxSx)
plt.contourf(X, Y, W(s,beta,J,h0,Jz,h,Jp), 100, cmap=cmap)
cb=plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$y$',fontsize=20)
plt.tight_layout()
plt.savefig("data_diamond.pdf")
plt.show()