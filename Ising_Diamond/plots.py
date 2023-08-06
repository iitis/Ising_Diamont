from correlation_functions import *
import matplotlib.pylab as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)

x= np.linspace(0, 2*np.pi, 50)
y= np.linspace(0, 2*np.pi, 50)
J=-np.sin(x)
Jz=-np.sin(y)
Jp=2*np.cos(y)

X, Y = np.meshgrid(x, y)
cmap = 'viridis'
W=np.vectorize(rho)
plt.contourf(X, Y, W(s,beta,X,h0,Y,h,0,Y), 100, cmap=cmap)
cb=plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$y$',fontsize=20)
plt.tight_layout()
plt.show()