from correlation_functions import *
import matplotlib.pylab as plt

#plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)


x= np.linspace(0, 5, 50)
y= np.linspace(0, 4, 50)

J=np.sin(x); Jz=np.sin(y); Jp=4*np.cos(y); Jm=4*np.cos(x); r=1; h=h0=0; beta=10

X, Y = np.meshgrid(x, y)
cmap = 'viridis'
W=sigma_i_sigma_j(beta,r, X, h, Y, h, X, Y)
plt.contourf(X, Y, W, 100, cmap=cmap)
cb=plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$y$',fontsize=20)
plt.tight_layout()
plt.show()