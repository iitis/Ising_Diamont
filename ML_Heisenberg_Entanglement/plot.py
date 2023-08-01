from code import *
import matplotlib.pylab as plt

#plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)

J=1
D=1
delta=np.linspace(0,10,50)

data=list(map(lambda delta: rho(D,J,delta),delta))

plt.plot(delta,data)

plt.xlabel(r'$\Delta$',fontsize=20)
plt.ylabel(r'$C$',fontsize=20)
plt.tight_layout()
plt.show()