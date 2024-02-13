from correlation_functions import *
from funcs import *
import matplotlib.pylab as plt
from multiprocessing import Pool
from tqdm import tqdm

import matplotlib.font_manager as fm
fm.fontManager.addfont("/home/zmzaouali/times.ttf")
from multiprocessing import Pool
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)

# parameters  for the (1/2,1/2) model
# s='1/2'; beta=np.float128(1e3); h0=h=1e-4
# x = np.linspace(1e-4,3*np.pi/2,500)
# y = np.linspace(1e-4, 2*np.pi,500)
# X, Y = np.meshgrid(x, y)
# J=-np.sin(X)
# Jz=-np.sin(Y)
# Jp=4*np.cos(Y)
# t=np.pi/3; f=np.pi

# t=f=0
#parameters for the (1/2,1) model
s='1'; beta=np.float128(450); h0=h=1e-4
x = np.linspace(np.pi/2,3*np.pi/2,100)
y = np.linspace(np.pi/2,3*np.pi/2,100)
X, Y = np.meshgrid(x, y)
J=Jz=np.sin(X)
Jp=4*np.sin(Y)
# t=0.; f=0.

# mc = monte_carlo_integration(s,beta,J,h0,Jz,h,Jp)
# mc.compute()
# results=mc.negativity
params_list = [(s, beta, xi, h0, yi, h, zi) for (xi, yi, zi) in zip(J.flatten(), Jz.flatten(), Jp.flatten())]
with Pool(processes=36) as pool:
    results = list(tqdm(pool.imap(nwf_monte_carlo, params_list), total=len(params_list)))
    np.savez("/home/zmzaouali/TN_QMS/Ising_Diamond/data/negativity_wigner_spin_one_full_100k.npz",results)
# results=np.load("/home/zmzaouali/TN_QMS/Ising_Diamond/data/phase_diagram_spin_one.npz")['arr_0']
# results_vals = np.array(results).reshape(X.shape)
# cmap = 'inferno'
# plt.contourf(X, Y, results_vals, 100, cmap=cmap)
# plt.text(np.pi/2,1.5,'FM',fontsize=15, color="white")
# plt.text(np.pi/2,3.5,r"$QFO_{III}$",fontsize=15, color="white")
# plt.text(np.pi/2,3*np.pi/2,'FM',fontsize=15, color="white")
# plt.text(np.pi/2,5.8,r"$QFO_{IV}$",fontsize=15, color="white")
# plt.text(3.8,1.5,'FRU',fontsize=15, color="white")
# plt.text(3.8,np.pi,r"$FRU_{III}$",fontsize=15, color="white")
# plt.text(3.8,5.6,r"$FRU_{IV}$",fontsize=15, color="white")

# plt.text(4,np.pi,'FM',fontsize=15, color="black")
# plt.text(3.8,4.3,r"$QFO_{I}$",fontsize=15, color="white")
# plt.text(3.8,2,r"$QFO_{II}$",fontsize=15, color="white")
# plt.text(3.2,4.3,r"$QFO_{III}$",fontsize=15, color="white")
# plt.text(2.2,np.pi,'FRU',fontsize=15, color="white")

# #cb=plt.colorbar()
# #cb.ax.tick_params(labelsize=20)
# plt.xlabel(r'$x$',fontsize=20)
# plt.ylabel(r'$y$',fontsize=20)
# plt.tight_layout()
# plt.savefig("/home/zmzaouali/TN_QMS/Ising_Diamond/data/phase_diagram_spin_one.pdf")
# plt.show()