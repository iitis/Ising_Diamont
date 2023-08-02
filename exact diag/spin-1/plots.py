from hamiltonian import *
import matplotlib.pylab as plt

#plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)








h=np.linspace(0,4,100)
#v=list(map(lambda h:Hamiltonian(h,s,N),h))
v=list(map(lambda h:QI(h,s,N),h))
# v=lambda h:QI(h,s,N)
# dv=nd.Derivative(v)
# df=dv(h)
plt.plot(h,v)
plt.show()