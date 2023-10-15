from hamiltonian import *
from Wigner import *
import matplotlib.pylab as plt

import multiprocessing
from functools import partial
import progressbar

#plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)


t1=t2=np.linspace(0,np.pi,20)
f1=f2=np.linspace(0,2*np.pi,20)
s=1
N=4
h=np.linspace(0.0,6,100)
#w=list(map(lambda h: sum(list(map(lambda t1: sum(list(map(lambda f1 : wigner(t1,t1,f1,f1,h,s,N)*(1/np.pi)*np.sin(2*t1),f1))),t1))),h))
#np.savez('data_neg_s=%s_N=%s'%(s,N),w)

w=list(map(lambda h:state(h,s,N),h))

plt.plot(h,w)
plt.savefig("neg_ent.pdf")
plt.show()
# for x in s:
#     data=list(map(lambda h:EoF(h,x,N),h))
#     np.savez("EoF_s=%s_N=%s"%(x,N),data)

#data1=np.load("/home/zakaria/TN_QMS/data.npz")
#plt.plot(h,data1['arr_0'],label=r'$\frac{1}{2}-1$')

# plt.plot(h,data,label=r'$\frac{1}{2}-1$')
# plt.xlabel(r'$h$',fontsize=20)
# #plt.ylabel(r'$\mathcal{N}_W$',fontsize=20)
# plt.ylabel(r'EoF',fontsize=20)
# plt.legend(fontsize=20)
# plt.tight_layout()
# plt.savefig('data_neg_s=%s_N=%s.pdf'%(s,N))
# plt.show()