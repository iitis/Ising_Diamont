from hamiltonian import *
from Wigner import *
import matplotlib.pylab as plt

#plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)

t1=t2=np.linspace(0,np.pi/2,30)
f1=f2=np.linspace(0,np.pi,30)
usePBC = True

s=1
d0 = 2 

d=d0*d1*d0
N=3

h=np.linspace(0,10,50)
#v=list(map(lambda h:Hamiltonian(h,s,N),h))
b=[]
for x in h:
    v=list(map(lambda t1,f1 : wigner(t1,t1,f1,f1,x,s,N),t1,f1))
    b+=[abs(sum(c for c in v if c<0))]
# print(b)
# print(len(b))
# v=lambda h:QI(h,s,N)
# dv=nd.Derivative(v)
# df=dv(h)
plt.plot(h,b)
plt.show()