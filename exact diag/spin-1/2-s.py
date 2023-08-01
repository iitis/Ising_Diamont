import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer
from qutip import *
from ncon import ncon
import matplotlib.pylab as plt

from toqito.state_props import negativity, log_negativity


#plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=15)

usePBC = True
s=1
d0 = 2 
d1=int(2*s+1)
d=d0*d1
N=5

def Hamiltonian(h,s,N):

    hloc = np.real(np.kron(np.array(sigmax()), np.array(jmat(s,'x')))+np.kron(np.array(sigmay()), np.array(jmat(s,'y')))+
            np.kron(np.array(sigmaz()), np.array(jmat(s,'z')))- h * (np.kron(np.array(sigmaz()), np.array(qeye(d1)))+
            np.kron(np.array(qeye(2)), np.array(jmat(s,'z'))))).reshape(2, d1, 2, d1)

    def doApplyHam(psiIn: np.ndarray, hloc: np.ndarray, N: int, usePBC: bool):
        d0 = hloc.shape[0]
        d1 = hloc.shape[1]
        d = d0*d1
        psiOut = np.zeros(psiIn.size)
        for k in range(N-1):
            psiOut += np.tensordot(hloc.reshape(d, d),
                                    psiIn.reshape(d**k, d, d**(N-1-k)),
                                    axes=[[1], [1]]).transpose(1, 0, 2).reshape(d**N)
        if usePBC:
            psiOut += np.tensordot(hloc.reshape(d,d),psiIn.reshape(d**(N-1), d),
                             axes=[[1], [1]]).transpose(1, 0).reshape(d**N)
        return psiOut

    def doApplyHamClosed(psiIn):
        return doApplyHam(psiIn, hloc, N, usePBC)

    H = LinearOperator(((d0*d1)**N, (d0*d1)**N), matvec=doApplyHamClosed)
    Energy, psi = eigsh(H, k=1, which='SA')
    return psi

def QI(h,s,N):
    psi_r=Hamiltonian(h,s,N).reshape(2*d1,d**(N-1))
    psi_c=np.conj(psi_r)
    #rdm=Qobj(ncon([psi_r,psi_c],((-1,1),(-2,1))))
    rdm=ncon([psi_r,psi_c],((-1,1),(-2,1)))
    #rdm.dims=[[2,d1],[2,d1]]
    return log_negativity(rdm,[2,d1])

h=np.linspace(0.1,5,100)

v=list(map(lambda h:QI(h,s,N),h))

plt.plot(h,v)
plt.show()

#print(Hamiltonian(1,1,5))