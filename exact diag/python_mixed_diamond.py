import numpy as np
from tqdm import tqdm
from time import sleep
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer
from ncon import ncon
from qutip import *
import matplotlib.pylab as plt


plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ['Times New Roman'],
})
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.tick_params(axis='both', which='major', labelsize=20)

d = 2 
sX = np.array([[0, 1.0], [1.0, 0]])
sY = np.array([[0, -1.0j], [1.0j, 0]])
sZ = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])

def cnc(N,h:float):
    model = 'ising'
    Nsites = N 
    usePBC = True
    numval = 1
    def doApplyHam(psiIn: np.ndarray, hloc: np.ndarray, N: int, usePBC: bool):
        d = hloc.shape[0]
        psiOut = np.zeros(psiIn.size)
        for k in range(N - 1):
            psiOut += np.tensordot(hloc.reshape(d**2, d**2), psiIn.reshape(d**k, d**2, d**(N - 2 - k)),axes=[[1], [1]]).transpose(1, 0, 2).reshape(d**N)
        if usePBC:
           psiOut += np.tensordot(hloc.reshape(d, d, d, d), psiIn.reshape(d, d**(N - 2), d), axes=[[2, 3], [2, 0]]).transpose(1, 2, 0).reshape(d**N)
        return psiOut
    if model == 'XX':
        hloc = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
    elif model == 'ising':
        hloc = (-np.kron(sX, sX) + h * np.kron(sZ, sI) +h * np.kron(sI, sZ)).reshape(2, 2, 2, 2)
    def doApplyHamClosed(psiIn):
        return doApplyHam(psiIn, hloc, Nsites, usePBC)
    
    H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)
    Energy, psi = eigsh(H, k=numval, which='SA')
    psi_r=psi.reshape(4,2**(Nsites-2))
    psi_c=np.conj(psi_r)
    rdm=Qobj(ncon([psi_r,psi_c],((-1,1),(-2,1))))
    rdm.dims=[[2,2],[2,2]]
    return concurrence(rdm)

h=np.linspace(0,2,100)
for N in tqdm(range(4,18,2)):
    tqdm.write("Calculating %s."%N)
    data=list(map(lambda h:cnc(N,h),h))
    plt.plot(h,data,label=r'$N=%s$'%N)
    plt.legend(fontsize=20)
    plt.tight_layout()
plt.show()


# import multiprocessing
# from functools import partial
# b=[]
# if __name__ == '__main__':
#     pool = multiprocessing.Pool(20)
#     h=np.linspace(0,2,100)
#     for N in tqdm(range(4,18,2)):
#         tqdm.write("Calculating %s."%N)
#         func = partial(cnc,N)
#         b=pool.map(func,h)
#         plt.plot(h,b,label=r'$N=%s$'%N)
#         plt.legend(fontsize=20)
#         plt.tight_layout()
#     plt.show()
