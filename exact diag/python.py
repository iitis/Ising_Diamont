import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer
from qutip import *
from ncon import ncon

def doApplyHam(psiIn: np.ndarray, hloc: np.ndarray, N: int, usePBC: bool):
    d = hloc.shape[0]
    psiOut = np.zeros(psiIn.size)
    for k in range(N - 1):
         psiOut += np.tensordot(hloc.reshape(d**2, d**2),
                                psiIn.reshape(d**k, d**2, d**(N - 2 - k)),
                                axes=[[1], [1]]).transpose(1, 0, 2).reshape(d**N)
    if usePBC:
        psiOut += np.tensordot(hloc.reshape(d, d, d, d),psiIn.reshape(d, d**(N - 2), d),
                             axes=[[2, 3], [2, 0]]).transpose(1, 2, 0).reshape(d**N)
    return psiOut

model = 'ising'  
Nsites = 10
usePBC = True
numval = 1

d = 2  # local dimension
sX = np.array([[0, 1.0], [1.0, 0]])
sY = np.array([[0, -1.0j], [1.0j, 0]])
sZ = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])

if model == 'XX':
    hloc = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
elif model == 'ising':
    hloc = (-np.kron(sX, sX) + np.kron(sZ, sI) + np.kron(sI, sZ)).reshape(2, 2, 2, 2)

def doApplyHamClosed(psiIn):
    return doApplyHam(psiIn, hloc, Nsites, usePBC)


H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)

start_time = timer()
Energy, psi = eigsh(H, k=numval, which='SA')

psi_r=psi.reshape(4,2**(Nsites-2))
psi_c=np.conj(psi_r)
rdm=Qobj(ncon([psi_r,psi_c],((-1,1),(-2,1))))
rdm.dims=[[2,2],[2,2]]
diag_time = timer() - start_time

print('N=%d, Time=%f, concurrence=%f ' %(Nsites, diag_time, concurrence(rdm)))

# rho=ket2dm(Qobj(psi))
# rho.dims=[[2]*Nsites,[2]*Nsites]
# print(concurrence(rho.ptrace([0,1])))

