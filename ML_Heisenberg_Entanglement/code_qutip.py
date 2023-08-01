import numpy as np
from qutip import *

# 4 sites Hamiltonian using Qutip
def ham(D,J,delta):
    h1 = D*(tensor(sigmaz()**2,qeye(2),qeye(2),qeye(2))+tensor(qeye(2),sigmaz()**2,qeye(2),qeye(2))+tensor(qeye(2),qeye(2),sigmaz()**2,qeye(2))+tensor(qeye(2),qeye(2),qeye(2),sigmaz()**2))
    h2 = J*(tensor(sigmax(),sigmax(),qeye(2),qeye(2))+tensor(sigmay(),sigmay(),qeye(2),qeye(2))+delta*tensor(sigmaz(),sigmaz(),qeye(2),qeye(2)))
    h3 = (J/(2**4)) * (tensor(sigmax(),sigmax(),sigmax(),qeye(2))+tensor(sigmay(),sigmay(),sigmay(),qeye(2))+delta*tensor(sigmaz(),sigmaz(),sigmaz(),qeye(2)))
    h4 = (J/(3**4)) * (tensor(sigmax(),sigmax(),sigmax(),sigmax())+tensor(sigmay(),sigmay(),sigmay(),sigmay())+delta*tensor(sigmaz(),sigmaz(),sigmaz(),sigmaz()))
    return h1+h2+h3+h4


def rho(D,J,delta):
    e,v=ham(D,J,delta).groundstate()
    r=ket2dm(v)
    return entropy_vn(r.ptrace([0,1]))

print(ham(0.1,1,0.5).groundstate())