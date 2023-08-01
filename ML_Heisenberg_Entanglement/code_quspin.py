from __future__ import print_function, division
import sys,os

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np


def Ham(J,L,delta,D):
    basis = spin_basis_1d(L,pauli=False)

    J_ij = [[J,i,i+1] for i in range(L-1)]
    J_zz = [[delta,i,i+1] for i in range(L-1)]
    h_z=[[D,i] for i in range(L)]

    static = [["xx",J_ij],["yy",J_ij],["zz",J_zz],["z",h_z]]
    dynamic=[]

    H= hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
    E,psi=H.eigsh(k=1,which='SA')
    return psi

delta=0.5
J=1
L=
D=0.1

Ham(J,L,delta,D)