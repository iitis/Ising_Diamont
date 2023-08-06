import numpy as np
from correlation_functions import *
from qutip import *
from toqito.state_props import l1_norm_coherence, negativity, log_negativity


def rho(s,beta,J,h0,Jz,h,Jm,Jp):
    d=2
    id=tensor(qeye(2),qeye(d),qeye(d),qeye(2))
    single_site= mag(beta,J,h0,Jz,h,Jm,Jp)*(tensor(sigmaz(),qeye(d),qeye(d),qeye(2))+tensor(qeye(2),qeye(d),qeye(d),sigmaz())) + Sz(s,beta,J,h0,Jz,h,Jm,Jp) * (tensor(qeye(2),jmat(1/2,'z'),qeye(d),qeye(2))+tensor(qeye(2),qeye(d),jmat(1/2,'z'),qeye(2)))
    
    two_site = (sigma_i_sigma_j(beta,1,J,h0,Jz,h,Jm,Jp) * tensor(sigmaz(),qeye(d),qeye(d),sigmaz()) + SzSz(beta,J,h0,Jz,h,Jm,Jp) * tensor(qeye(2),jmat(1/2,'z'),jmat(1/2,'z'),qeye(2)) + 
                2 * SxSx(beta,J,h0,Jz,h,Jp) * tensor(qeye(2),jmat(1/2,'x'),jmat(1/2,'x'),qeye(2)))
    
    three_site = mag(beta,J,h0,Jz,h,Jm,Jp)*Shsi(s,beta,J,h0,Jz,h,Jm,Jp)*(tensor(sigmaz(),jmat(1/2,'z'),qeye(d),sigmaz()) + 
                                                                         tensor(sigmaz(),qeye(d),jmat(1/2,'z'),sigmaz())) 
    
    four_site = sigma_i_sigma_j(beta,1,J,h0,Jz,h,Jm,Jp) * SzSz(beta,J,h0,Jz,h,Jm,Jp) * tensor(sigmaz(),jmat(1/2,'z'),jmat(1/2,'z'),sigmaz())

    state=  (1/(2*d*2*d)) * (id + single_site + two_site + three_site + four_site)
    return log_negativity(np.array(state),[2,2,2,2])

s='1/2'; beta=1; h0=h=0
#print(rho(s,beta,1,h0,1,h,2,1))


