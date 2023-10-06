import numpy as np
from correlation_functions import *
from qutip import *
from toqito.state_props import l1_norm_coherence, negativity, log_negativity


def rho(s:str,beta:float, J:float, h0:float, Jz:float, h:float, Jp:float):
    if s=='1/2':
        d=2
        sz=sigmaz()
        sx=sigmax()
        sy=sigmay()
    else:
        d=3
        sx=jmat(1,'x')
        sy=jmat(1,'y')
        sz=jmat(1,'z')
    id=tensor(qeye(2),qeye(d),qeye(d),qeye(2))
    single_site= mag(s,beta,J,h0,Jz,h,Jp)*(tensor(sigmaz(),qeye(d),qeye(d),qeye(2))+tensor(qeye(2),qeye(d),qeye(d),sigmaz())) + Sz(s,beta,J,h0,Jz,h,Jp) * (tensor(qeye(2),sz,qeye(d),qeye(2))+
                                                                                                                                               tensor(qeye(2),qeye(d),sz,qeye(2)))
    two_site = (sigma_i_sigma_j(s,beta,1,J,h0,Jz,h,Jp) * tensor(sigmaz(),qeye(d),qeye(d),sigmaz()) + SzSz(s,beta,1,J,h0,Jz,h,Jp) * tensor(qeye(2),sz,sz,qeye(2)) + 
                SxSx(s,beta,J,h0,Jz,h,Jp) * (tensor(qeye(2),sx,sx,qeye(2)) + tensor(qeye(2),sy,sy,qeye(2)) )+
                Szsj(s,beta,1,J,h0,Jz,h,Jp)*(tensor(sigmaz(),sz,qeye(d),qeye(2))+tensor(qeye(2),qeye(d),sz,sigmaz())+
                                              tensor(sigmaz(),qeye(d),sz,qeye(2))+tensor(qeye(2),sz,qeye(d),sigmaz())))
    three_site = mag(s,beta,J,h0,Jz,h,Jp)*Szsj(s,beta,1,J,h0,Jz,h,Jp)*(tensor(sigmaz(),sz,qeye(d),sigmaz()) + 
                                                                         tensor(sigmaz(),qeye(d),sz,sigmaz()) + tensor (qeye(2),sz,sz,sigmaz()) + 
                                                                         tensor (sigmaz(),sz,sz,qeye(2))) 
    four_site = sigma_i_sigma_j(s,beta,1,J,h0,Jz,h,Jp) * SzSz(s,beta,1,J,h0,Jz,h,Jp) * tensor(sigmaz(),sz,sz,sigmaz()) + sigma_i_sigma_j(s,beta,1,J,h0,Jz,h,Jp) * SxSx(s,beta,J,h0,Jz,h,Jp) * (
        tensor(sigmaz(),sx,sx,sigmaz()) + tensor(sigmaz(),sy,sy,sigmaz())
    )
    state=  ((1/(2*d*2*d)) * (id + single_site + two_site + three_site + four_site)).ptrace([1,2])
    return concurrence(state)

# s='1/2'; beta=np.float128(200); h0=h=1e-4; J=1; Jz=0.5; Jp=2

# print(rho(s,beta,J,h0,Jz,h,Jp))


