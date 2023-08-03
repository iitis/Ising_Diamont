from hamiltonian import *

def state(h,s,N):
    psi_r=Hamiltonian(h,s,N).reshape(d0*d1,d0*d**(N-1))
    psi_c=np.conj(psi_r)
    #rdm=ncon([psi_r,psi_c],((-1,1),(-2,1)))
    rdm=Qobj(ncon([psi_r,psi_c],((-1,1),(-2,1))))
    rdm.dims=[[2,3],[2,3]]
    return rdm #log_negativity(rdm,dim=[2,d1])


def wigner(t1,t2,f1,f2,h,s,N):
    u1=(1j*sigmaz()*f1).expm() * (1j*sigmay()*t1).expm()
    Pi_1=qeye(2)-np.sqrt(3)*sigmaz()
    delta_1=(1/2) * (u1*Pi_1*(u1.dag()))

    u2=(1j*jmat(s,'z')*f2).expm() * (1j*jmat(s,'y')*t2).expm()
    eta=np.sqrt(2*s*(2*s+1)*(2*s+2)/2)
    d=2*s+1
    Pi_2=np.zeros((d,d))
    for i in range(d-1):
        Pi_2[i,i]=1-eta*np.sqrt(2/(d*(d-1)))
    Pi_2[d-1,d-1]=1+eta*np.sqrt(2*(d-1)/d)
    delta_2=(1/d) * (u2*Qobj(Pi_2)*(u2.dag()))

    delta = tensor(delta_1,delta_2)
    return np.real((state(h,s,N)*delta).tr())



#print(wigner(t1,t2,f1,f2,4,s,N))


# d=[-1,2,1,-5,-3]
# neg=[x for x in d if x<0]
# print(-sum(neg))
#print(2*jmat(1/2,'x'))
