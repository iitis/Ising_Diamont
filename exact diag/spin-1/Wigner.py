from hamiltonian import *

def QI(h,s,N):
    psi_r=Hamiltonian(h,s,N).reshape(d0*d1,d0*d**(N-1))
    psi_c=np.conj(psi_r)
    rdm=ncon([psi_r,psi_c],((-1,1),(-2,1)))
    #rdm=Qobj(ncon([psi_r,psi_c],((-1,1),(-2,1))))
    #rdm.dims=[[2,d1],[2,d1]]
    return log_negativity(rdm,dim=[2,d1])


def wigner(t1,t2,f1,f2,h,s,N):
    u1=np.exp(1j*sigmaz()*f1)*np.exp(1j*sigmay()*t1)
    
    u2=np.exp(1j*jmat(s,'z')*f2)*np.exp(1j*jmat(s,'y')*t2)
    eta=np.sqrt(2*s*(2*s+1)*(2*s+2)/2)
    Pi=qeye(2*s+1) - eta * A
    return Pi

d=[-1,2,1,-5,-3]
neg=[x for x in d if x<0]
print(-sum(neg))
#print(2*jmat(1/2,'x'))