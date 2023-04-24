using Printf, LinearAlgebra, LinearMaps, Arpack, TensorOperations, QuantumInformation, Plots, Latexify

include("doapplyham.jl");

N = 10
usePBC = true
numval = 1;

d = 2; # local dimension
sX = [0 1; 1 0]; sY = [0 -im; im 0]; sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

function E(Δ::Float64)
    hloc = (1/4)*reshape(kron(sX,sX) + kron(sY,sY) + Δ*kron(sZ,sZ),2,2,2,2);
    doApplyHamClosed = LinearMap(psiIn -> doApplyHam(psiIn,hloc,N,usePBC), d^N;
    ismutating=false, issymmetric=true, ishermitian=true, isposdef=false)
  
    Energy, psi = eigs(doApplyHamClosed; nev = numval, tol = 1e-12, which=:SR, maxiter = 300);
  
    state=reshape(psi,d,d,d,d,d,d,d,d,d,d);
    r=ncon([state,state],[[-1,-2, 1,2,3,4,5,6,7,8],[-3,-4,1,2,3,4,5,6,7,8]],[false,true]);
    rho=reshape(r,d^2,d^2);
    #cnc=max(0,abs(rho[2,3])-sqrt(rho[1,1]*rho[4,4]),abs(rho[1,4])-sqrt())
    return negativity(rho,[2,2],2)
end

Δ=range(-2,2,100)
y=E.(Δ)

plot(Δ,y)
