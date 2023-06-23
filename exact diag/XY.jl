using LinearAlgebra, LinearMaps, Arpack, TensorOperations, QuantumInformation, Plots

include("doapplyham.jl");

N = 10
usePBC = true
numval = 1;

d = 2; # local dimension
sX = [0 1; 1 0]; sY = [0 -im; im 0]; sZ = [1 0; 0 -1]; sI = [1 0; 0 1];

function E(γ::Float64, λ::Float64)
    hloc = -reshape((λ/2)*real((1+γ)*kron(sX,sX) + (1-γ)*kron(sY,sY)) + kron(sZ,sI),2,2,2,2);
    doApplyHamClosed = LinearMap(psiIn -> doApplyHam(psiIn,hloc,N,usePBC), d^N;
    ismutating=false, issymmetric=true, ishermitian=true, isposdef=false)
  
    Energy, psi = eigs(doApplyHamClosed; nev = numval, tol = 1e-12, which=:SR, maxiter = 300);
  
    state=reshape(psi,d,d,d,d,d,d,d,d,d,d);
    r=ncon([state,state],[[-1,-2, 1,2,3,4,5,6,7,8],[-3,-4,1,2,3,4,5,6,7,8]],[false,true]);
    rho=reshape(r,d^2,d^2);
    cnc=max(0,abs(rho[2,3])-sqrt(rho[1,1]*rho[4,4]),abs(rho[1,4])-rho[2,2]);
    return cnc
end

<<<<<<< HEAD
γ=0.5
=======
γ=1.0
>>>>>>> 0ff8755ed599137741d754c437827fba315826e9
λ=range(0,2,100)
y=E.(γ,λ)

plot(λ,y)
