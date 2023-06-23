using ITensors, Plots

function ent(N::Int,h::Float64)
  sites = siteinds("S=1/2",N)
  os = OpSum()
  for j=1:N-1
    os += h,"Sz",j,"Sz",j+1
    os += 1,"Sx",j,"Sx",j+1
    os += 1,"Sy",j,"Sy",j+1
  end
  H = MPO(os,sites)

  psi0 = randomMPS(sites,10)
  nsweeps = 10
  outputlevel = 0
  maxdim = [10,20,100,100,200]
  cutoff = [1E-10]

  energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff, outputlevel)
  b=Int(N//2)
  orthogonalize!(psi,b)
  s = siteinds(psi)
  U,S,V=svd(psi[b], (linkind(psi, b-1), s[b]))
  SvN = 0.0
  for n in 1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(2,p)
  end
  return SvN/N
end

h=range(-2,2,100)
N=100
data=ent.(N,h)
plot(h,data)
savefig("ent.pdf")