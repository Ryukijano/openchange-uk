
import numpy as np, math
rng=np.random.default_rng(0)
knots=17; t=np.linspace(0,3,knots); dt=3/(knots-1)
w=np.full(knots,2*dt); w[[0,-1]]=dt; phi=np.exp(-t**2/2); w=w*phi
def sigreg(z, slices=256, reps=20):
    N,D=z.shape; out=[]
    for _ in range(reps):
        A=rng.standard_normal((D,slices)); A/=np.linalg.norm(A,axis=0)
        x=(z@A)[...,None]*t                      # N,slices,knots
        err=(np.cos(x).mean(0)-phi)**2+np.sin(x).mean(0)**2
        out.append(((err@w)*N).mean())
    return float(np.mean(out))
N=2048
for D in [16,128,512]:
    rows=[f"D={D} gaussian {sigreg(rng.standard_normal((N,D))):.2f}"]
    for mu in [1,2,3,5]:
        s=np.sign(rng.standard_normal((N,1))); z=rng.standard_normal((N,D))
        z[:,0]=s[:,0]*mu + (math.sqrt(1-mu**2) if mu<1 else 0)*z[:,0]; z[:,0]/=z[:,0].std()
        rows.append(f"gap1d mu={mu}: {sigreg(z):.2f}")
    for k in [8,32,128]:
        if k>D: continue
        s=np.sign(rng.standard_normal((N,1))); z=rng.standard_normal((N,D))
        z[:,:k]=s*3.0+0.3*z[:,:k]; z[:,:k]/=z[:,:k].std(0)
        rows.append(f"gap in {k}/{D} dims: {sigreg(z):.2f}")
    print(" | ".join(rows))
# perfectly aligned slice on a pure two-point mixture
z=np.sign(rng.standard_normal((N,1))); x=z[...,None]*t
err=(np.cos(x).mean(0)-phi)**2+np.sin(x).mean(0)**2
print("aligned slice, Rademacher gap:", float((err@w)*N))

