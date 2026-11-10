"""Cuckoo Search for the established 4-D hyperchaotic Lorenz system.

System:
    x' = a(y-x) + w
    y' = cx-y-xz
    z' = xy-bz
    w' = -yz+rw

The optimizer is content/key-bound through the supplied initial state. It is
used only to select an admissible parameter set; the final Lyapunov spectrum
is recomputed independently with RK4 + Benettin/QR.
"""
import math
import numpy as np


class CuckooSearch:
    def __init__(self, initials, pop_size=6, pa=0.25, max_iter=1,
                 levy_beta=1.5, seed=None, quick_steps=500, dt=0.001):
        self.initials=np.asarray(initials,float)
        self.pop_size=int(pop_size); self.pa=float(pa); self.max_iter=int(max_iter)
        self.levy_beta=float(levy_beta); self.quick_steps=int(quick_steps); self.dt=float(dt)
        self.rng=np.random.default_rng(seed)
        # Established hyperchaotic Lorenz operating region around the standard
        # reference point a=10, b=8/3, c=28, r=-1.
        self.lb=np.array([9.0,2.5,27.0,-1.52])
        self.ub=np.array([11.0,3.0,29.0,-0.06])
        self.candidate_pool=[]

    @staticmethod
    def f(s,p):
        x,y,z,w=s; a,b,c,r=p
        return np.array([a*(y-x)+w,c*x-y-x*z,x*y-b*z,-y*z+r*w],float)

    @staticmethod
    def J(s,p):
        x,y,z,w=s; a,b,c,r=p
        return np.array([[-a,a,0,1],[c-z,-1,-x,0],[y,x,-b,0],[0,-z,-y,r]],float)

    @staticmethod
    def divergence(p):
        a,b,c,r=p; return -a-1-b+r

    def rk4(self,s,p,h=None):
        h=self.dt if h is None else h
        k1=self.f(s,p); k2=self.f(s+.5*h*k1,p); k3=self.f(s+.5*h*k2,p); k4=self.f(s+h*k3,p)
        return s+h*(k1+2*k2+2*k3+k4)/6

    def spectrum(self,p,steps=None,transient=1500):
        steps=self.quick_steps if steps is None else int(steps); h=self.dt; s=self.initials.copy()
        for _ in range(transient):
            s=self.rk4(s,p)
            if not np.all(np.isfinite(s)) or np.linalg.norm(s)>1e5: return np.full(4,-np.inf)
        V=np.eye(4); acc=np.zeros(4)
        for _ in range(steps):
            def rhs(x,Y): return self.f(x,p),self.J(x,p)@Y
            k1s,k1v=rhs(s,V); k2s,k2v=rhs(s+.5*h*k1s,V+.5*h*k1v)
            k3s,k3v=rhs(s+.5*h*k2s,V+.5*h*k2v); k4s,k4v=rhs(s+h*k3s,V+h*k3v)
            s=s+h*(k1s+2*k2s+2*k3s+k4s)/6; V=V+h*(k1v+2*k2v+2*k3v+k4v)/6
            if not np.all(np.isfinite(s)) or np.linalg.norm(s)>1e5: return np.full(4,-np.inf)
            Q,R=np.linalg.qr(V); d=np.abs(np.diag(R))
            if np.any(d<=0) or not np.all(np.isfinite(d)): return np.full(4,-np.inf)
            acc+=np.log(d); V=Q
        return acc/(steps*h)

    def _stream_stats(self,p,n=600):
        s=self.initials.copy(); vals=np.empty(n)
        for i in range(n):
            s=self.rk4(s,p); vals[i]=s[0]
        # Explicitly use signed integer byte indices for np.bincount.
        # This avoids uint64 -> int64 safe-casting errors on some NumPy versions.
        q=np.floor(np.abs(vals)*1e6).astype(np.int64)
        q=np.mod(q,256)
        counts=np.bincount(q,minlength=256).astype(np.float64)
        prob=counts[counts>0]/n
        ent=-np.sum(prob*np.log2(prob))
        ac=abs(np.corrcoef(q[:-1],q[1:])[0,1]) if n>2 else 1
        return float(ent),float(ac)

    def fitness(self,p):
        p=np.asarray(p,float); div=self.divergence(p)
        if div>=0: return -1e6
        le=self.spectrum(p)
        if not np.all(np.isfinite(le)): return -1e6
        pos=np.sum(le>1e-3)
        if pos<2: return -1e5+float(np.sum(le))
        ent,ac=self._stream_stats(p)
        return float(np.sort(le)[-2:].sum()+0.05*ent-0.5*ac)

    def levy(self,n):
        b=self.levy_beta
        sig=(math.gamma(1+b)*math.sin(math.pi*b/2)/(math.gamma((1+b)/2)*b*2**((b-1)/2)))**(1/b)
        return self.rng.normal(0,sig,n)/np.maximum(np.abs(self.rng.normal(0,1,n)),1e-12)**(1/b)

    def optimize(self,verbose=False):
        # Feasible-first search is preferable to allowing invalid candidates
        # to dominate a penalty-only population.
        nests=[]; vals=[]
        for _ in range(max(60,self.pop_size*5)):
            p=self.rng.uniform(self.lb,self.ub); f=self.fitness(p)
            if f>-1e5:
                nests.append(p); vals.append(f)
                if len(nests)>=self.pop_size: break
        if len(nests)<self.pop_size:
            raise RuntimeError('No hyperchaotic candidate found in the configured parameter region')
        nests=np.asarray(nests); vals=np.asarray(vals)
        span=self.ub-self.lb
        for g in range(self.max_iter):
            for i in range(self.pop_size):
                p=np.clip(nests[i]+0.01*self.levy(4)*span,self.lb,self.ub); f=self.fitness(p)
                if f>vals[i]: nests[i],vals[i]=p,f
            for i in np.argsort(vals)[:max(1,int(self.pa*self.pop_size))]:
                for _ in range(15):
                    p=self.rng.uniform(self.lb,self.ub); f=self.fitness(p)
                    if f>-1e5: nests[i],vals[i]=p,f; break
            if verbose: print('CS',g+1,np.max(vals))
        order=np.argsort(vals)[::-1]; self.candidate_pool=[nests[i].copy() for i in order]
        return nests[order[0]].copy(),float(vals[order[0]])
