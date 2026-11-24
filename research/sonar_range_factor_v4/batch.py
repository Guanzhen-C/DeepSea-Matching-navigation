"""Vectorized M1 height profiles for one node pair, isolated from navigation."""
from dataclasses import dataclass

import numpy as np

from research.sonar_range_factor_v2.models import _profiled
from research.sonar_range_factor_v3.likelihood import RadialLoss


@dataclass(frozen=True)
class Result:
    height: np.ndarray
    cost: np.ndarray
    status: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray
    hessian: np.ndarray
    curvature: np.ndarray
    gradient_z: np.ndarray
    work: np.ndarray
    reference_rows: int
    candidate_count: np.ndarray


def _ib_mul(lo,hi,low,high):
    a,b,c,d=lo*low,lo*high,hi*low,hi*high
    return np.minimum(np.minimum(a,b),np.minimum(c,d)),np.maximum(np.maximum(a,b),np.maximum(c,d))


def _state(d,z,r,h):
    delta=h[:,None]-z
    rho=np.sqrt(d+delta*delta)
    return rho,rho-r,delta/rho,d/(rho**3)


def _point(d,z,r,h,W):
    rho,q,b,c=_state(d,z,r,h)
    wq=np.einsum("nij,nj->ni",W,q)
    g=np.sum(wq*b,axis=1)
    H=np.sum(np.einsum("nij,nj->ni",W,b)*b+wq*c,axis=1)
    return g,H


def _boxes(d,z,r,W,l,h):
    """Same algebraic enclosures as the v2 floating reference, evaluated by rows."""
    rl,ql,bl,kl=_state(d,z,r,l)
    rh,qh,bh,kh=_state(d,z,r,h)
    g_lo=np.zeros_like(l)
    g_hi=g_lo.copy()
    h_lo=np.sum(W.diagonal(axis1=1,axis2=2)*(1-r*d/rh**3),axis=1)
    h_hi=np.sum(W.diagonal(axis1=1,axis2=2)*(1-r*d/rl**3),axis=1)
    # Magnitude before cancellation supplies (conservative, non-certified)
    # floating-point guard. A formal certificate remains the v3 Decimal path.
    span=np.sum(abs(W.diagonal(axis1=1,axis2=2))*(1+abs(r*d/rh**3)),axis=1)
    for i in range(2):
        for j in range(2):
            def add(x0,x1,lo,hi):
                a,b=_ib_mul(x0,x1,lo,hi)
                a,b=a*W[:,i,j],b*W[:,i,j]
                return np.minimum(a,b),np.maximum(a,b),np.maximum(abs(a),abs(b))
            mn,mx,s=add(qh[:,i],ql[:,i],bl[:,j],bh[:,j])
            g_lo+=mn
            g_hi+=mx
            span+=s
            if i!=j:
                for term in ((bl[:,i],bh[:,i],bl[:,j],bh[:,j]),
                             (kl[:,i],kh[:,i],qh[:,j],ql[:,j])):
                    mn,mx,s=add(*term)
                    h_lo+=mn
                    h_hi+=mx
                    span+=s
    pad=2e-12*(1+span+abs(g_lo)+abs(g_hi)+abs(h_lo)+abs(h_hi))
    return g_lo-pad,g_hi+pad,h_lo-pad,h_hi+pad


def _bracket_root(d,z,r,W,l,h,warm):
    """Safeguarded Newton: a warm start may change cost, never the search space."""
    x=np.where((warm>l)&(warm<h)&np.isfinite(warm),warm,(l+h)*.5)
    active=np.ones(len(l),bool)
    for _ in range(55):
        if not active.any():
            break
        g,H=_point(d,z,r,x,W)
        near=(abs(g)<=4e-13)&(abs(g)<=np.maximum(H,1e-12)*2e-11)
        active &= ~near&(h-l>4e-12)
        l=np.where(active&(g<0),x,l)
        h=np.where(active&(g>=0),x,h)
        with np.errstate(divide="ignore",invalid="ignore"):
            nxt=x-g/H
        good=np.isfinite(nxt)&(nxt>l)&(nxt<h)
        x=np.where(active,np.where(good,nxt,(l+h)*.5),x)
    return x,active


def _bundle(C, e, A, loss):
    if loss.kind=="gaussian":
        return e,A
    if loss.kind!="huber":
        raise ValueError("Batch workflow supports Gaussian and phase-5 radial Huber")
    t=np.einsum("ni,ni->n",e,e)
    mask=t>loss.scale**2
    radius=np.sqrt(t[mask])
    scale=np.sqrt(2*loss.scale*(radius-.5*loss.scale)/t[mask])
    radial=e[mask]/radius[:,None]
    along=np.einsum("ni,nij->nj",radial,A[mask])
    A[mask]=scale[:,None,None]*A[mask]+(loss.scale/(radius*scale)-scale)[:,None,None]*radial[:,:,None]*along[:,None,:]
    e[mask]*=scale[:,None]
    return e,A


class Batch:
    """Caches measurement-side matrices and the last exact pose evaluation."""
    def __init__(self,tracks,loss=RadialLoss()):
        self.tracks=tuple(tracks)
        if not self.tracks or any(len(t.ranges)!=2 for t in self.tracks):
            raise ValueError("Nonempty set of dual-view tracks required")
        self.loss=loss
        self.L=np.stack([t.xy for t in self.tracks])
        self.u=np.stack([t.lever for t in self.tracks])
        self.z=np.stack([t.sensor_z for t in self.tracks])
        self.r=np.stack([t.ranges for t in self.tracks])
        self.C=np.stack([t.whitener for t in self.tracks])
        self.W=np.stack([t.precision for t in self.tracks])
        self.lo,self.hi=np.array([t.bounds for t in self.tracks]).T
        self.last=None
        self.warm=np.full(len(self.tracks),np.nan)
        self.cache_hits=0

    def geom(self,poses):
        c,s=np.cos(poses[:,2]),np.sin(poses[:,2])
        R=np.array([[c,-s],[s,c]]).transpose(2,0,1)
        u=np.einsum("ijk,nik->nij",R,self.u)
        v=self.L[:,None,:]-(poses[None,:,:2]+u)
        return v,u,np.einsum("nis,nis->ni",v,v)

    def evaluate(self,poses,*,max_boxes=128,allow_reference=True,use_cache=True):
        poses=np.asarray(poses,float)
        if poses.shape!=(2,3) or not np.isfinite(poses).all() or max_boxes<0:
            raise ValueError("Two finite world Pose2 and nonnegative work budget required")
        policy=(max_boxes,allow_reference)
        if use_cache and self.last is not None and policy==self.last[0] and np.array_equal(poses,self.last[1]):
            self.cache_hits+=1
            return self.last[2]
        v,u,d=self.geom(poses)
        n=len(d)
        status=np.full(n,"ok",dtype="<U80")
        singular=(d+(self.hi[:,None]-self.z)**2<=1e-20).any(axis=1)
        status[singular]="nadir"
        needs=np.zeros(n,bool)
        work=np.zeros(n,int)
        candidates=[(np.arange(n),self.lo.copy()),(np.arange(n),self.hi.copy())]
        q=lambda i:(d[i],self.z[i],self.r[i],self.W[i])
        ids=np.flatnonzero(~singular&(self.lo<self.hi))
        l,h=self.lo[ids],self.hi[ids]
        while len(ids):
            np.add.at(work,ids,1)
            overflow=work[ids]>max_boxes
            needs[ids[overflow]]=True
            valid=~needs[ids]
            ids,l,h=ids[valid],l[valid],h[valid]
            if not len(ids):
                break
            a,z,r,W=q(ids)
            gl,gh,Hl,Hh=_boxes(a,z,r,W,l,h)
            # Concave interiors cannot minimize; endpoints of non-flat domain
            # are kept, and distinct/nearly tied candidates are not discarded.
            suspect=(gl<=0)&(gh>=0)&(Hh>=0)
            convex=suspect&(Hl>0)
            if convex.any():
                ii,ll,hh=ids[convex],l[convex],h[convex]
                args=q(ii)
                g0,_=_point(*args[:3],ll,args[3])
                g1,_=_point(*args[:3],hh,args[3])
                root=(g0<=0)&(g1>=0)
                if root.any():
                    kk,l0,h0=ii[root],ll[root],hh[root]
                    args=q(kk)
                    x,bad=_bracket_root(*args,l0,h0,self.warm[kk] if use_cache else np.full(len(kk),np.nan))
                    x=np.where(g0[root]==0,l0,np.where(g1[root]==0,h0,x))
                    needs[kk[bad]]=True
                    candidates.append((kk,x))
                # Almost-zero signs at a split boundary are ambiguous.
                uncertain=(~root)&((abs(g0)<1e-11)|(abs(g1)<1e-11))
                ii0=ii[uncertain]
                # Exact zero of the structural ceiling is resolved by the final chart check.
                structural=np.all(self.z[ii0]==self.hi[ii0,None],axis=1)&(hh[uncertain]==self.hi[ii0])
                needs[ii0[~structural]]=True
            split=suspect&~convex
            ids,l,h=ids[split],l[split],h[split]
            if not len(ids):
                break
            tiny=(h-l<=1e-7)
            needs[ids[tiny]]=True
            ids,l,h=ids[~tiny],l[~tiny],h[~tiny]
            mid=(l+h)*.5
            ids,l,h=np.r_[ids,ids],np.r_[l,mid],np.r_[mid,h]
        idx=np.concatenate([x[0] for x in candidates])
        heights=np.concatenate([x[1] for x in candidates])
        count=np.bincount(idx,minlength=n)
        delta=heights[:,None]-self.z[idx]
        rr=np.sqrt(d[idx]+delta**2)-self.r[idx]
        er=np.einsum("nij,nj->ni",self.C[idx],rr)
        costs=np.sum(er**2,axis=-1)*.5
        costs[~np.isfinite(costs)]=np.inf
        order=np.lexsort((heights,costs,idx))
        head=order[np.r_[True,idx[order][1:]!=idx[order][:-1]]]
        h,fc=heights[head],costs[head]
        ties=(abs(heights-h[idx])>8e-7)&(costs<=fc[idx]+1e-11*(1+fc[idx]))
        np.logical_or.at(needs,idx,ties)
        status[needs&~singular]="uncertain_or_budget"
        return self._linearize(poses,policy,v,u,d,h,fc,status,work,allow_reference,use_cache,count)

    def _linearize(self,poses,policy,v,u,d,h,fc,status,work,allow_reference,use_cache,count):
        n=len(h)
        with np.errstate(divide="ignore",invalid="ignore"):
            rho,q,b,c=_state(d,self.z,self.r,h)
            G=np.zeros((n,2,2,3))
            G[:,:,0,0]=G[:,:,1,1]=1
            G[:,:,:,2]=np.stack((-u[:,:,1],u[:,:,0]),axis=-1)
            ap=-np.einsum("nix,nixj->nij",v,G)/rho[:,:,None]
            A=np.zeros((n,2,6))
            A[:,0,:3],A[:,1,3:]=ap[:,0],ap[:,1]
            e=np.einsum("nij,nj->ni",self.C,q)
            J0=self.C@A
            B=np.einsum("nij,nj->ni",self.C,b)
            Wq=np.einsum("nij,nj->ni",self.W,q)
            gz=np.sum(e*B,axis=1)
            Hzz=np.sum(B*B+Wq*c,axis=1)
            pp=J0.transpose(0,2,1)@J0
            term=(G.transpose(0,1,3,2)@G-ap[:,:,:,None]*ap[:,:,None,:])/rho[:,:,None,None]
            term[:,:,2,2]+=np.sum(v*u,axis=2)/rho
            pp[:,:3,:3]+=Wq[:,0,None,None]*term[:,0]
            pp[:,3:,3:]+=Wq[:,1,None,None]*term[:,1]
            zp=np.einsum("ni,nij->nj",B,J0)
            iz=-ap*b[:,:,None]/rho[:,:,None]*Wq[:,:,None]
            zp[:,:3]+=iz[:,0]
            zp[:,3:]+=iz[:,1]
            interior=(h>self.lo)&(h<self.hi)
            structural=(h==self.hi)&np.all(self.z==self.hi[:,None],axis=1)&(Hzz>1e-10)
            bad=(interior&(Hzz<=1e-10))|(~interior&~structural&(self.lo<self.hi)&(abs(gz)<=1e-9))
            status[(status=="ok")&bad]="active_switch_or_curvature"
            dz=np.where(interior[:,None],-zp/Hzz[:,None],0.)
            J=J0+B[:,:,None]*dz[:,None,:]
            Hess=pp+zp[:,:,None]*dz[:,None,:]
        fallback=0
        if allow_reference:
            for i in np.flatnonzero((status!="ok")&(status!="nadir")):
                fallback+=1
                try:
                    opt,r,g,H,zdp=_profiled(self.tracks[i],(v[i],u[i],d[i]))
                except (ValueError,RuntimeError) as exc:
                    message=str(exc)
                    status[i]=("ties_or_unresolved" if "Competing" in message else
                               "active_switch_or_curvature" if any(x in message for x in ("Active","curvature")) else
                               "uncertain_or_budget")
                    continue
                status[i]="ok_reference"
                h[i],fc[i],e[i]=opt.height,opt.cost,r
                curvature=H[-1,-1]
                Hzz[i],gz[i]=curvature,g[:,-1]@r
                J[i]=g[:,:-1]+np.outer(g[:,-1],zdp)
                Hess[i]=H[:-1,:-1]+np.outer(H[:-1,-1],zdp)
        ok=np.char.startswith(status,"ok")
        nonfinite=~(np.isfinite(J).all(axis=(1,2))&np.isfinite(Hess).all(axis=(1,2))&np.isfinite(e).all(axis=1))
        status[ok&nonfinite]="nonfinite"
        ok &= ~nonfinite
        e[~ok],J[~ok],Hess[~ok]=0.,0.,np.nan
        e,J=_bundle(self.C,e,J,self.loss)
        self.warm=np.where(ok,h,self.warm)
        result=Result(h,fc,status,e,J,Hess,Hzz,gz,work,fallback,count)
        for arr in vars(result).values():
            if isinstance(arr,np.ndarray):
                arr.setflags(write=False)
        self.last=(policy,poses.copy(),result) if use_cache else None
        return result
