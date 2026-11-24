"""Robust losses on a joint whitened range block, with exact smooth-chart Jacobians."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import chi2

from research.sonar_range_factor_v2.models import geometry, profiled_residual_jacobian


@dataclass(frozen=True)
class RadialLoss:
    kind: str="gaussian"
    scale: float=1.
    mixture_prior: float=.05
    log_gaussian_to_outlier_at_zero: float=2.

    def value_slope(self,t,dim):
        """Return phi(t), phi'(t) for t=||Cq||² with actual Gaussian moment covariance."""
        if not math.isfinite(t) or t<0 or dim<1 or not math.isfinite(self.scale) or self.scale<=0:
            raise ValueError("Invalid radial input")
        k=self.scale
        if self.kind=="gaussian":
            return .5*t,.5
        if self.kind=="huber":
            return (.5*t,.5) if t<=k*k else (k*(math.sqrt(t)-.5*k),k/(2*math.sqrt(t)))
        if self.kind=="cauchy_penalty":
            return .5*k*k*math.log1p(t/(k*k)),.5/(1+t/(k*k))
        if self.kind=="student":
            if k<=2:
                raise ValueError("Student-t requires nu>2 to match moment covariance")
            return .5*(k+dim)*math.log1p(t/(k-2)),.5*(k+dim)/(k-2+t)
        if self.kind=="mixture":
            # Constant density for the misassociation component on the fixed
            # observation support; mixture weight includes density ratio at zero.
            if not 0<self.mixture_prior<1 or not math.isfinite(self.log_gaussian_to_outlier_at_zero):
                raise ValueError("Invalid mixture weight or density ratio")
            lp=(math.log1p(-self.mixture_prior)-math.log(self.mixture_prior)
                +self.log_gaussian_to_outlier_at_zero)
            log_weight=-float(np.logaddexp(0.,-lp))
            w0=math.exp(log_weight)
            depletion=w0*(-math.expm1(-.5*t))
            # Keep tiny costs rather than subtracting nearly equal log densities.
            val=(-math.log1p(-depletion) if depletion<.95 else
                 float(np.logaddexp(0.,lp)-np.logaddexp(0.,lp-.5*t)))
            slope=.5*math.exp(-float(np.logaddexp(0.,.5*t-lp)))
            return val,slope
        raise ValueError("Unknown radial loss")


def matched_huber(dim):
    """Same nominal central Gaussian mass as a scalar 3-sigma cutoff."""
    return RadialLoss("huber",float(math.sqrt(chi2.ppf(math.erf(3./math.sqrt(2.)),dim))))


def transform(e,J,loss):
    e,J=np.asarray(e,float),np.asarray(J,float)
    t=float(e@e)
    phi,slope=loss.value_slope(t,len(e))
    if t==0:
        a=math.sqrt(2*slope)
        return e*a,J*a
    a=math.sqrt(2*phi/t)
    if a==0:
        return e*a,J*a
    # Apply the radial direction as a rank-one update (stable at the origin).
    u=e/math.sqrt(t)
    alpha=(2*slope/a)-a
    return a*e,a*J+alpha*np.outer(u,u@J)


def profile_pair(track,pose,loss):
    if len(track.ranges)!=2:
        raise ValueError("M1 requires one pair")
    e,J,opt=profiled_residual_jacobian(track,pose)
    v,A=transform(e,J,loss)
    return v,A,opt


def nuisance_design(track,poses,height,*,sound_speed=1480.):
    """Range residual perturbations [map XY, all origin XY, z_i, dc/c, dt0_s]."""
    if not math.isfinite(sound_speed) or sound_speed<=0 or not track.bounds[0]<=height<=track.bounds[1]:
        raise ValueError("Invalid nominal calibration geometry")
    v,_,d2=geometry(track,poses)
    dz=height-track.sensor_z
    rho=np.sqrt(d2+dz**2)
    if np.any(rho<=1e-10):
        raise ValueError("Nadir/zero range propagation")
    n=len(rho)
    G=np.zeros((n,2+3*n+2))
    xy=v/rho[:,None]
    G[:,:2]=xy
    idx=np.arange(n)
    G[idx,2+2*idx]=-xy[:,0]
    G[idx,3+2*idx]=-xy[:,1]
    G[idx,2+2*n+idx]=-dz/rho
    G[:,-2]=-rho
    G[:,-1]=-sound_speed/2.
    return G


def propagate(track,poses,height,measurement_cov,nuisance_cov):
    G=nuisance_design(track,poses,height)
    n,m=G.shape
    R,B=np.asarray(measurement_cov,float),np.asarray(nuisance_cov,float)
    for S,size in ((R,n),(B,m)):
        if S.shape!=(size,size) or not np.isfinite(S).all() or not np.allclose(S,S.T):
            raise ValueError("Expected finite symmetric covariance")
        if np.linalg.eigvalsh(S).min()<-1e-12*max(1.,np.abs(S).max()):
            raise ValueError("Nuisance and receiver covariance must be PSD")
    result=R+G@B@G.T
    result=(result+result.T)*.5
    np.linalg.cholesky(result)
    return result
