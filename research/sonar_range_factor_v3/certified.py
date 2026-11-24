"""Outward-rounded certificate for a frozen two-range problem, not a pose certificate."""
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import heapq
import math
import time

from research.sonar_range_factor_v2.models import geometry


def binary(x):
    return Decimal.from_float(float(x))


@dataclass(frozen=True)
class I:
    lo: Decimal
    hi: Decimal


class Bounds:
    """Use explicit contexts; Decimal.sqrt is correctly rounded to nearest."""
    def __init__(self, digits):
        if not 18 <= digits <= 120:
            raise ValueError("Use 18 to 120 decimal significant digits")
        self.down = Context(prec=digits, rounding=ROUND_FLOOR)
        self.up = Context(prec=digits, rounding=ROUND_CEILING)
        self.near = Context(prec=digits, rounding=ROUND_HALF_EVEN)
        self.zero = Decimal(0)
        self.one = Decimal(1)

    def point(self, x):
        return I(x, x)

    def add(self, a, b):
        return I(self.down.add(a.lo,b.lo), self.up.add(a.hi,b.hi))

    def sub(self, a, b):
        return I(self.down.subtract(a.lo,b.hi), self.up.subtract(a.hi,b.lo))

    def scale(self, a, b):
        return self.mul(a, self.point(b))

    def mul(self, a, b):
        endpoints = [(x,y) for x in (a.lo,a.hi) for y in (b.lo,b.hi)]
        return I(min(self.down.multiply(x,y) for x,y in endpoints),
                 max(self.up.multiply(x,y) for x,y in endpoints))

    def sq(self, a):
        low = self.zero if a.lo <= 0 <= a.hi else min(a.lo.copy_abs(),a.hi.copy_abs())
        high = max(a.lo.copy_abs(),a.hi.copy_abs())
        return I(self.down.multiply(low,low), self.up.multiply(high,high))

    def div(self, a, b):
        if b.lo <= 0:
            raise ZeroDivisionError("No finite interval derivative at a zero range")
        return self.mul(a, I(self.down.divide(self.one,b.hi), self.up.divide(self.one,b.lo)))

    def sqrt(self, a):
        if a.lo < 0:
            raise ValueError("Negative squared distance enclosure")
        ll, hh = self.near.sqrt(a.lo), self.near.sqrt(a.hi)
        return I(max(self.zero,self.near.next_minus(ll)) if ll else ll,
                 self.near.next_plus(hh) if hh else hh)

    def sum(self, intervals):
        value = self.point(self.zero)
        for x in intervals:
            value = self.add(value,x)
        return value

    def midpoint(self, a, b):
        return self.near.multiply(self.near.add(a,b), Decimal("0.5"))


class FrozenHeight:
    """The exact real objective over binary64 d², z, measured r and binary64 C.

    R(psi), XY offsets and the covariance Cholesky are upstream floating point;
    their rounding is not enclosed here. C is applied, not a rounded C.T @ C.
    """
    def __init__(self, track, poses, digits=36):
        if len(track.ranges)!=2:
            raise ValueError("Strict certifier currently handles exactly two views")
        d2 = geometry(track,poses)[2]
        if not all(math.isfinite(x) and x>=0 for x in d2):
            raise ValueError("The represented squared distances must be finite")
        self.a = Bounds(digits)
        self.d2, self.z, self.r = [tuple(map(binary, x))
                                  for x in (d2, track.sensor_z, track.ranges)]
        self.C = tuple(tuple(map(binary,x)) for x in track.whitener)
        self.lo, self.hi = map(binary, track.bounds)

    def apply(self,x):
        a = self.a
        return [a.sum(a.scale(y,c) for y,c in zip(x,row)) for row in self.C]

    def terms(self,l,h,derivatives):
        a = self.a
        residual, first, second = [], [], []
        for d,z,r in zip(self.d2,self.z,self.r):
            t = a.sub(I(l,h),a.point(z))
            rho = a.sqrt(a.add(a.point(d),a.sq(t)))
            residual.append(a.sub(rho,a.point(r)))
            if derivatives:
                first.append(a.div(t,rho))
                second.append(a.div(a.point(d),a.mul(a.sq(rho),rho)))
        e = self.apply(residual)
        f = a.scale(a.sum(a.sq(x) for x in e), Decimal("0.5"))
        if derivatives:
            b,c = self.apply(first),self.apply(second)
            g = a.sum(a.mul(x,y) for x,y in zip(e,b))
            H = a.sum([a.sq(x) for x in b]+[a.mul(x,y) for x,y in zip(e,c)])
            return f,g,H
        return f,None,None

    def minimum(self, *, gap="1e-9", max_boxes=4096, timeout_s=2.):
        """Certified objective bracket, even if interrupted; no uniqueness claim."""
        a = self.a
        target = Decimal(gap)
        if not target.is_finite() or target <= 0 or max_boxes < 0 or (
                timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0)):
            raise ValueError("Invalid certificate budget")
        deadline = math.inf if timeout_s is None else time.monotonic()+timeout_s
        cache = {}

        def point(z):
            if z not in cache:
                cache[z] = self.terms(z,z,False)[0]
            return cache[z]

        center = min(self.hi,max(self.lo,a.midpoint(self.lo,self.hi)))
        best = min((point(z).hi,z) for z in {self.lo,self.hi,center})
        queue = []
        retired_lower = None

        def insert(l,h):
            nonlocal best,retired_lower
            mid = min(h,max(l,a.midpoint(l,h)))
            for z in {l,h,mid}:
                best = min(best,(point(z).hi,z))
            try:
                f,g,_ = self.terms(l,h,True)
                if g.lo >= 0:
                    f0 = point(l)
                    floor = f0.lo
                elif g.hi <= 0:
                    f0 = point(h)
                    floor = f0.lo
                else:
                    # Mean-value lower bound from a point evaluation and a
                    # rigorous envelope of the derivative on the whole box.
                    variation = a.mul(g,a.sub(I(l,h),a.point(mid)))
                    floor = max(f.lo,a.add(point(mid),variation).lo)
            except ZeroDivisionError:
                floor = self.terms(l,h,False)[0].lo
            floor = max(Decimal(0),floor)
            if floor <= best[0]:
                heapq.heappush(queue,(floor,l,h))
            else:
                retired_lower = floor if retired_lower is None else min(retired_lower,floor)

        insert(self.lo,self.hi)
        explored = 0
        reason = "gap_certified"
        while queue:
            low,l,h = queue[0]
            if a.up.subtract(best[0],low) <= target:
                break
            if explored >= max_boxes or time.monotonic() >= deadline:
                reason = "budget" if explored >= max_boxes else "timeout"
                break
            mid = a.midpoint(l,h)
            if not l < mid < h:
                reason = "precision_limit"
                break
            heapq.heappop(queue)
            insert(l,mid)
            insert(mid,h)
            explored += 1
        lower = min([best[0]]+([queue[0][0]] if queue else [])+
                    ([retired_lower] if retired_lower is not None else []))
        return dict(status=reason, lower=str(lower),upper=str(best[0]),
                    gap=str(a.up.subtract(best[0],lower)),height=str(best[1]),
                    splits=explored, unpruned_boxes=len(queue))

    def stationary(self, *, width="1e-5", max_boxes=4096, timeout_s=2.):
        """All interior stationary points are covered by isolated or unresolved boxes."""
        a = self.a
        tol = Decimal(width)
        if not tol.is_finite() or tol<=0 or max_boxes<0 or (
                timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s<0)):
            raise ValueError("Invalid stationary budget")
        deadline = math.inf if timeout_s is None else time.monotonic()+timeout_s
        pending = [(self.lo,self.hi)] if self.lo<self.hi else []
        details, endpoints = [], []
        # Endpoints are always independently evaluated, including singleton.
        for z in sorted({self.lo,self.hi}):
            f = self.terms(z,z,False)[0]
            endpoints.append(dict(height=str(z),lower=str(f.lo),upper=str(f.hi)))

        def sign(x):
            return 1 if x.lo>0 else -1 if x.hi<0 else 0

        def emit(l,h,label):
            details.append(dict(lo=str(l),hi=str(h),kind=label))

        explored = 0
        reason = "covered"
        while pending:
            if explored>=max_boxes or time.monotonic()>=deadline:
                reason = "budget" if explored>=max_boxes else "timeout"
                for l,h in pending:
                    emit(l,h,"unresolved_"+reason)
                break
            l,h = pending.pop()
            explored += 1
            try:
                _,g,H = self.terms(l,h,True)
                if sign(g):
                    continue
                hh = sign(H)
                gl = sign(self.terms(l,l,True)[1])
                gh = sign(self.terms(h,h,True)[1])
                if hh and gl==gh and gl:
                    continue
                proven = hh!=0 and gl==-hh and gh==hh
                kind = ("unique_minimum" if hh>0 else "unique_maximum") if proven else "possible"
            except ZeroDivisionError:
                kind = "possible_nonsmooth"
            if a.up.subtract(h,l)<=tol:
                emit(l,h,kind)
                continue
            mid = a.midpoint(l,h)
            if not l<mid<h:
                emit(l,h,"unresolved_precision")
                if reason=="covered":
                    reason = "precision_limit"
                continue
            pending.extend(((mid,h),(l,mid)))
        return dict(status=reason,boxes=details,endpoints=endpoints,visited=explored)
