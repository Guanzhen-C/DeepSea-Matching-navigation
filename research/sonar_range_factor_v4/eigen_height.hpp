#pragma once
// Microkernel for an already-isolated convex bracket. Does not screen global boxes.
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>

namespace oceansim_range_v4 {
struct Result {double height; int status; int iterations;};
// status: 0 success, 1 invalid/bracket or curvature, 2 iteration budget
inline Result convex_root(const Eigen::Vector2d& d, const Eigen::Vector2d& z,
                          const Eigen::Vector2d& r, const Eigen::Matrix2d& W,
                          double lo, double hi, double warm, int budget=55) {
  auto eval=[&](double x) {
    const Eigen::Array2d t=Eigen::Array2d::Constant(x)-z.array();
    const Eigen::Array2d rho=(d.array()+t.square()).sqrt();
    const Eigen::Vector2d b=(t/rho).matrix();
    const Eigen::Vector2d wq=W*(rho.matrix()-r);
    return Eigen::Vector2d(wq.dot(b), b.dot(W*b)+wq.dot((d.array()/rho.cube()).matrix()));
  };
  if(!d.allFinite() || !z.allFinite() || !r.allFinite() || !W.allFinite() ||
     !std::isfinite(lo) || !std::isfinite(hi) || lo>hi || hi>z.minCoeff() ||
     d.minCoeff()<0 || r.minCoeff()<=0 || budget<0) {
    return {std::numeric_limits<double>::quiet_NaN(),1,0};
  }
  const auto l=eval(lo), h=eval(hi);
  if(!l.allFinite() || !h.allFinite() || l[0]>0 || h[0]<0) return {0,1,0};
  if(l[0]==0 || h[0]==0) return {l[0]==0?lo:hi,0,0};
  double x=std::isfinite(warm)&&warm>lo&&warm<hi?warm:lo+0.5*(hi-lo);
  for(int i=0;i<budget;++i) {
    const auto y=eval(x);
    if(!y.allFinite() || y[1]<=0) return {x,1,i+1};
    if((std::abs(y[0])<=4e-13&&std::abs(y[0])<=y[1]*2e-11)||hi-lo<=4e-12)
      return {x,0,i+1};
    if(y[0]<0) lo=x; else hi=x;
    const double next=x-y[0]/y[1];
    x=next>lo&&next<hi?next:lo+0.5*(hi-lo);
  }
  return {x,2,budget};
}
}  // namespace oceansim_range_v4
