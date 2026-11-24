#include "eigen_height.hpp"
#include <iomanip>
#include <iostream>

int main() {
  using namespace oceansim_range_v4;
  Eigen::Vector2d d(125.,218.), z(0.,4.);
  Eigen::Vector2d r(std::sqrt(125.+121.),std::sqrt(218.+225.));
  Eigen::Matrix2d W;
  W<< 1.,-.2,-.2,.8;
  const auto a=convex_root(d,z,r,W,-13.,-8.,-10.);
  const auto b=convex_root(d,z,r,W,-13.,-8.,-10.,0);
  const auto c=convex_root(d,z,r,W,-13.,-12.,-12.5);
  std::cout<<std::setprecision(17)<<a.height<<" "<<a.status<<" "<<a.iterations
           <<" "<<b.status<<" "<<c.status<<"\n";
  return a.status || b.status!=2 || c.status!=1;
}
