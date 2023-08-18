#include <iostream>
#include "mat4.h"

int main() {
  using namespace math;

  mat4f a(1,0,0,0,
          0,0,1,0,
          0,1,0,0,
          0,0,0,1);
  std::cout << "a: " << a << '\n';

  mat4f b = mat4f::identity();
  std::cout << "b: " << b << '\n';

  a = inverse(transpose(a));
  std::cout << "inv(trans(a)): " << a << '\n';

  mat4f c = a*b;
  std::cout << "a*b: " << c << '\n';

  vec4f v{0.f,.4f,.8f,1.2f};
  std::cout << "v: " << v << '\n';

  vec4f u = c*v;
  std::cout << "c*v: " << u << '\n';
}


