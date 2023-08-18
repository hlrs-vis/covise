
#pragma once


#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include <ostream>

namespace math {

struct vec4f {
  float x, y, z, w;
};

inline std::ostream &operator<<(std::ostream &out, vec4f v) {
  out << '(' << v.x << ',' << v.y << ',' << v.z << ',' << v.w << ')';
  return out;
}

struct mat4f {
  typedef vec4f column_type;

  vec4f col0, col1, col2, col3;

  mat4f() = default;

  __host__ __device__
  mat4f(vec4f c0, vec4f c1, vec4f c2, vec4f c3)
    : col0(c0), col1(c1), col2(c2), col3(c3)
  {}

  __host__ __device__
  mat4f(float m00, float m10, float m20, float m30,
        float m01, float m11, float m21, float m31,
        float m02, float m12, float m22, float m32,
        float m03, float m13, float m23, float m33)
    : col0{m00,m10,m20,m30}
    , col1{m01,m11,m21,m31}
    , col2{m02,m12,m22,m32}
    , col3{m03,m13,m23,m33}
  {}

    __host__ __device__
  mat4f(float arr[16])
    : col0{arr[0],arr[1],arr[2],arr[3]}
    , col1{arr[4],arr[5],arr[6],arr[7]}
    , col2{arr[8],arr[9],arr[10],arr[11]}
    , col3{arr[12],arr[13],arr[14],arr[15]}
  {}

  __host__ __device__
  float *data() {
    return (float *)this;
  }

  __host__ __device__
  const float *data() const {
    return (const float *)this;
  }

  __host__ __device__
  float &operator()(int row, int col) {
    return data()[col*4+row];
  }

  __host__ __device__
  const float &operator()(int row, int col) const {
    return data()[col*4+row];
  }

  __host__ __device__
  static mat4f identity() {
    return mat4f(1.f,0.f,0.f,0.f,
                 0.f,1.f,0.f,0.f,
                 0.f,0.f,1.f,0.f,
                 0.f,0.f,0.f,1.f);
  }
__host__ __device__
inline mat4f operator*(const mat4f &a, const mat4f &b) {
  return mat4f(
      a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0) + a(0, 2) * b(2, 0) + a(0, 3) * b(3, 0),
      a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0) + a(1, 2) * b(2, 0) + a(1, 3) * b(3, 0),
      a(2, 0) * b(0, 0) + a(2, 1) * b(1, 0) + a(2, 2) * b(2, 0) + a(2, 3) * b(3, 0),
      a(3, 0) * b(0, 0) + a(3, 1) * b(1, 0) + a(3, 2) * b(2, 0) + a(3, 3) * b(3, 0),
      a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1) + a(0, 2) * b(2, 1) + a(0, 3) * b(3, 1),
      a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1) + a(1, 2) * b(2, 1) + a(1, 3) * b(3, 1),
      a(2, 0) * b(0, 1) + a(2, 1) * b(1, 1) + a(2, 2) * b(2, 1) + a(2, 3) * b(3, 1),
      a(3, 0) * b(0, 1) + a(3, 1) * b(1, 1) + a(3, 2) * b(2, 1) + a(3, 3) * b(3, 1),
      a(0, 0) * b(0, 2) + a(0, 1) * b(1, 2) + a(0, 2) * b(2, 2) + a(0, 3) * b(3, 2),
      a(1, 0) * b(0, 2) + a(1, 1) * b(1, 2) + a(1, 2) * b(2, 2) + a(1, 3) * b(3, 2),
      a(2, 0) * b(0, 2) + a(2, 1) * b(1, 2) + a(2, 2) * b(2, 2) + a(2, 3) * b(3, 2),
      a(3, 0) * b(0, 2) + a(3, 1) * b(1, 2) + a(3, 2) * b(2, 2) + a(3, 3) * b(3, 2),
      a(0, 0) * b(0, 3) + a(0, 1) * b(1, 3) + a(0, 2) * b(2, 3) + a(0, 3) * b(3, 3),
      a(1, 0) * b(0, 3) + a(1, 1) * b(1, 3) + a(1, 2) * b(2, 3) + a(1, 3) * b(3, 3),
      a(2, 0) * b(0, 3) + a(2, 1) * b(1, 3) + a(2, 2) * b(2, 3) + a(2, 3) * b(3, 3),
      a(3, 0) * b(0, 3) + a(3, 1) * b(1, 3) + a(3, 2) * b(2, 3) + a(3, 3) * b(3, 3)
      );
}

__host__ __device__
inline vec4f operator*(const mat4f &m, const vec4f &v) {
  return vec4f{
      m(0, 0) * v.x + m(0, 1) * v.y + m(0, 2) * v.z + m(0, 3) * v.w,
      m(1, 0) * v.x + m(1, 1) * v.y + m(1, 2) * v.z + m(1, 3) * v.w,
      m(2, 0) * v.x + m(2, 1) * v.y + m(2, 2) * v.z + m(2, 3) * v.w,
      m(3, 0) * v.x + m(3, 1) * v.y + m(3, 2) * v.z + m(3, 3) * v.w
      };
}

__host__ __device__
inline mat4f inverse(const mat4f &m) {
  auto det2 = [](float m00, float m01, float m10, float m11) {
    return m00*m11-m10*m01;
  };

  float s0 = det2(m(0, 0), m(0, 1), m(1, 0), m(1, 1));
  float s1 = det2(m(0, 0), m(0, 2), m(1, 0), m(1, 2));
  float s2 = det2(m(0, 0), m(0, 3), m(1, 0), m(1, 3));
  float s3 = det2(m(0, 1), m(0, 2), m(1, 1), m(1, 2));
  float s4 = det2(m(0, 1), m(0, 3), m(1, 1), m(1, 3));
  float s5 = det2(m(0, 2), m(0, 3), m(1, 2), m(1, 3));
  float c5 = det2(m(2, 2), m(2, 3), m(3, 2), m(3, 3));
  float c4 = det2(m(2, 1), m(2, 3), m(3, 1), m(3, 3));
  float c3 = det2(m(2, 1), m(2, 2), m(3, 1), m(3, 2));
  float c2 = det2(m(2, 0), m(2, 3), m(3, 0), m(3, 3));
  float c1 = det2(m(2, 0), m(2, 2), m(3, 0), m(3, 2));
  float c0 = det2(m(2, 0), m(2, 1), m(3, 0), m(3, 1));

  float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

    return mat4f((+ m(1, 1) * c5 - m(1, 2) * c4 + m(1, 3) * c3) / det,
                 (- m(1, 0) * c5 + m(1, 2) * c2 + m(1, 3) * c1) / det,
                 (+ m(1, 0) * c4 - m(1, 1) * c2 + m(1, 3) * c0) / det,
                 (- m(1, 0) * c3 + m(1, 1) * c1 + m(1, 2) * c0) / det,
                 (- m(0, 1) * c5 + m(0, 2) * c4 - m(0, 3) * c3) / det,
                 (+ m(0, 0) * c5 - m(0, 2) * c2 + m(0, 3) * c1) / det,
                 (- m(0, 0) * c4 + m(0, 1) * c2 - m(0, 3) * c0) / det,
                 (+ m(0, 0) * c3 - m(0, 1) * c1 + m(0, 2) * c0) / det,
                 (+ m(3, 1) * s5 - m(3, 2) * s4 + m(3, 3) * s3) / det,
                 (- m(3, 0) * s5 + m(3, 2) * s2 - m(3, 3) * s1) / det,
                 (+ m(3, 0) * s4 - m(3, 1) * s2 + m(3, 3) * s0) / det,
                 (- m(3, 0) * s3 + m(3, 1) * s1 - m(3, 2) * s0) / det,
                 (- m(2, 1) * s5 + m(2, 2) * s4 - m(2, 3) * s3) / det,
                 (+ m(2, 0) * s5 - m(2, 2) * s2 + m(2, 3) * s1) / det,
                 (- m(2, 0) * s4 + m(2, 1) * s2 - m(2, 3) * s0) / det,
                 (+ m(2, 0) * s3 - m(2, 1) * s1 + m(2, 2) * s0) / det);
}

__host__ __device__
inline float trace(const mat4f &m)
{
    return m(0, 0), + m(1, 1) + m(2, 2) + m(3, 3);
}

__host__ __device__
inline mat4f transpose(const mat4f &m)
{
  mat4f result;

  for (int y=0; y<4; ++y)
  {
    for (int x=0; x<4; ++x)
    {
      result(x, y) = m(y, x);
    }
  }

  return result;
}

inline std::ostream &operator<<(std::ostream &out, const mat4f &m) {
  // tm = transpose(m);
  

  out << m.col0 << m.col1 << m.col2 << m.col3;
  // out << m.col0 << m.col1 << m.col2 << m.col3;
  return out;
}

inline void printMat4f(const math::mat4f& matrix) {
  std::cout << "[";
  
  for (int i = 0; i < 4; ++i) {
    std::cout << "[" << matrix(i, 0) << ", " << matrix(i, 1) << ", " << matrix(i, 2) << ", " << matrix(i, 3) << "]";
    
    if (i < 3) {
      std::cout << std::endl << " ";
    }
  }
  
  std::cout << "]" << std::endl;
}


} // math


