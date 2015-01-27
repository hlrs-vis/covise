#include "gaalet.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <cmath>

typedef gaalet::algebra<gaalet::signature<3,0> > em;
typedef em::mv<1,2,4>::type Vector;
typedef em::mv<0,3,5,6>::type Rotor;

struct rotation_functor
{
   rotation_functor(const Rotor& setR)
      : R(setR),
        invR(!setR)
   { }

   __host__ __device__
      Vector operator()(const Vector& x) const
      { 
         return grade<1>(R*x*invR);
      }

   Rotor R;
   Rotor invR;
};


int main()
{
   thrust::host_vector<Vector> h_x(100);
   for(int i=0; i<100; ++i) {
      h_x[i][0] = 0.1*i; h_x[i][1] = 0.2*i; h_x[i][2] = 0.3*i;
   }

   thrust::device_vector<Vector> d_x = h_x;

   thrust::device_vector<Vector> d_y(100);

   Rotor R;
   R[0] = cos(-0.5*0.5*M_PI); R[1] = sin(-0.5*0.5*M_PI);

   thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), rotation_functor(R));

   thrust::host_vector<Vector> h_y = d_y;

   for(int i=0; i<100; ++i) {
      std::cout << i << ": x: " << h_x[i] << ", y: " << h_y[i] << std::endl;
   }
}
