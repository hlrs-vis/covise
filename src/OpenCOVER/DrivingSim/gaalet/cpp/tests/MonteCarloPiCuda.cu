#include "gaalet.h"
#include <iostream>
#include <cstdlib>

typedef gaalet::algebra<gaalet::signature<4,1> > cm;

__device__ __shared__ unsigned int d_n_s;

__global__ void test()
{
   __shared__ float r;
   
   __shared__ cm::mv<0x01>::type e1;
   __shared__ cm::mv<0x02>::type e2;
   __shared__ cm::mv<0x04>::type e3;
   __shared__ cm::mv<0x08>::type ep;
   __shared__ cm::mv<0x10>::type em;
   __shared__ cm::mv<0x00>::type one;

   __shared__ cm::mv<0x08, 0x10>::type e0;
   __shared__ cm::mv<0x08, 0x10>::type einf;

   __shared__ cm::mv<0x08, 0x10>::type S;

   if(threadIdx.x == 0 && threadIdx.y==0 && threadIdx.z==0) {
      d_n_s = 0;
      r = (float)blockDim.x;
      e1[0] = 1.0;
      e2[0] = 1.0;
      e3[0] = 1.0;
      em[0] = 1.0;
      ep[0] = 1.0;
      one[0] = 1.0;
      e0 = 0.5*(em-ep);
      einf = em+ep;
      S = e0 - 0.5*r*r*einf;
   }

   cm::mv<0x01, 0x02, 0x04>::type x = ((float)threadIdx.x*e1 + (float)threadIdx.y*e2 + (float)threadIdx.z*e3)*r;
   cm::mv<0x01, 0x02, 0x04, 0x08, 0x10>::type P = x + 0.5*(x&x)*einf + e0;
   float d = eval(S&P);
   if(d>=0.0) {
      //atomicAdd(&n_s, 1);
      ++d_n_s;
   }
}


int main()
{
   std::cout << "Hello Gaalet Monte Carlo on Cuda!" << std::endl;

   cudaSetDevice(0);
   cudaError_t error;

   unsigned int r = 5;
   dim3 threads( r, r, r );

   test <<< 1, threads >>>();
   cudaThreadSynchronize();
   error = cudaGetLastError();
   if(error != cudaSuccess) 
   {
      std::cerr << "Cuda error: " << cudaGetErrorString(error) << std::endl;
   }

   unsigned int n_s;
   cudaMemcpyFromSymbol(&n_s, "d_n_s", sizeof(n_s), 0, cudaMemcpyDeviceToHost);
   error = cudaGetLastError();
   if(error != cudaSuccess) 
   {
      std::cerr << "Cuda error: " << cudaGetErrorString(error) << std::endl;
   }

   unsigned int n_q = r*r*r;
   float pi = 6.0*(float)n_s/(float)n_q;

   std::cout << "Pi: " << pi << std::endl;

   cudaThreadExit();
}
