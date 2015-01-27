#include "gaalet.h"
#include <cmath>

typedef gaalet::algebra<gaalet::signature<4,1> > cm;

__global__ void test()
{
   //int i = threadIdx.x;

   cm::mv<0x01>::type e1(1.0);
   cm::mv<0x02>::type e2(1.0);
   cm::mv<0x04>::type e3(1.0);
   cm::mv<0x08>::type ep(1.0);
   cm::mv<0x10>::type em(1.0);

   cm::mv<0x00>::type one(1.0);

   cm::mv<0x08, 0x10>::type e0 = 0.5*(em-ep);
   cm::mv<0x08, 0x10>::type einf = em+ep;
}

int main()
{
   std::cout << "Hello Gaalet on cuda!" << std::endl;

   cudaSetDevice( 0 );

   unsigned int num_threads = 4;
   dim3 threads( num_threads );

   test <<< 1, threads >>>();

   cudaThreadExit();
}
