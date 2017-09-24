#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
#include <stdio.h>
#include <DraftTube/include/nonlinpart.h>

static double mypower(double basis, int exp);

void nonlinearpartition(float *x, int numx, float len, float factor)
{
   int i;
   float sum;
   float delta;
   float part;

   delta = float(pow(factor, (1.0/(double)(numx-2))));
   for (sum = 0.0, i = 1; i < numx; i++)
      sum += float(mypower(delta, i-1));

   part = 1.0f/sum;
   x[0] = 0.0f;
   for (i = 1; i < numx-1; i++)
      x[i] = x[i-1] + part * float(mypower(delta, i-1) * len);
   x[numx-1] = len;
#ifdef   DEBUG
   fprintf(stderr, "\tnonlinearpartition(): numx = %d, delta = %f, sum = %f, part = %f, len = %f, factor = %f, part[1] = %f\n", numx, delta, sum, part, len, factor, x[1]);
#endif
}


static double mypower(double basis, int exp)
{
   int i;
   double res;

   for (res = 1.0, i=0; i < exp; i++)
      res *= basis;

   return res;
}


#ifdef   TEST
int main(int argc, char **argv)
{
   float x[4];

   nonliniearpartition(x, 4, 200.0, 4);
   printf(" 4::\t0: %f\n\t1: %f\n\t2: %f\n\t3: %f\n", x[0], x[1], x[2], x[3]);
   nonliniearpartition(x, 4, 200.0, 10);
   printf("10::\t0: %f\n\t1: %f\n\t2: %f\n\t3: %f\n", x[0], x[1], x[2], x[3]);
}
#endif                                            // TEST
