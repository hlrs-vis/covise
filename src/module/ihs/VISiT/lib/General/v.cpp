#ifdef   DEMO
#include <stdio.h>
#endif
#include <math.h>
#include "include/v.h"

void V_Add(float *a, float *b, float *res)
{
   int i;

   for (i=0; i<3; i++)
      *(res+i) = *(a+i) + *(b+i);
}


void V_Sub(float *a, float *b, float *res)
{
   int i;

   for (i=0; i<3; i++)
      *(res+i) = *(a+i) - *(b+i);
}


void V_Norm(float *a)
{
   V_MultScal(a, 1/V_Len(a));
}


float V_Len(float *a)
{
   return (sqrt(V_ScalarProduct(a, a)));
}


void V_MultScal(float *a, float s)
{
   int i;

   for (i=0; i<3; i++)
      *(a+i) *= s;
}


float V_ScalarProduct(float *a, float *b)
{
   return (*(a+0)**(b+0)+*(a+1)**(b+1)+*(a+2)**(b+2));
}


float V_Angle(float *a, float *b)
{
   return (V_ScalarProduct(a, b)/(V_Len(a)*V_Len(b)));
}


void V_Copy(float *d, float *s)
{
   *(d+0) = *(s+0);
   *(d+1) = *(s+1);
   *(d+2) = *(s+2);
}


void V_0(float *a)
{
   *(a+0) = *(a+1) = *(a+2) = 0.0;
}


#ifdef   DEMO
int main(int argc, char **argv)
{
   float a[3], b[3];
   float res[3];
   int i;

   a[0] = -8; a[1] = 1; a[2] = 4;
   b[0] = 3; b[1] = 4; b[2] = 12;
   dprintf(0, "a = (%f,%f,%f), Len=%f\n", a[0], a[1], a[2], V_Len(a));
   dprintf(0, "b = (%f,%f,%f), Len=%f\n", b[0], b[1], b[2], V_Len(b));

   V_Sub(a, b, res);
   dprintf(0, "a-b = (%f,%f,%f)\n", res[0], res[1], res[2]);
   V_Sub(b, a, res);
   dprintf(0, "b-a = (%f,%f,%f)\n", res[0], res[1], res[2]);

   dprintf(0, "ScalarProduct: %f\n", V_ScalarProduct(a, b));
   dprintf(0, "Angle(a,b): %f\n", V_Angle(a, b));
   dprintf(0, "Angle(a,b): %f\n", acos(V_Angle(a, b)));
   dprintf(0, "Angle(a,b): %f\n", acos(V_Angle(a, b))/M_PI*180);
   dprintf(0, "==================\n");
   b[0] = 1; b[1] = 0; b[2] = 0;
   a[0] = 1; a[1] = 0; a[2] = 0;
   V_DEB(a);
   dprintf(0, "Angle(a,b): %f\n", acos(V_Angle(a, b)));
   a[0] = 1; a[1] = 1; a[2] = 0;
   V_DEB(a);
   dprintf(0, "Angle(a,b): %f\n", acos(V_Angle(a, b)));
   a[0] = 0; a[1] = 1; a[2] = 0;
   V_DEB(a);
   dprintf(0, "Angle(a,b): %f\n", acos(V_Angle(a, b)));
   a[0] = -1; a[1] = 1; a[2] = 0;
   V_DEB(a);
   dprintf(0, "Angle(a,b): %f\n", acos(V_Angle(a, b)));
   a[0] = -1; a[1] = 0; a[2] = 0;
   V_DEB(a);
   dprintf(0, "Angle(a,b): %f\n", acos(V_Angle(a, b)));
   a[0] = -1; a[1] = -1; a[2] = 0;
   V_DEB(a);
   dprintf(0, "Angle(a,b): %f\n", acos(V_Angle(a, b)));
   a[0] = 0; a[1] = -1; a[2] = 0;
   V_DEB(a);
   dprintf(0, "Angle(a,b): %f\n", acos(V_Angle(a, b)));
}
#endif
