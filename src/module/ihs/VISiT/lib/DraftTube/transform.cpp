#include "include/transform.h"
#include "../General/include/v.h"

// very funny way, but easy for debugging ;-)
void Transform(float T[3][3], float m[3], float *x, float *y, float *z)
{
   float r[3];
   float t[3];

   r[0] = T[0][0]* *x + T[0][1]* *y + T[0][2]* *z;
   r[1] = T[1][0]* *x + T[1][1]* *y + T[1][2]* *z;
   r[2] = T[2][0]* *x + T[2][1]* *y + T[2][2]* *z;
   V_Add(r, m, t);

   *x = t[0];
   *y = t[1];
   *z = t[2];
}
