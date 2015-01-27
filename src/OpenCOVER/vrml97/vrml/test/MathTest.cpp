/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <vrml97/vrml/MathUtils.h>
#include <stdio.h>
#include <math.h>

void Mprint(const double *M)
{
    for (int i = 0; i < 16; i++)
    {
        fprintf(stderr, "%1.2f ", M[i]);
        if (i % 4 == 3)
            fprintf(stderr, "\n");
    }
}

int main()
{
    double M1[16], M2[16], M3[16];
#if 0
   Midentity(M1);
   Mprint(M1);
   Minvert(M2, M1);
   Mprint(M2);
#endif

#if 0
   Mtrans(M1, 1.0, 2.0, 3.0);
   //Mprint(M1);
   Minvert(M2, M1);
   //Mprint(M2);
   Mmult(M3, M1, M2);
   Mprint(M3);
#endif

    float axis[4] = { 0.0, 0.0, 1.0, M_PI / 4.0 };
    float angle;
    Mrotation(M1, axis);
    Mprint(M1);
#if 0
   Minvert(M2, M1);
   //Mprint(M2);
   Mmult(M3, M1, M2);
   Mprint(M3);
   MgetRot(axis, &angle, M1);
   fprintf(stderr, "axis=(%2.1f %2.1f %2.1f), angle=%f\n",
      axis[0], axis[1], axis[2], angle);
#endif

#if 0
   Mtrans(M1, 1.0, 2.0, 3.0);
   Mtrans(M2, 2.0, 2.0, 2.0);
   Mmult(M3, M1, M2);
   Mprint(M3);
#endif
    MgetRot(axis, &axis[3], M1);

    fprintf(stderr, "axis=(%f %f %f),  angle=%f\n", axis[0], axis[1], axis[2], axis[3]);
}
