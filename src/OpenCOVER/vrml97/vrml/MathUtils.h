/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRML_MATHUTILS_H
#define VRML_MATHUTILS_H

#include <math.h>

//
// Some useful math stuff that should someday be made into C++...
//

// Define single precision floating point "zero".

#define FPTOLERANCE 1.0e-7
#define FPZERO(n) (fabs(n) <= FPTOLERANCE)
#define FPEQUAL(a, b) FPZERO((a) - (b))

// Vector ops

#define Vset(V, A) /* V <= A */ \
    (((V)[0] = (A)[0]), ((V)[1] = (A)[1]), ((V)[2] = (A)[2]))
#define Vscale(V, s) /* V *= s */ \
    (((V)[0] *= (s)), ((V)[1] *= (s)), ((V)[2] *= s))
#define Vdot(A, B) /* A.B */ \
    ((A)[0] * (B)[0] + (A)[1] * (B)[1] + (A)[2] * (B)[2])
#define Vadd(A, B) (((A)[0] += (B)[0]), ((A)[1] += (B)[1]), ((A)[2] += (B)[2]))
#define Vsub(A, B) (((A)[0] -= (B)[0]), ((A)[1] -= (B)[1]), ((A)[2] -= (B)[2]))

typedef double VrmlMat4[16];
typedef float VrmlVec3[3];
typedef float VrmlVec4[4];

double Vlength(const float V[3]); // |V|

// V <= A - B
void Vdiff(float V[3], const float A[3], const float B[3]);
// V <= A x B
void Vcross(float V[3], const float A[3], const float B[3]);
void Vnorm(float V[3]); // V <= V / |V|

void Vprint(const float V[3]);

// V <= MA
void VM(float V[3], const double *M, const float A[3]);

// Matrix ops

void Midentity(double *M); // M <= I
void Mrotation(double *M, const float r[4]); // M <= rotation about axis/angle
// M <= rotation about axis/angle
void Mrotation(double *M, const float axis[3], float angle);
void Mscale(double *M, const float s[3]); // M <= scale by s
void Mtrans(double *M, const float t[3]); // M <= translate by t
// M <= translate by t
void Mtrans(double *M, float tx, float ty, float tz);
bool Minvert(double *M, const double *N); // M <= 1/N
void Mcopy(double *M, const double *N);
// M <= M1*M2
void Mmult(double *M, const double *M1, const double *M2);

void MM(double *M, const double *N); // M <= MN

bool MgetRot(float orientation[3], float *angle, const double *M);
void MgetTrans(float trans[3], const double *M);

void Mprint(const double *M);
#endif
