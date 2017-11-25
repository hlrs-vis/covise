/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include "MathUtils.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

double Vlength(const float V[3])
{
    double vlen = sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
    return (FPZERO(vlen) ? 0.0 : vlen);
}

void Vdiff(float V[3], const float A[3], const float B[3])
{
    V[0] = A[0] - B[0];
    V[1] = A[1] - B[1];
    V[2] = A[2] - B[2];
}

void Vcross(float V[3], const float A[3], const float B[3])
{
    float x, y, z; // Use temps so V can be A or B
    x = A[1] * B[2] - A[2] * B[1];
    y = A[2] * B[0] - A[0] * B[2];
    z = A[0] * B[1] - A[1] * B[0];
    V[0] = x;
    V[1] = y;
    V[2] = z;
}

void Vnorm(float V[3])
{
    float vlen = (float)sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);
    if (!FPZERO(vlen))
    {
        V[0] /= vlen;
        V[1] /= vlen;
        V[2] /= vlen;
    }
}

// Note that these matrices are stored in natural (C) order (the transpose
// of the OpenGL matrix). Could change this someday...

void Midentity(double *M)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            M[i * 4 + j] = (i == j) ? 1.0 : 0.0;
}

// Convert from axis/angle to transformation matrix GG p466

void Mrotation(double *M, const float axis[3], float angle)
{
    float a[3] = { axis[0], axis[1], axis[2] };

    Vnorm(a);
    double s = sin(angle);
    double c = cos(angle);
    double t = 1.0 - c;
    double x = a[0];
    double y = a[1];
    double z = a[2];

    M[0 * 4 + 0] = t * x * x + c;
    M[0 * 4 + 1] = t * x * y + s * z;
    M[0 * 4 + 2] = t * x * z - s * y;
    M[0 * 4 + 3] = 0.0;
    M[1 * 4 + 0] = t * x * y - s * z;
    M[1 * 4 + 1] = t * y * y + c;
    M[1 * 4 + 2] = t * y * z + s * x;
    M[1 * 4 + 3] = 0.0;
    M[2 * 4 + 0] = t * x * z + s * y;
    M[2 * 4 + 1] = t * y * z - s * x;
    M[2 * 4 + 2] = t * z * z + c;
    M[2 * 4 + 3] = 0.0;
    M[3 * 4 + 0] = M[3 * 4 + 1] = M[3 * 4 + 2] = 0.0;
    M[3 * 4 + 3] = 1.0;
}

void Mrotation(double *M, const float axisAngle[4])
{
    Mrotation(M, axisAngle, axisAngle[3]);
}

void Mscale(double *M, const float scale[3])
{
    Midentity(M);
    for (int i = 0; i < 3; ++i)
        M[i * 4 + i] = scale[i];
}

void Mtrans(double *M, const float trans[3])
{
    Midentity(M);
    for (int i = 0; i < 3; ++i)
        M[3 * 4 + i] = trans[i];
}

void Mtrans(double *M, float tx, float ty, float tz)
{
    Midentity(M);
    M[3 * 4 + 0] = tx;
    M[3 * 4 + 1] = ty;
    M[3 * 4 + 2] = tz;
}

void MM(double *M, const double *N)
{
    double m[16];

    memcpy(m, M, sizeof(m));
    Mmult(M, m, N);
}

void Mmult(double *M, const double *M1, const double *M2)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            M[i * 4 + j] = M1[i * 4 + 0] * M2[0 * 4 + j] + M1[i * 4 + 1] * M2[1 * 4 + j] + M1[i * 4 + 2] * M2[2 * 4 + j] + M1[i * 4 + 3] * M2[3 * 4 + j];
}

void VM(float V[3], const double *M, const float A[3])
{
    float v[3] = // Allow for V/A aliasing
        {
          A[0], A[1], A[2]
        };
    float V3 = (float)(M[0 * 4 + 3] * v[0] + M[1 * 4 + 3] * v[1] + M[2 * 4 + 3] * v[2] + M[3 * 4 + 3]);
    for (int i = 0; i < 3; ++i)
        V[i] = (float)(M[0 * 4 + i] * v[0] + M[1 * 4 + i] * v[1] + M[2 * 4 + i] * v[2] + M[3 * 4 + i]) / V3;
}

bool Minvert(double *M, const double *N)
{
    for (int i = 0; i < 16; i++)
        M[i] = N[i];

    int p[4];
    for (int k = 0; k < 4; k++)
    {
        p[k] = 0;
        double sup = 0.0;
        for (int i = k; i < 4; i++)
        {
            double s = 0.0;
            for (int j = k; j < 4; j++)
                s += fabs(M[i * 4 + j]);
            double q = fabs(M[i * 4 + k]) / s;
            if (sup < q)
            {
                sup = q;
                p[k] = i;
            }
        }
        if (FPZERO(sup))
            return false;
        if (p[k] != k)
            for (int j = 0; j < 4; j++)
            {
                double h = M[k * 4 + j];
                M[k * 4 + j] = M[p[k] * 4 + j];
                M[p[k] * 4 + j] = h;
            }
        double pivot = M[k * 4 + k];
        for (int j = 0; j < 4; j++)
            if (j != k)
            {
                M[k * 4 + j] = -M[k * 4 + j] / pivot;
                for (int i = 0; i < 4; i++)
                    if (i != k)
                        M[i * 4 + j] += M[i * 4 + k] * M[k * 4 + j];
            }
        for (int i = 0; i < 4; i++)
            M[i * 4 + k] = M[i * 4 + k] / pivot;
        M[k * 4 + k] = 1.0 / pivot;
    }

    for (int k = 4 - 1; k >= 0; k--)
        if (p[k] != k)
            for (int i = 0; i < 4; i++)
            {
                double h = M[i * 4 + k];
                M[i * 4 + k] = M[i * 4 + p[k]];
                M[i * 4 + p[k]] = h;
            }

    return true;
}

void Mcopy(double *M, const double *N)
{
    for (int i = 0; i < 16; i++)
        M[i] = N[i];
}

bool MgetRot(float orientation[3], float *angle, const double *M)
{
    // copy
    double M2[3][3];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            M2[i][j] = M[i * 4 + j];
        }
    }

    // normalize vectors in upper left 3x3 matrix
    for (int i = 0; i < 3; i++)
    {
        double len = sqrt(M2[i][0] * M2[i][0] + M2[i][1] * M2[i][1] + M2[i][2] * M2[i][2]);
        if (FPZERO(len))
        {
            fprintf(stderr, "MgetRot: matrix almost singular\n");
            continue;
        }
        M2[i][0] /= len;
        M2[i][1] /= len;
        M2[i][2] /= len;
    }

    // transform to quaternion representation (X, Y, Z, W)
    double Q[4] = { 0.0, 0.0, 0.0, 0.0 };

    double tr = M2[0][0] + M2[1][1] + M2[2][2];
    if (tr > 0.0)
    {
        double s = sqrt(tr + 1.0);
        Q[3] = s * 0.5;
        s = 0.5 / s;
        Q[0] = (M2[1][2] - M2[2][1]) * s;
        Q[1] = (M2[2][0] - M2[0][2]) * s;
        Q[2] = (M2[0][1] - M2[1][0]) * s;
    }
    else
    {
        int i = 0;
        if (M2[1][1] > M2[i][i])
            i = 1;
        if (M2[2][2] > M2[i][i])
            i = 2;
        int index[3] = { 1, 2, 0 };
        int j = index[i];
        int k = index[j];

        double s = sqrt(M2[i][i] - M2[j][j] - M2[k][k] + 1.0);
        Q[i] = s * 0.5;
        if (!FPZERO(s))
        {
            s = 0.5 / s;
        }
        Q[j] = (M2[i][j] + M2[j][i]) * s;
        Q[k] = (M2[i][k] + M2[k][i]) * s;
        Q[3] = (M2[j][k] + M2[k][j]) * s;
    }

    // extract axis and angle from quaternion
    *angle = (float)acos(Q[3]);
    if (!FPZERO(*angle))
    {
        orientation[0] = (float)Q[0];
        orientation[1] = (float)Q[1];
        orientation[2] = (float)Q[2];
        Vnorm(orientation);
        *angle *= 2.0;
    }
    else
    {
        orientation[0] = 1.0;
        orientation[1] = 0.0;
        orientation[2] = 0.0;
    }

    return true;
}

void MgetTrans(float trans[3], const double *M)
{
    for (int i = 0; i < 3; i++)
    {
        trans[i] = (float)M[3 * 4 + i];
    }
}

void Mprint(const double *M)
{
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            fprintf(stderr, " %f", M[i * 4 + j]);
        fprintf(stderr, "\n");
    }
}

void Vprint(const float V[3])
{
    fprintf(stderr, "(%f %f %f)\n", V[0], V[1], V[2]);
}
