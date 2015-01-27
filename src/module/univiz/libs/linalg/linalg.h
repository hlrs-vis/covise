/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Linaer Algebra Library
//
// CGL ETH Zuerich
// Ronald Peikert
// $Id$
//
// $Log$
//

#ifndef __LINALG_H__
#define __LINALG_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WIN32
#define _USE_MATH_DEFINES
#pragma warning(disable : 4996 4244)

#include <errno.h>
#include <float.h>
#include <math.h>

#define isnan _isnan

/* DBL_MANT_DIG must be less than 4 times of bits of int */
#ifndef DBL_MANT_DIG
#define DBL_MANT_DIG 53 /* in this case, at least 12 digit precision */
#endif
#define BIG_CRITERIA_BIT (1 << DBL_MANT_DIG / 2)
#if BIG_CRITERIA_BIT > 0
#define BIG_CRITERIA (1.0 * BIG_CRITERIA_BIT)
#else
#define BIG_CRITERIA (1.0 * (1 << DBL_MANT_DIG / 4) * (1 << (DBL_MANT_DIG / 2 + 1 - DBL_MANT_DIG / 4)))
#endif
#define SMALL_CRITERIA_BIT (1 << (DBL_MANT_DIG / 3))
#if SMALL_CRITERIA_BIT > 0
#define SMALL_CRITERIA (1.0 / SMALL_CRITERIA_BIT)
#else
#define SMALL_CRITERIA (1.0 * (1 << DBL_MANT_DIG / 4) * (1 << (DBL_MANT_DIG / 3 + 1 - DBL_MANT_DIG / 4)))
#endif

inline double acosh(double x)
{
    if (x < 1)
        x = -1; /* NaN */
    else if (x == 1)
        return 0;
    else if (x > BIG_CRITERIA)
        x += x;
    else
        x += sqrt((x + 1) * (x - 1));
    return log(x);
}

inline double asinh(double x)
{
    int neg = x < 0;
    double z = fabs(x);

    if (z < SMALL_CRITERIA)
        return x;
    if (z < (1.0 / (1 << DBL_MANT_DIG / 5)))
    {
        double x2 = z * z;
        z *= 1 + x2 * (-1.0 / 6.0 + x2 * 3.0 / 40.0);
    }
    else if (z > BIG_CRITERIA)
    {
        z = log(z + z);
    }
    else
    {
        z = log(z + sqrt(z * z + 1));
    }
    if (neg)
        z = -z;
    return z;
}

inline double atanh(double x)
{
    int neg = x < 0;
    double z = fabs(x);

    if (z < SMALL_CRITERIA)
        return x;
    z = log(z > 1 ? -1 : (1 + z) / (1 - z)) / 2;
    if (neg)
        z = -z;
    return z;
}

#endif

#include <math.h>

typedef double vec2[2];
typedef float fvec2[2];
typedef int ivec2[2];
typedef vec2 mat2[2];
typedef fvec2 fmat2[2];
typedef ivec2 imat2[2];

typedef double vec3[3];
typedef float fvec3[3];
typedef int ivec3[3];
typedef vec3 mat3[3];
typedef fvec3 fmat3[3];
typedef ivec3 imat3[3];

typedef double vec4[4];
typedef float fvec4[4];
typedef int ivec4[4];
typedef vec4 mat4[4];
typedef fvec4 fmat4[4];
typedef ivec4 imat4[4];

#ifndef linalg_max
#define linalg_max(a, b) ((a) < (b) ? (b) : (a))
#endif

#ifndef linalg_min
#define linalg_min(a, b) ((a) > (b) ? (b) : (a))
#endif

// Vector dumps

inline void vec3dump(vec3 v, FILE *fp)
{
    fprintf(fp, "\n");
    fprintf(fp, "%g %g %g \n", v[0], v[1], v[2]);
}

inline void fvec3dump(fvec3 v, FILE *fp)
{
    fprintf(fp, "\n");
    fprintf(fp, "%g %g %g \n", v[0], v[1], v[2]);
}

// Matrix dumps

inline void mat2dump(mat2 a, FILE *fp)
{
    fprintf(fp, "\n");
    fprintf(fp, "%15.6f %15.6f \n", a[0][0], a[0][1]);
    fprintf(fp, "%15.6f %15.6f \n", a[1][0], a[1][1]);
}

inline void mat3dump(mat3 a, FILE *fp)
{
    fprintf(fp, "\n");
    fprintf(fp, "%15.6f %15.6f %15.6f \n", a[0][0], a[0][1], a[0][2]);
    fprintf(fp, "%15.6f %15.6f %15.6f \n", a[1][0], a[1][1], a[1][2]);
    fprintf(fp, "%15.6f %15.6f %15.6f \n", a[2][0], a[2][1], a[2][2]);
    //fprintf(fp, "%g %g %g \n", a[0][0], a[0][1], a[0][2]);
    //fprintf(fp, "%g %g %g \n", a[1][0], a[1][1], a[1][2]);
    //fprintf(fp, "%g %g %g \n", a[2][0], a[2][1], a[2][2]);
}

inline void mat4dump(mat4 a, FILE *fp)
{
    fprintf(fp, "\n");
    fprintf(fp, "%15.6f %15.6f %15.6f %15.6f \n", a[0][0], a[0][1], a[0][2], a[0][3]);
    fprintf(fp, "%15.6f %15.6f %15.6f %15.6f \n", a[1][0], a[1][1], a[1][2], a[1][3]);
    fprintf(fp, "%15.6f %15.6f %15.6f %15.6f \n", a[2][0], a[2][1], a[2][2], a[2][3]);
    fprintf(fp, "%15.6f %15.6f %15.6f %15.6f \n", a[3][0], a[3][1], a[3][2], a[3][3]);
}

inline void fmat4dump(fmat4 a, FILE *fp)
{
    fprintf(fp, "\n");
    fprintf(fp, "%15.6f %15.6f %15.6f %15.6f \n", a[0][0], a[0][1], a[0][2], a[0][3]);
    fprintf(fp, "%15.6f %15.6f %15.6f %15.6f \n", a[1][0], a[1][1], a[1][2], a[1][3]);
    fprintf(fp, "%15.6f %15.6f %15.6f %15.6f \n", a[2][0], a[2][1], a[2][2], a[2][3]);
    fprintf(fp, "%15.6f %15.6f %15.6f %15.6f \n", a[3][0], a[3][1], a[3][2], a[3][3]);
}

// Matrix reads

inline void mat2read(mat2 a, FILE *fp)
{
    fscanf(fp, "%lf%lf", &a[0][0], &a[0][1]);
    fscanf(fp, "%lf%lf", &a[1][0], &a[1][1]);
}

inline void mat3read(mat3 a, FILE *fp)
{
    fscanf(fp, "%lf%lf%lf", &a[0][0], &a[0][1], &a[0][2]);
    fscanf(fp, "%lf%lf%lf", &a[1][0], &a[1][1], &a[1][2]);
    fscanf(fp, "%lf%lf%lf", &a[2][0], &a[2][1], &a[2][2]);
}

inline void mat4read(mat4 a, FILE *fp)
{
    fscanf(fp, "%lf%lf%lf%lf", &a[0][0], &a[0][1], &a[0][2], &a[0][3]);
    fscanf(fp, "%lf%lf%lf%lf", &a[1][0], &a[1][1], &a[1][2], &a[1][3]);
    fscanf(fp, "%lf%lf%lf%lf", &a[2][0], &a[2][1], &a[2][2], &a[2][3]);
    fscanf(fp, "%lf%lf%lf%lf", &a[3][0], &a[3][1], &a[3][2], &a[3][3]);
}

// Copying
inline void vec2copy(vec2 a, vec2 b)
{
    memcpy(b, a, sizeof(vec2));
}
inline void fvec2copy(fvec2 a, fvec2 b)
{
    memcpy(b, a, sizeof(fvec2));
}
inline void ivec2copy(ivec2 a, ivec2 b)
{
    memcpy(b, a, sizeof(ivec2));
}

inline void vec3copy(vec3 a, vec3 b)
{
    memcpy(b, a, sizeof(vec3));
}
inline void fvec3copy(fvec3 a, fvec3 b)
{
    memcpy(b, a, sizeof(fvec3));
}
inline void ivec3copy(ivec3 a, ivec3 b)
{
    memcpy(b, a, sizeof(ivec3));
}

inline void vec4copy(vec4 a, vec4 b)
{
    memcpy(b, a, sizeof(vec4));
}
inline void fvec4copy(fvec4 a, fvec4 b)
{
    memcpy(b, a, sizeof(fvec4));
}
inline void ivec4copy(ivec4 a, ivec4 b)
{
    memcpy(b, a, sizeof(ivec4));
}

inline void mat2copy(mat2 a, mat2 b)
{
    memcpy(b, a, sizeof(mat2));
}
inline void fmat2copy(fmat2 a, fmat2 b)
{
    memcpy(b, a, sizeof(fmat2));
}
inline void imat2copy(imat2 a, imat2 b)
{
    memcpy(b, a, sizeof(imat2));
}

inline void mat3copy(mat3 a, mat3 b)
{
    memcpy(b, a, sizeof(mat3));
}
inline void fmat3copy(fmat3 a, fmat3 b)
{
    memcpy(b, a, sizeof(fmat3));
}
inline void imat3copy(imat3 a, imat3 b)
{
    memcpy(b, a, sizeof(imat3));
}

inline void mat4copy(mat4 a, mat4 b)
{
    memcpy(b, a, sizeof(mat4));
}
inline void fmat4copy(fmat4 a, fmat4 b)
{
    memcpy(b, a, sizeof(fmat4));
}
inline void imat4copy(imat4 a, imat4 b)
{
    memcpy(b, a, sizeof(imat4));
}

// Vectors from scalars
inline void vec2set(vec2 a, double a0, double a1)
{
    a[0] = a0;
    a[1] = a1;
}
inline void fvec2set(fvec2 a, float a0, float a1)
{
    a[0] = a0;
    a[1] = a1;
}
inline void ivec2set(ivec2 a, int a0, int a1)
{
    a[0] = a0;
    a[1] = a1;
}

inline void vec3set(vec3 a, double a0, double a1, double a2)
{
    a[0] = a0;
    a[1] = a1;
    a[2] = a2;
}
inline void fvec3set(fvec3 a, float a0, float a1, float a2)
{
    a[0] = a0;
    a[1] = a1;
    a[2] = a2;
}
inline void ivec3set(ivec3 a, int a0, int a1, int a2)
{
    a[0] = a0;
    a[1] = a1;
    a[2] = a2;
}

// Matrices from vectors
inline void mat2setcols(mat2 a, vec2 a0, vec2 a1)
{
    a[0][0] = a0[0];
    a[0][1] = a1[0];
    a[1][0] = a0[1];
    a[1][1] = a1[1];
}

inline void mat3setrows(mat3 a, vec3 a0, vec3 a1, vec3 a2)
{
    a[0][0] = a0[0];
    a[0][1] = a0[1];
    a[0][2] = a0[2];
    a[1][0] = a1[0];
    a[1][1] = a1[1];
    a[1][2] = a1[2];
    a[2][0] = a2[0];
    a[2][1] = a2[1];
    a[2][2] = a2[2];
}

inline void fmat3setrows(mat3 a, fvec3 a0, fvec3 a1, fvec3 a2)
{
    a[0][0] = a0[0];
    a[0][1] = a0[1];
    a[0][2] = a0[2];
    a[1][0] = a1[0];
    a[1][1] = a1[1];
    a[1][2] = a1[2];
    a[2][0] = a2[0];
    a[2][1] = a2[1];
    a[2][2] = a2[2];
}

inline void mat3setcols(mat3 a, vec3 a0, vec3 a1, vec3 a2)
{
    a[0][0] = a0[0];
    a[0][1] = a1[0];
    a[0][2] = a2[0];
    a[1][0] = a0[1];
    a[1][1] = a1[1];
    a[1][2] = a2[1];
    a[2][0] = a0[2];
    a[2][1] = a1[2];
    a[2][2] = a2[2];
}

inline void fmat3setcols(fmat3 a, fvec3 a0, fvec3 a1, fvec3 a2)
{
    a[0][0] = a0[0];
    a[0][1] = a1[0];
    a[0][2] = a2[0];
    a[1][0] = a0[1];
    a[1][1] = a1[1];
    a[1][2] = a2[1];
    a[2][0] = a0[2];
    a[2][1] = a1[2];
    a[2][2] = a2[2];
}

// Vectors from matrices
inline void mat3getcols(mat3 a, vec3 a0, vec3 a1, vec3 a2)
{
    a0[0] = a[0][0];
    a1[0] = a[0][1];
    a2[0] = a[0][2];
    a0[1] = a[1][0];
    a1[1] = a[1][1];
    a2[1] = a[1][2];
    a0[2] = a[2][0];
    a1[2] = a[2][1];
    a2[2] = a[2][2];
}

inline void fmat3getcols(fmat3 a, fvec3 a0, fvec3 a1, fvec3 a2)
{
    a0[0] = a[0][0];
    a1[0] = a[0][1];
    a2[0] = a[0][2];
    a0[1] = a[1][0];
    a1[1] = a[1][1];
    a2[1] = a[1][2];
    a0[2] = a[2][0];
    a1[2] = a[2][1];
    a2[2] = a[2][2];
}

// Conversions
inline void fvec2tovec2(fvec2 a, vec2 b)
{
    b[0] = (double)a[0];
    b[1] = (double)a[1];
}
inline void ivec2tovec2(ivec2 a, vec2 b)
{
    b[0] = (double)a[0];
    b[1] = (double)a[1];
}
inline void vec2tofvec2(vec2 a, fvec2 b)
{
    b[0] = (float)a[0];
    b[1] = (float)a[1];
}
inline void ivec2tofvec2(ivec2 a, fvec2 b)
{
    b[0] = (float)a[0];
    b[1] = (float)a[1];
}
inline void vec2toivec2(vec2 a, ivec2 b)
{
    b[0] = (int)a[0];
    b[1] = (int)a[1];
}
inline void fvec2toivec2(fvec2 a, ivec2 b)
{
    b[0] = (int)a[0];
    b[1] = (int)a[1];
}

inline void fvec3tovec3(fvec3 a, vec3 b)
{
    b[0] = (double)a[0];
    b[1] = (double)a[1];
    b[2] = (double)a[2];
}

inline void ivec3tovec3(ivec3 a, vec3 b)
{
    b[0] = (double)a[0];
    b[1] = (double)a[1];
    b[2] = (double)a[2];
}

inline void vec3tofvec3(vec3 a, fvec3 b)
{
    b[0] = (float)a[0];
    b[1] = (float)a[1];
    b[2] = (float)a[2];
}

inline void ivec3tofvec3(ivec3 a, fvec3 b)
{
    b[0] = (float)a[0];
    b[1] = (float)a[1];
    b[2] = (float)a[2];
}

inline void vec3toivec3(vec3 a, ivec3 b)
{
    b[0] = (int)a[0];
    b[1] = (int)a[1];
    b[2] = (int)a[2];
}

inline void fvec3toivec3(fvec3 a, ivec3 b)
{
    b[0] = (int)a[0];
    b[1] = (int)a[1];
    b[2] = (int)a[2];
}

inline void vec3tovec2(vec3 a, vec2 b)
{
    b[0] = a[0];
    b[1] = a[1];
}

inline void fmat2tomat2(fmat2 a, mat2 b)
{
    fvec2tovec2(a[0], b[0]);
    fvec2tovec2(a[1], b[1]);
}

inline void imat2tomat2(imat2 a, mat2 b)
{
    ivec2tovec2(a[0], b[0]);
    ivec2tovec2(a[1], b[1]);
}

inline void mat2tofmat2(mat2 a, fmat2 b)
{
    vec2tofvec2(a[0], b[0]);
    vec2tofvec2(a[1], b[1]);
}

inline void imat2tofmat2(imat2 a, fmat2 b)
{
    ivec2tofvec2(a[0], b[0]);
    ivec2tofvec2(a[1], b[1]);
}

inline void mat2toimat2(mat2 a, imat2 b)
{
    vec2toivec2(a[0], b[0]);
    vec2toivec2(a[1], b[1]);
}

inline void fmat2toimat2(fmat2 a, imat2 b)
{
    fvec2toivec2(a[0], b[0]);
    fvec2toivec2(a[1], b[1]);
}

inline void fmat3tomat3(fmat3 a, mat3 b)
{
    fvec3tovec3(a[0], b[0]);
    fvec3tovec3(a[1], b[1]);
    fvec3tovec3(a[2], b[2]);
}

inline void imat3tomat3(imat3 a, mat3 b)
{
    ivec3tovec3(a[0], b[0]);
    ivec3tovec3(a[1], b[1]);
    ivec3tovec3(a[2], b[2]);
}

inline void mat3tofmat3(mat3 a, fmat3 b)
{
    vec3tofvec3(a[0], b[0]);
    vec3tofvec3(a[1], b[1]);
    vec3tofvec3(a[2], b[2]);
}

inline void imat3tofmat3(imat3 a, fmat3 b)
{
    ivec3tofvec3(a[0], b[0]);
    ivec3tofvec3(a[1], b[1]);
    ivec3tofvec3(a[2], b[2]);
}

inline void mat3toimat3(mat3 a, imat3 b)
{
    vec3toivec3(a[0], b[0]);
    vec3toivec3(a[1], b[1]);
    vec3toivec3(a[2], b[2]);
}

inline void fmat3toimat3(fmat3 a, imat3 b)
{
    fvec3toivec3(a[0], b[0]);
    fvec3toivec3(a[1], b[1]);
    fvec3toivec3(a[2], b[2]);
}

inline void mat3tomat2(mat3 a, mat2 b)
{
    b[0][0] = a[0][0];
    b[1][0] = a[1][0];
    b[0][1] = a[0][1];
    b[1][1] = a[1][1];
}

inline void fmat3tovecN(fmat3 a, double *b)
{
    int pos = 0;
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            b[pos] = a[i][j];
            pos++;
        }
    }
}

// Constants
inline void vec2zero(vec2 a)
{
    a[0] = a[1] = 0.0;
}
inline void fvec2zero(fvec2 a)
{
    a[0] = a[1] = 0.0f;
}
inline void ivec2zero(ivec2 a)
{
    a[0] = a[1] = 0;
}

inline void vec3zero(vec3 a)
{
    a[0] = a[1] = a[2] = 0.0;
}
inline void fvec3zero(fvec3 a)
{
    a[0] = a[1] = a[2] = 0.0f;
}
inline void ivec3zero(ivec3 a)
{
    a[0] = a[1] = a[2] = 0;
}

inline void vec4zero(vec4 a)
{
    a[0] = a[1] = a[2] = a[3] = 0.0;
}
inline void fvec4zero(fvec4 a)
{
    a[0] = a[1] = a[2] = a[3] = 0.0f;
}
inline void ivec4zero(ivec4 a)
{
    a[0] = a[1] = a[2] = a[3] = 0;
}

inline void mat2zero(mat2 a)
{
    vec2zero(a[0]);
    vec2zero(a[1]);
}
inline void fmat2zero(fmat2 a)
{
    fvec2zero(a[0]);
    fvec2zero(a[1]);
}
inline void imat2zero(imat2 a)
{
    ivec2zero(a[0]);
    ivec2zero(a[1]);
}

inline void mat3zero(mat3 a)
{
    vec3zero(a[0]);
    vec3zero(a[1]);
    vec3zero(a[2]);
}
inline void fmat3zero(fmat3 a)
{
    fvec3zero(a[0]);
    fvec3zero(a[1]);
    fvec3zero(a[2]);
}
inline void imat3zero(imat3 a)
{
    ivec3zero(a[0]);
    ivec3zero(a[1]);
    ivec3zero(a[2]);
}

inline void mat4zero(mat4 a)
{
    vec4zero(a[0]);
    vec4zero(a[1]);
    vec4zero(a[2]);
    vec4zero(a[3]);
}
inline void fmat4zero(fmat4 a)
{
    fvec4zero(a[0]);
    fvec4zero(a[1]);
    fvec4zero(a[2]);
    fvec4zero(a[3]);
}
inline void imat4zero(imat4 a)
{
    ivec4zero(a[0]);
    ivec4zero(a[1]);
    ivec4zero(a[2]);
    ivec4zero(a[3]);
}

inline void mat2ident(mat2 a)
{
    mat2zero(a);
    a[0][0] = a[1][1] = 1.0;
}
inline void fmat2ident(fmat2 a)
{
    fmat2zero(a);
    a[0][0] = a[1][1] = 1.0f;
}
inline void imat2ident(imat2 a)
{
    imat2zero(a);
    a[0][0] = a[1][1] = 1;
}

inline void mat3ident(mat3 a)
{
    mat3zero(a);
    a[0][0] = a[1][1] = a[2][2] = 1.0;
}
inline void fmat3ident(fmat3 a)
{
    fmat3zero(a);
    a[0][0] = a[1][1] = a[2][2] = 1.0f;
}
inline void imat3ident(imat3 a)
{
    imat3zero(a);
    a[0][0] = a[1][1] = a[2][2] = 1;
}

inline void mat4ident(mat4 a)
{
    mat4zero(a);
    a[0][0] = a[1][1] = a[2][2] = a[3][3] = 1.0;
}
inline void fmat4ident(fmat4 a)
{
    fmat4zero(a);
    a[0][0] = a[1][1] = a[2][2] = a[3][3] = 1.0f;
}
inline void imat4ident(imat4 a)
{
    imat4zero(a);
    a[0][0] = a[1][1] = a[2][2] = a[3][3] = 1;
}

// Predicates
inline bool vec2iszero(vec2 a)
{
    return a[0] == 0.0 && a[1] == 0.0;
}
inline bool fvec2iszero(fvec2 a)
{
    return a[0] == 0.0f && a[1] == 0.0f;
}
inline bool ivec2iszero(ivec2 a)
{
    return a[0] == 0 && a[1] == 0;
}

inline bool vec3iszero(vec3 a)
{
    return a[0] == 0.0 && a[1] == 0.0 && a[2] == 0.0;
}
inline bool fvec3iszero(fvec3 a)
{
    return a[0] == 0.0f && a[1] == 0.0f && a[2] == 0.0f;
}
inline bool ivec3iszero(ivec3 a)
{
    return a[0] == 0 && a[1] == 0 && a[2] == 0;
}

inline bool vec4iszero(vec4 a)
{
    return a[0] == 0.0 && a[1] == 0.0 && a[2] == 0.0 && a[3] == 0.0;
}
inline bool fvec4iszero(fvec4 a)
{
    return a[0] == 0.0f && a[1] == 0.0f && a[2] == 0.0f && a[3] == 0.0f;
}
inline bool ivec4iszero(ivec4 a)
{
    return a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0;
}

inline bool vec2eq(vec2 a, vec2 b)
{
    return a[0] == b[0] && a[1] == b[1];
}
inline bool fvec2eq(fvec2 a, fvec2 b)
{
    return a[0] == b[0] && a[1] == b[1];
}
inline bool ivec2eq(ivec2 a, ivec2 b)
{
    return a[0] == b[0] && a[1] == b[1];
}

inline bool vec3eq(vec3 a, vec3 b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}
inline bool fvec3eq(fvec3 a, fvec3 b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}
inline bool ivec3eq(ivec3 a, ivec3 b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

inline bool vec4eq(vec4 a, vec4 b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}
inline bool fvec4eq(fvec4 a, fvec4 b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}
inline bool ivec4eq(ivec4 a, ivec4 b)
{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

// Determinant
inline double mat2det(mat2 a)
{
    return a[0][0] * a[1][1] - a[0][1] * a[1][0];
}
inline float fmat2det(fmat2 a)
{
    return a[0][0] * a[1][1] - a[0][1] * a[1][0];
}
inline int imat2det(imat2 a)
{
    return a[0][0] * a[1][1] - a[0][1] * a[1][0];
}

inline double vec2det(vec2 a, vec2 b)
{
    return a[0] * b[1] - a[1] * b[0];
}
inline float fvec2det(fvec2 a, fvec2 b)
{
    return a[0] * b[1] - a[1] * b[0];
}
inline int ivec2det(ivec2 a, ivec2 b)
{
    return a[0] * b[1] - a[1] * b[0];
}

inline double mat3det(mat3 a)
{
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) + a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

inline float fmat3det(fmat3 a)
{
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) + a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

inline int imat3det(imat3 a)
{
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) + a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

inline double mat4det(mat4 a)
{
    return a[0][0] * (a[1][1] * (a[2][2] * a[3][3] - a[2][3] * a[3][2]) - a[1][2] * (a[2][1] * a[3][3] - a[2][3] * a[3][1]) + a[1][3] * (a[2][1] * a[3][2] - a[2][2] * a[3][1])) - a[0][1] * (a[1][0] * (a[2][2] * a[3][3] - a[2][3] * a[3][2]) - a[1][2] * (a[2][0] * a[3][3] - a[2][3] * a[3][0]) + a[1][3] * (a[2][0] * a[3][2] - a[2][2] * a[3][0])) + a[0][2] * (a[1][0] * (a[2][1] * a[3][3] - a[2][3] * a[3][1]) - a[1][1] * (a[2][0] * a[3][3] - a[2][3] * a[3][0]) + a[1][3] * (a[2][0] * a[3][1] - a[2][1] * a[3][0])) - a[0][3] * (a[1][0] * (a[2][1] * a[3][2] - a[2][2] * a[3][1]) - a[1][1] * (a[2][0] * a[3][2] - a[2][2] * a[3][0]) + a[1][2] * (a[2][0] * a[3][1] - a[2][1] * a[3][0]));
}

inline double vec3det(vec3 a, vec3 b, vec3 c)
{
    return a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

inline float fvec3det(fvec3 a, fvec3 b, fvec3 c)
{
    return a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

inline int ivec3det(ivec3 a, ivec3 b, ivec3 c)
{
    return a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

// Addition
inline void vec2add(vec2 a, vec2 b, vec2 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
}
inline void fvec2add(fvec2 a, fvec2 b, fvec2 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
}
inline void ivec2add(ivec2 a, ivec2 b, ivec2 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
}

inline void vec3add(vec3 a, vec3 b, vec3 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
}
inline void fvec3add(fvec3 a, fvec3 b, fvec3 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
}
inline void ivec3add(ivec3 a, ivec3 b, ivec3 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
}

inline void mat2add(mat2 a, mat2 b, mat2 c)
{
    vec2add(a[0], b[0], c[0]);
    vec2add(a[1], b[1], c[1]);
}
inline void fmat2add(fmat2 a, fmat2 b, fmat2 c)
{
    fvec2add(a[0], b[0], c[0]);
    fvec2add(a[1], b[1], c[1]);
}
inline void imat2add(imat2 a, imat2 b, imat2 c)
{
    ivec2add(a[0], b[0], c[0]);
    ivec2add(a[1], b[1], c[1]);
}

inline void mat3add(mat3 a, mat3 b, mat3 c)
{
    vec3add(a[0], b[0], c[0]);
    vec3add(a[1], b[1], c[1]);
    vec3add(a[2], b[2], c[2]);
}

inline void fmat3add(fmat3 a, fmat3 b, fmat3 c)
{
    fvec3add(a[0], b[0], c[0]);
    fvec3add(a[1], b[1], c[1]);
    fvec3add(a[2], b[2], c[2]);
}

inline void imat3add(imat3 a, imat3 b, imat3 c)
{
    ivec3add(a[0], b[0], c[0]);
    ivec3add(a[1], b[1], c[1]);
    ivec3add(a[2], b[2], c[2]);
}

// Subtraction
inline void vec2sub(vec2 a, vec2 b, vec2 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
}
inline void fvec2sub(fvec2 a, fvec2 b, fvec2 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
}
inline void ivec2sub(ivec2 a, ivec2 b, ivec2 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
}

inline void vec3sub(vec3 a, vec3 b, vec3 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}
inline void fvec3sub(fvec3 a, fvec3 b, fvec3 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}
inline void ivec3sub(ivec3 a, ivec3 b, ivec3 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}

inline void vec4sub(vec4 a, vec4 b, vec4 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
    c[3] = a[3] - b[3];
}
inline void fvec4sub(fvec4 a, fvec4 b, fvec4 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
    c[3] = a[3] - b[3];
}
inline void ivec4sub(ivec4 a, ivec4 b, ivec4 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
    c[3] = a[3] - b[3];
}

inline void mat2sub(mat2 a, mat2 b, mat2 c)
{
    vec2sub(a[0], b[0], c[0]);
    vec2sub(a[1], b[1], c[1]);
}
inline void fmat2sub(fmat2 a, fmat2 b, fmat2 c)
{
    fvec2sub(a[0], b[0], c[0]);
    fvec2sub(a[1], b[1], c[1]);
}
inline void imat2sub(imat2 a, imat2 b, imat2 c)
{
    ivec2sub(a[0], b[0], c[0]);
    ivec2sub(a[1], b[1], c[1]);
}

inline void mat3sub(mat3 a, mat3 b, mat3 c)
{
    vec3sub(a[0], b[0], c[0]);
    vec3sub(a[1], b[1], c[1]);
    vec3sub(a[2], b[2], c[2]);
}

inline void fmat3sub(fmat3 a, fmat3 b, fmat3 c)
{
    fvec3sub(a[0], b[0], c[0]);
    fvec3sub(a[1], b[1], c[1]);
    fvec3sub(a[2], b[2], c[2]);
}

inline void imat3sub(imat3 a, imat3 b, imat3 c)
{
    ivec3sub(a[0], b[0], c[0]);
    ivec3sub(a[1], b[1], c[1]);
    ivec3sub(a[2], b[2], c[2]);
}

// Averaging
inline void vec3avg(vec3 a, vec3 b, vec3 c)
{
    c[0] = (a[0] + b[0]) * .5;
    c[1] = (a[1] + b[1]) * .5;
    c[2] = (a[2] + b[2]) * .5;
}

inline void fvec3avg(fvec3 a, fvec3 b, fvec3 c)
{
    c[0] = (a[0] + b[0]) * .5;
    c[1] = (a[1] + b[1]) * .5;
    c[2] = (a[2] + b[2]) * .5;
}

inline void vec3avg3(vec3 a, vec3 b, vec3 c, vec3 d)
{
    d[0] = (a[0] + b[0] + c[0]) / 3.;
    d[1] = (a[1] + b[1] + c[1]) / 3.;
    d[2] = (a[2] + b[2] + c[2]) / 3.;
}

inline void fvec3avg3(fvec3 a, fvec3 b, fvec3 c, fvec3 d)
{
    d[0] = (a[0] + b[0] + c[0]) / 3.;
    d[1] = (a[1] + b[1] + c[1]) / 3.;
    d[2] = (a[2] + b[2] + c[2]) / 3.;
}

inline void vec3avg4(vec3 a, vec3 b, vec3 c, vec3 d, vec3 e)
{
    e[0] = (a[0] + b[0] + c[0] + d[0]) * .25;
    e[1] = (a[1] + b[1] + c[1] + d[1]) * .25;
    e[2] = (a[2] + b[2] + c[2] + d[2]) * .25;
}

inline void fvec3avg4(fvec3 a, fvec3 b, fvec3 c, fvec3 d, fvec3 e)
{
    e[0] = (a[0] + b[0] + c[0] + d[0]) * .25;
    e[1] = (a[1] + b[1] + c[1] + d[1]) * .25;
    e[2] = (a[2] + b[2] + c[2] + d[2]) * .25;
}

// Sorting
inline void vec3sortd(vec3 a, vec3 b, vec3 opt = NULL)
{ // sort descending
    // if opt != NULL, also reorders opt

    vec3copy(a, b);
    if (b[0] < b[1])
    {
        double w = b[0];
        b[0] = b[1];
        b[1] = w;
        if (opt)
        {
            double w2 = opt[0];
            opt[0] = opt[1];
            opt[1] = w2;
        }
    }
    if (b[1] < b[2])
    {
        double w = b[1];
        b[1] = b[2];
        b[2] = w;
        if (opt)
        {
            double w2 = opt[1];
            opt[1] = opt[2];
            opt[2] = w2;
        }
    }
    if (b[0] < b[1])
    {
        double w = b[0];
        b[0] = b[1];
        b[1] = w;
        if (opt)
        {
            double w2 = opt[0];
            opt[0] = opt[1];
            opt[1] = w2;
        }
    }
}

inline void vec3sortdAbs(vec3 a, vec3 b, vec3 opt = NULL)
{ // sort descending by absolute value
    // if opt != NULL, also reorders opt

    vec3copy(a, b);
    if (fabs(b[0]) < fabs(b[1]))
    {
        double w = b[0];
        b[0] = b[1];
        b[1] = w;
        if (opt)
        {
            double w2 = opt[0];
            opt[0] = opt[1];
            opt[1] = w2;
        }
    }
    if (fabs(b[1]) < fabs(b[2]))
    {
        double w = b[1];
        b[1] = b[2];
        b[2] = w;
        if (opt)
        {
            double w2 = opt[1];
            opt[1] = opt[2];
            opt[2] = w2;
        }
    }
    if (fabs(b[0]) < fabs(b[1]))
    {
        double w = b[0];
        b[0] = b[1];
        b[1] = w;
        if (opt)
        {
            double w2 = opt[0];
            opt[0] = opt[1];
            opt[1] = w2;
        }
    }
}

// Dot product
inline double vec2dot(vec2 a, vec2 b)
{
    return a[0] * b[0] + a[1] * b[1];
}
inline float fvec2dot(fvec2 a, fvec2 b)
{
    return a[0] * b[0] + a[1] * b[1];
}
inline int ivec2dot(ivec2 a, ivec2 b)
{
    return a[0] * b[0] + a[1] * b[1];
}

inline double vec3dot(vec3 a, vec3 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline float fvec3dot(fvec3 a, fvec3 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline int ivec3dot(ivec3 a, ivec3 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double vec4dot(vec4 a, vec4 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
inline float fvec4dot(fvec4 a, fvec4 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
inline int ivec4dot(ivec4 a, ivec4 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

// Magnitude squared
inline double vec2sqr(vec2 a)
{
    return vec2dot(a, a);
}
inline float fvec2sqr(fvec2 a)
{
    return fvec2dot(a, a);
}

inline double vec3sqr(vec3 a)
{
    return vec3dot(a, a);
}
inline float fvec3sqr(fvec3 a)
{
    return fvec3dot(a, a);
}

inline double vec4sqr(vec4 a)
{
    return vec4dot(a, a);
}
inline float fvec4sqr(fvec4 a)
{
    return fvec4dot(a, a);
}

// Magnitude
inline double vec2mag(vec2 a)
{
    return sqrt(vec2sqr(a));
}
inline float fvec2mag(fvec2 a)
{
    return (float)sqrt(fvec2sqr(a));
}

inline double vec3mag(vec3 a)
{
    return sqrt(vec3sqr(a));
}
inline float fvec3mag(fvec3 a)
{
    return (float)sqrt(fvec3sqr(a));
}

inline double vec4mag(vec4 a)
{
    return sqrt(vec4sqr(a));
}
inline float fvec4mag(fvec4 a)
{
    return (float)sqrt(fvec4sqr(a));
}

// Normalization
inline void vec2nrm(vec2 a, vec2 b)
{
    double l = vec2mag(a);
    if (l == 0)
        l = 1;
    b[0] = a[0] / l;
    b[1] = a[1] / l;
}

inline void fvec2nrm(fvec2 a, fvec2 b)
{
    double l = fvec2mag(a);
    if (l == 0)
        l = 1;
    b[0] = (float)(a[0] / l);
    b[1] = (float)(a[1] / l);
}

inline void vec3nrm(vec3 a, vec3 b)
{
    double l = vec3mag(a);
    if (l == 0)
        l = 1;
    b[0] = a[0] / l;
    b[1] = a[1] / l;
    b[2] = a[2] / l;
}

inline void fvec3nrm(fvec3 a, fvec3 b)
{
    double l = fvec3mag(a);
    if (l == 0)
        l = 1;
    b[0] = (float)(a[0] / l);
    b[1] = (float)(a[1] / l);
    b[2] = (float)(a[2] / l);
}

// Distance
inline double vec2dist(vec2 a, vec2 b)
{
    vec2 c;
    vec2sub(a, b, c);
    return vec2mag(c);
}

inline float fvec2dist(fvec2 a, fvec2 b)
{
    fvec2 c;
    fvec2sub(a, b, c);
    return fvec2mag(c);
}

inline double vec2distSqr(vec2 a, vec2 b)
{
    vec2 c;
    vec2sub(a, b, c);
    return vec2dot(c, c);
}

inline double fvec2distSqr(fvec2 a, fvec2 b)
{
    fvec2 c;
    fvec2sub(a, b, c);
    return fvec2dot(c, c);
}

inline double vec3dist(vec3 a, vec3 b)
{
    vec3 c;
    vec3sub(a, b, c);
    return vec3mag(c);
}

inline float fvec3dist(fvec3 a, fvec3 b)
{
    fvec3 c;
    fvec3sub(a, b, c);
    return fvec3mag(c);
}

inline double fvec3distSqr(fvec3 a, fvec3 b)
{
    fvec3 c;
    fvec3sub(a, b, c);
    return fvec3dot(c, c);
}

inline double vec3distSqr(vec3 a, vec3 b)
{
    vec3 c;
    vec3sub(a, b, c);
    return vec3dot(c, c);
}

inline double vec4dist(vec4 a, vec4 b)
{
    vec4 c;
    vec4sub(a, b, c);
    return vec4mag(c);
}

inline float fvec4dist(fvec4 a, fvec4 b)
{
    fvec4 c;
    fvec4sub(a, b, c);
    return fvec4mag(c);
}

inline double vec4distSqr(vec4 a, vec4 b)
{
    vec4 c;
    vec4sub(a, b, c);
    return vec4dot(c, c);
}

inline double fvec4distSqr(fvec4 a, fvec4 b)
{
    fvec4 c;
    fvec4sub(a, b, c);
    return fvec4dot(c, c);
}

// TODO: sadlo moved from other linalg version, move to another place?
inline void fvec3away(fvec3 a, fvec3 b) /** b not (nearly) parallel to a **/
{
    float ax, ay, az;

    b[0] = b[1] = b[2] = 0;

    ax = a[0] < 0 ? -a[0] : a[0];
    ay = a[1] < 0 ? -a[1] : a[1];
    az = a[2] < 0 ? -a[2] : a[2];

    if (ax < ay)
    {
        if (ax < az)
            b[0] = 1;
        else
            b[2] = 1;
    }
    else
    {
        if (ay < az)
            b[1] = 1;
        else
            b[2] = 1;
    }
}

// Scaling
inline void vec2scal(vec2 a, double b, vec2 c)
{
    c[0] = a[0] * b;
    c[1] = a[1] * b;
}
inline void fvec2scal(fvec2 a, float b, fvec2 c)
{
    c[0] = a[0] * b;
    c[1] = a[1] * b;
}
inline void ivec2scal(ivec2 a, int b, ivec2 c)
{
    c[0] = a[0] * b;
    c[1] = a[1] * b;
}

inline void vec3scal(vec3 a, double b, vec3 c)
{
    c[0] = a[0] * b;
    c[1] = a[1] * b;
    c[2] = a[2] * b;
}
inline void fvec3scal(fvec3 a, float b, fvec3 c)
{
    c[0] = a[0] * b;
    c[1] = a[1] * b;
    c[2] = a[2] * b;
}
inline void ivec3scal(ivec3 a, int b, ivec3 c)
{
    c[0] = a[0] * b;
    c[1] = a[1] * b;
    c[2] = a[2] * b;
}

inline void mat3scal(mat3 a, double b, mat3 c)
{
    vec3scal(a[0], b, c[0]);
    vec3scal(a[1], b, c[1]);
    vec3scal(a[2], b, c[2]);
}

inline void fmat3scal(fmat3 a, float b, fmat3 c)
{
    fvec3scal(a[0], b, c[0]);
    fvec3scal(a[1], b, c[1]);
    fvec3scal(a[2], b, c[2]);
}

inline void imat3scal(imat3 a, int b, imat3 c)
{
    ivec3scal(a[0], b, c[0]);
    ivec3scal(a[1], b, c[1]);
    ivec3scal(a[2], b, c[2]);
}

// Cross product

inline double vec2cross(vec2 a, vec2 b)
{
    return a[0] * b[1] - a[1] * b[0];
}

inline float fvec2cross(fvec2 a, fvec2 b)
{
    return a[0] * b[1] - a[1] * b[0];
}

inline void vec3cross(vec3 a, vec3 b, vec3 c)
{
    vec3 d;
    d[0] = a[1] * b[2] - a[2] * b[1];
    d[1] = a[2] * b[0] - a[0] * b[2];
    d[2] = a[0] * b[1] - a[1] * b[0];
    vec3copy(d, c);
}

inline void fvec3cross(fvec3 a, fvec3 b, fvec3 c)
{
    fvec3 d;
    d[0] = a[1] * b[2] - a[2] * b[1];
    d[1] = a[2] * b[0] - a[0] * b[2];
    d[2] = a[0] * b[1] - a[1] * b[0];
    fvec3copy(d, c);
}

// outer product
inline void vec3outer(vec3 a, vec3 b, mat3 m)
{ // m = a * bT
    vec3 w1, w2, w3;
    vec3scal(b, a[0], w1);
    vec3scal(b, a[1], w2);
    vec3scal(b, a[2], w3);
    mat3setrows(m, w1, w2, w3);
}

// Matrix product
inline void mat2mul(mat2 a, mat2 b, mat2 c)
{
    mat2 d;
    d[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
    d[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
    d[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
    d[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
    mat2copy(d, c);
}

inline void fmat2mul(fmat2 a, fmat2 b, fmat2 c)
{
    fmat2 d;
    d[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
    d[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
    d[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
    d[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
    fmat2copy(d, c);
}

inline void imat2mul(imat2 a, imat2 b, imat2 c)
{
    imat2 d;
    d[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
    d[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
    d[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
    d[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
    imat2copy(d, c);
}

inline void mat3mul(mat3 a, mat3 b, mat3 c)
{
    mat3 d;
    d[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
    d[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
    d[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];
    d[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
    d[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
    d[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];
    d[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
    d[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
    d[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
    mat3copy(d, c);
}

inline void fmat3mul(fmat3 a, fmat3 b, fmat3 c)
{
    fmat3 d;
    d[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
    d[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
    d[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];
    d[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
    d[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
    d[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];
    d[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
    d[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
    d[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
    fmat3copy(d, c);
}

inline void imat3mul(imat3 a, imat3 b, imat3 c)
{
    imat3 d;
    d[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
    d[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
    d[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];
    d[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
    d[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
    d[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];
    d[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
    d[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
    d[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
    imat3copy(d, c);
}

inline void mat4mul(mat4 a, mat4 b, mat4 c)
{
    mat4 d;
    d[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] + a[0][3] * b[3][0];
    d[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] + a[0][3] * b[3][1];
    d[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] + a[0][3] * b[3][2];
    d[0][3] = a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] + a[0][3] * b[3][3];
    d[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] + a[1][3] * b[3][0];
    d[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] + a[1][3] * b[3][1];
    d[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] + a[1][3] * b[3][2];
    d[1][3] = a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] + a[1][3] * b[3][3];
    d[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] + a[2][3] * b[3][0];
    d[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] + a[2][3] * b[3][1];
    d[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] + a[2][3] * b[3][2];
    d[2][3] = a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] + a[2][3] * b[3][3];
    d[3][0] = a[3][0] * b[0][0] + a[3][1] * b[1][0] + a[3][2] * b[2][0] + a[3][3] * b[3][0];
    d[3][1] = a[3][0] * b[0][1] + a[3][1] * b[1][1] + a[3][2] * b[2][1] + a[3][3] * b[3][1];
    d[3][2] = a[3][0] * b[0][2] + a[3][1] * b[1][2] + a[3][2] * b[2][2] + a[3][3] * b[3][2];
    d[3][3] = a[3][0] * b[0][3] + a[3][1] * b[1][3] + a[3][2] * b[2][3] + a[3][3] * b[3][3];
    mat4copy(d, c);
}

inline void mat23MMT(double a[2][3], double b[2][2])
// Multiply a 2x3 matrix with its transpose */
{
    b[0][0] = a[0][0] * a[0][0] + a[0][1] * a[0][1] + a[0][2] * a[0][2];
    b[0][1] = a[0][0] * a[1][0] + a[0][1] * a[1][1] + a[0][2] * a[1][2];
    b[1][0] = a[0][0] * a[1][0] + a[0][1] * a[1][1] + a[0][2] * a[1][2];
    b[1][1] = a[1][0] * a[1][0] + a[1][1] * a[1][1] + a[1][2] * a[1][2];
}

// Matrix-vector product
inline void mat2vec(mat2 a, vec2 b, vec2 c)
{
    vec2 d;
    d[0] = a[0][0] * b[0] + a[0][1] * b[1];
    d[1] = a[1][0] * b[0] + a[1][1] * b[1];
    vec2copy(d, c);
}

inline void fmat2vec(fmat2 a, fvec2 b, fvec2 c)
{
    fvec2 d;
    d[0] = a[0][0] * b[0] + a[0][1] * b[1];
    d[1] = a[1][0] * b[0] + a[1][1] * b[1];
    fvec2copy(d, c);
}

inline void imat2vec(imat2 a, ivec2 b, ivec2 c)
{
    ivec2 d;
    d[0] = a[0][0] * b[0] + a[0][1] * b[1];
    d[1] = a[1][0] * b[0] + a[1][1] * b[1];
    ivec2copy(d, c);
}

inline void mat3vec(mat3 a, vec3 b, vec3 c)
{
    vec3 d;
    d[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2];
    d[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2];
    d[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2];
    vec3copy(d, c);
}

inline void fmat3vec(fmat3 a, fvec3 b, fvec3 c)
{
    fvec3 d;
    d[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2];
    d[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2];
    d[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2];
    fvec3copy(d, c);
}

inline void imat3vec(imat3 a, ivec3 b, ivec3 c)
{
    ivec3 d;
    d[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2];
    d[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2];
    d[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2];
    ivec3copy(d, c);
}

inline void mat23vec(double m[2][3], vec3 a, vec2 b)
// Multiply a 2x3 matrix with a 3-vector
{
    b[0] = m[0][0] * a[0] + m[0][1] * a[1] + m[0][2] * a[2];
    b[1] = m[1][0] * a[0] + m[1][1] * a[1] + m[1][2] * a[2];
}

// Linear interpolation
inline void vec2lerp(vec2 a, vec2 b, double t, vec2 c)
{
    double s = 1.0 - t;
    c[0] = a[0] * s + b[0] * t;
    c[1] = a[1] * s + b[1] * t;
}

inline void fvec2lerp(fvec2 a, fvec2 b, float t, fvec2 c)
{
    float s = 1.0f - t;
    c[0] = a[0] * s + b[0] * t;
    c[1] = a[1] * s + b[1] * t;
}

inline void vec3lerp(vec3 a, vec3 b, double t, vec3 c)
{
    double s = 1.0 - t;
    c[0] = a[0] * s + b[0] * t;
    c[1] = a[1] * s + b[1] * t;
    c[2] = a[2] * s + b[2] * t;
}

inline void fvec3lerp(fvec3 a, fvec3 b, float t, fvec3 c)
{
    float s = 1.0f - t;
    c[0] = a[0] * s + b[0] * t;
    c[1] = a[1] * s + b[1] * t;
    c[2] = a[2] * s + b[2] * t;
}

inline void vec4lerp(vec4 a, vec4 b, double t, vec4 c)
{
    double s = 1.0 - t;
    c[0] = a[0] * s + b[0] * t;
    c[1] = a[1] * s + b[1] * t;
    c[2] = a[2] * s + b[2] * t;
    c[3] = a[3] * s + b[3] * t;
}

inline void mat2lerp(mat2 a, mat2 b, double t, mat2 c)
{
    vec2lerp(a[0], b[0], t, c[0]);
    vec2lerp(a[1], b[1], t, c[1]);
}

inline void fmat2lerp(fmat2 a, fmat2 b, float t, fmat2 c)
{
    fvec2lerp(a[0], b[0], t, c[0]);
    fvec2lerp(a[1], b[1], t, c[1]);
}

inline void mat3lerp(mat3 a, mat3 b, double t, mat3 c)
{
    vec3lerp(a[0], b[0], t, c[0]);
    vec3lerp(a[1], b[1], t, c[1]);
    vec3lerp(a[2], b[2], t, c[2]);
}

inline void fmat3lerp(fmat3 a, fmat3 b, float t, fmat3 c)
{
    fvec3lerp(a[0], b[0], t, c[0]);
    fvec3lerp(a[1], b[1], t, c[1]);
    fvec3lerp(a[2], b[2], t, c[2]);
}

inline void mat4lerp(mat4 a, mat4 b, double t, mat4 c)
{
    vec4lerp(a[0], b[0], t, c[0]);
    vec4lerp(a[1], b[1], t, c[1]);
    vec4lerp(a[2], b[2], t, c[2]);
    vec4lerp(a[3], b[3], t, c[3]);
}

// Other interpolations

inline void mat3bilint(mat3 m00, mat3 m10, mat3 m01, mat3 m11, double s, double t, mat3 m)
//   Do a bilinear interpolation of a MATRIX in a quadrangle
// given the 4 corner values and the two parameters
{
    mat3 tmp;

    mat3zero(m);
    mat3scal(m00, (1 - s) * (1 - t), tmp);
    mat3add(m, tmp, m);
    mat3scal(m10, s * (1 - t), tmp);
    mat3add(m, tmp, m);
    mat3scal(m01, (1 - s) * t, tmp);
    mat3add(m, tmp, m);
    mat3scal(m11, s * t, tmp);
    mat3add(m, tmp, m);
}

inline void vec3bilint(vec3 v00, vec3 v10, vec3 v01, vec3 v11, double s, double t, vec3 v)
// Do a bilinear interpolation of a VECTOR in a quadrangle
// given the 4 corner values and the two parameters
{
    vec3 tmp;

    vec3zero(v);
    vec3scal(v00, (1 - s) * (1 - t), tmp);
    vec3add(v, tmp, v);
    vec3scal(v10, s * (1 - t), tmp);
    vec3add(v, tmp, v);
    vec3scal(v01, (1 - s) * t, tmp);
    vec3add(v, tmp, v);
    vec3scal(v11, s * t, tmp);
    vec3add(v, tmp, v);
}

inline void vec3lint(vec3 v0, vec3 v1, double t, vec3 v)
// Do a linear interpolation given the 2 corner values and the variable
{
    vec3 tmp;

    vec3zero(v);
    vec3scal(v0, (1 - t), tmp);
    vec3add(v, tmp, v);
    vec3scal(v1, t, tmp);
    vec3add(v, tmp, v);
}

inline void fvec3lerp3(fvec3 a, fvec3 b, fvec3 c, float s, float t, fvec3 out)
// Do a linear interpolation in a triangle
// given the corner values and the local coords
{
    out[0] = (1 - s - t) * a[0] + s * b[0] + t * c[0];
    out[1] = (1 - s - t) * a[1] + s * b[1] + t * c[1];
    out[2] = (1 - s - t) * a[2] + s * b[2] + t * c[2];
}

inline void vec3lerp3(vec3 a, vec3 b, vec3 c, double s, double t, vec3 out)
// Do a linear interpolation in a triangle
// given the corner values and the local coords
{
    out[0] = (1 - s - t) * a[0] + s * b[0] + t * c[0];
    out[1] = (1 - s - t) * a[1] + s * b[1] + t * c[1];
    out[2] = (1 - s - t) * a[2] + s * b[2] + t * c[2];
}

inline void mat3lerp3(mat3 a, mat3 b, mat3 c, double s, double t, mat3 out)
// Do a linear interpolation in a triangle
// given the corner values and the local coords
{
    for (int i = 0; i < 3; i++)
    {
        out[i][0] = (1 - s - t) * a[i][0] + s * b[i][0] + t * c[i][0];
        out[i][1] = (1 - s - t) * a[i][1] + s * b[i][1] + t * c[i][1];
        out[i][2] = (1 - s - t) * a[i][2] + s * b[i][2] + t * c[i][2];
    }
}

// Matrix transpose
inline void mat2trp(mat2 a, mat2 b)
{
    if (a != b)
        mat2copy(a, b);
    double x;
    x = b[0][1];
    b[0][1] = b[1][0];
    b[1][0] = x;
}

inline void fmat2trp(fmat2 a, fmat2 b)
{
    if (a != b)
        fmat2copy(a, b);
    float x;
    x = b[0][1];
    b[0][1] = b[1][0];
    b[1][0] = x;
}

inline void mat3trp(mat3 a, mat3 b)
{
    if (a != b)
        mat3copy(a, b);
    double x;
    x = b[0][1];
    b[0][1] = b[1][0];
    b[1][0] = x;
    x = b[0][2];
    b[0][2] = b[2][0];
    b[2][0] = x;
    x = b[1][2];
    b[1][2] = b[2][1];
    b[2][1] = x;
}

inline void fmat3trp(fmat3 a, fmat3 b)
{
    if (a != b)
        fmat3copy(a, b);
    float x;
    x = b[0][1];
    b[0][1] = b[1][0];
    b[1][0] = x;
    x = b[0][2];
    b[0][2] = b[2][0];
    b[2][0] = x;
    x = b[1][2];
    b[1][2] = b[2][1];
    b[2][1] = x;
}

inline void mat4trp(mat4 a, mat4 b)
{
    if (a != b)
        mat4copy(a, b);
    double x;
    x = b[0][1];
    b[0][1] = b[1][0];
    b[1][0] = x;
    x = b[0][2];
    b[0][2] = b[2][0];
    b[2][0] = x;
    x = b[0][3];
    b[0][3] = b[3][0];
    b[3][0] = x;
    x = b[1][2];
    b[1][2] = b[2][1];
    b[2][1] = x;
    x = b[1][3];
    b[1][3] = b[3][1];
    b[3][1] = x;
    x = b[2][3];
    b[2][3] = b[3][2];
    b[3][2] = x;
}

// Matrix symmetric and antisymmetric part
inline void mat3symm(mat3 m, mat3 s)
{
    mat3 mT;
    mat3trp(m, mT);
    mat3add(m, mT, s);
    mat3scal(s, 0.5, s);
}

inline void fmat3symm(fmat3 m, fmat3 s)
{
    fmat3 mT;
    fmat3trp(m, mT);
    fmat3add(m, mT, s);
    fmat3scal(s, 0.5, s);
}

inline void mat3asymm(mat3 m, mat3 a)
{
    mat3 mT;
    mat3trp(m, mT);
    mat3sub(m, mT, a);
    mat3scal(a, 0.5, a);
}

inline void fmat3asymm(fmat3 m, fmat3 a)
{
    fmat3 mT;
    fmat3trp(m, mT);
    fmat3sub(m, mT, a);
    fmat3scal(a, 0.5, a);
}

// Basic matrix operations
inline double mat3trace(mat3 m)
{
    return m[0][0] + m[1][1] + m[2][2];
}

inline void mat3MTM(mat3 m, mat3 mTm)
// Multiply transpose of matrix with matrix
{
    mat3 mT;
    mat3trp(m, mT);
    mat3mul(mT, m, mTm);
}

// Matrix inversion
inline void mat2inv(mat2 a, mat2 b)
{
    double d = mat2det(a);
    if (d == 0.0)
    {
        fprintf(stderr, "Inverting singular matrix!\n");
        d = 1.0;
    }
    b[0][0] = a[1][1] / d;
    b[1][0] = -a[1][0] / d;
    b[0][1] = -a[0][1] / d;
    b[1][1] = a[0][0] / d;
}

inline void fmat2inv(fmat2 a, fmat2 b)
{
    float d = fmat2det(a);
    if (d == 0.0f)
    {
        fprintf(stderr, "Inverting singular matrix!\n");
        d = 1.0f;
    }
    b[0][0] = a[1][1] / d;
    b[1][0] = -a[1][0] / d;
    b[0][1] = -a[0][1] / d;
    b[1][1] = a[0][0] / d;
}

inline void mat3inv(mat3 a, mat3 b)
{
    double d = mat3det(a);
    if (d == 0.0)
    {
        fprintf(stderr, "Inverting singular matrix!\n");
        d = 1.0;
    }
    b[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d;
    b[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d;
    b[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d;
    b[0][1] = (a[2][1] * a[0][2] - a[2][2] * a[0][1]) / d;
    b[1][1] = (a[2][2] * a[0][0] - a[2][0] * a[0][2]) / d;
    b[2][1] = (a[2][0] * a[0][1] - a[2][1] * a[0][0]) / d;
    b[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d;
    b[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d;
    b[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d;
}

inline void fmat3inv(fmat3 a, fmat3 b)
{
    float d = fmat3det(a);
    if (d == 0.0f)
    {
        fprintf(stderr, "Inverting singular matrix!\n");
        d = 1.0f;
    }
    b[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / d;
    b[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / d;
    b[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / d;
    b[0][1] = (a[2][1] * a[0][2] - a[2][2] * a[0][1]) / d;
    b[1][1] = (a[2][2] * a[0][0] - a[2][0] * a[0][2]) / d;
    b[2][1] = (a[2][0] * a[0][1] - a[2][1] * a[0][0]) / d;
    b[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / d;
    b[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / d;
    b[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / d;
}

inline void mat3invdet(mat3 a, double d, mat3 b)
// b = invert(a) given non-zero det d
{
    double dinv = 1.0 / d;

    b[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * dinv;
    b[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * dinv;
    b[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * dinv;
    b[0][1] = (a[2][1] * a[0][2] - a[2][2] * a[0][1]) * dinv;
    b[1][1] = (a[2][2] * a[0][0] - a[2][0] * a[0][2]) * dinv;
    b[2][1] = (a[2][0] * a[0][1] - a[2][1] * a[0][0]) * dinv;
    b[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * dinv;
    b[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * dinv;
    b[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * dinv;
}

inline void mat2invdet(mat2 a, double d, mat2 b)
// b = invert(a) given non-zero det d
{
    double dinv = 1.0 / d;

    b[0][0] = a[1][1] * dinv;
    b[1][0] = -a[1][0] * dinv;
    b[0][1] = -a[0][1] * dinv;
    b[1][1] = a[0][0] * dinv;
}

inline void fmat3invdet(fmat3 a, float d, fmat3 b)
// b = invert(a) given non-zero det d
{
    double dinv = 1.0 / d;

    b[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * dinv;
    b[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * dinv;
    b[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * dinv;
    b[0][1] = (a[2][1] * a[0][2] - a[2][2] * a[0][1]) * dinv;
    b[1][1] = (a[2][2] * a[0][0] - a[2][0] * a[0][2]) * dinv;
    b[2][1] = (a[2][0] * a[0][1] - a[2][1] * a[0][0]) * dinv;
    b[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * dinv;
    b[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * dinv;
    b[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * dinv;
}

// Compute any unit vector that is orthogonal to a given 3D vector

inline void vec3ortho(vec3 a, vec3 b)
{
    vec3zero(b); // Initialize

    // Find shortest projection on standard axes
    double c0 = a[0] * a[0];
    double c1 = a[1] * a[1];
    double c2 = a[2] * a[2];

    if (c0 < c1)
    {
        if (c0 < c2)
            b[0] = 1;
        else
            b[2] = 1;
    }
    else
    {
        if (c1 < c2)
            b[1] = 1;
        else
            b[2] = 1;
    }

    vec3 n;
    vec3cross(a, b, n);
    vec3cross(n, a, b);
    vec3nrm(b, b);
}

inline void fvec3ortho(fvec3 a, fvec3 b)
{
    fvec3zero(b); // Initialize

    // Find shortest projection on standard axes
    double c0 = a[0] * a[0];
    double c1 = a[1] * a[1];
    double c2 = a[2] * a[2];

    if (c0 < c1)
    {
        if (c0 < c2)
            b[0] = 1;
        else
            b[2] = 1;
    }
    else
    {
        if (c1 < c2)
            b[1] = 1;
        else
            b[2] = 1;
    }

    fvec3 n;
    fvec3cross(a, b, n);
    fvec3cross(n, a, b);
    fvec3nrm(b, b);
}

// Operations using special 4x4 matrices for rotation & translation

inline void mat4combine(mat3 rot, vec3 trl, mat4 a)
{
    a[0][0] = rot[0][0];
    a[0][1] = rot[0][1];
    a[0][2] = rot[0][2];
    a[0][3] = trl[0];
    a[1][0] = rot[1][0];
    a[1][1] = rot[1][1];
    a[1][2] = rot[1][2];
    a[1][3] = trl[1];
    a[2][0] = rot[2][0];
    a[2][1] = rot[2][1];
    a[2][2] = rot[2][2];
    a[2][3] = trl[2];
    a[3][0] = 0;
    a[3][1] = 0;
    a[3][2] = 0;
    a[3][3] = 1;
}

inline void mat4split(mat4 a, mat3 rot, vec3 trl)
{
    rot[0][0] = a[0][0];
    rot[0][1] = a[0][1];
    rot[0][2] = a[0][2];
    trl[0] = a[0][3];
    rot[1][0] = a[1][0];
    rot[1][1] = a[1][1];
    rot[1][2] = a[1][2];
    trl[1] = a[1][3];
    rot[2][0] = a[2][0];
    rot[2][1] = a[2][1];
    rot[2][2] = a[2][2];
    trl[2] = a[2][3];
}

inline void mat4vec3(mat4 a, vec3 b, vec3 c)
{
    vec3 d;
    d[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2] + a[0][3];
    d[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2] + a[1][3];
    d[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2] + a[2][3];
    vec3copy(d, c);
}

inline void mat4vec(mat4 a, vec4 b, vec4 c)
{
    vec4 d;
    d[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2] + a[0][3] * b[3];
    d[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2] + a[1][3] * b[3];
    d[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2] + a[2][3] * b[3];
    d[3] = a[3][0] * b[0] + a[3][1] * b[1] + a[3][2] * b[2] + a[3][3] * b[3];
    vec4copy(d, c);
}

// Inverse transformation (rotation & translation)
inline void mat4invRt(mat4 a, mat4 b)
{
    mat3 r, r1;
    vec3 t, t1;
    mat4split(a, r, t);
    mat3trp(r, r1);
    mat3vec(r1, t, t1);
    vec3scal(t1, -1, t1);
    mat4combine(r1, t1, b);
}

inline void mat3toRPY(mat3 rot, double &roll, double &pitch, double &yaw)
{
    // roll matrix:    [    1       0       0    ]
    //                 [    0     cos(r) -sin(r) ]
    //                 [    0     sin(r)  cos(r) ]

    // pitch matrix:   [  cos(p)    0     sin(p) ]
    //                 [    0       1       0    ]
    //                 [ -sin(p)    0     cos(p) ]

    // yaw matrix:     [  cos(w) -sin(w)    0    ]
    //                 [  sin(w)  cos(w)    0    ]
    //                 [    0       0       1    ]

    // yaw*pitch*roll: [ cos(p)*cos(w)        #              #       ]
    //                 [ cos(p)*sin(w)        #              #       ]
    //                 [    -sin(p)     cos(p)*sin(r)  cos(p)*cos(r) ]

    // special case pitch == +- Pi/2:  We can set yaw=0
    //
    //     pitch*roll: [   0         #        #    ]
    //                 [   0       cos(r)  -sin(r) ]
    //                 [ -sin(p)     0        0    ]

    pitch = asin(-rot[2][0]); // -pi/2 < pitch <= pi/2 (!)

    if (rot[2][1] != 0 || rot[2][2] != 0)
    { // normal case
        roll = atan2(rot[2][1], rot[2][2]); // cos(pitch) >= 0 (!)
        yaw = atan2(rot[1][0], rot[0][0]);
    }
    else
    { // special case
        roll = atan2(-rot[1][2], rot[1][1]);
        yaw = 0;
    }
}

// convert rotation matrix to rotation vector
// source: wikipedia (rotation representation)
inline void mat3toRotVect(mat3 m, vec3 v)
{
    // TODO: handle multiples of PI !! ###
    // TODO: alternative: by eigenvalues of m
    double theta = acos((m[0][0] + m[1][1] + m[2][2] - 1) / 2);
    double twoSinTheta = 2 * sin(theta);
    v[0] = (m[2][1] - m[1][2]) / twoSinTheta;
    v[1] = (m[0][2] - m[2][0]) / twoSinTheta;
    v[2] = (m[1][0] - m[0][1]) / twoSinTheta;

    vec3nrm(v, v);
    vec3scal(v, theta, v);
}

// convert rotation vector to rotation matrix
// source: wikipedia (rotation representation)
inline void rotVectTomat3(vec3 v, mat3 m)
{
    // A = I * cos(theta) + (1 - cos(theta)) v * vT - eps * sin(theta)
    // eps is "kind of vorticity tensor" build from v

    double theta = vec3mag(v);
    vec3nrm(v, v);
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);

    mat3 i;
    mat3ident(i);
    mat3scal(i, cosTheta, i);

    mat3 vv;
    vec3outer(v, v, vv);
    mat3scal(vv, 1 - cosTheta, vv);

    mat3 eps;
    mat3zero(eps);
    eps[1][0] = v[2];
    eps[2][0] = -v[1];
    eps[2][1] = v[0];
    eps[0][1] = -v[2];
    eps[0][2] = v[1];
    eps[1][2] = -v[0];
    mat3scal(eps, -sinTheta, eps);

    mat3add(i, vv, m);
    mat3add(m, eps, m);
}

inline void mat3orthonormalize(mat3 a)
{
    vec3nrm(a[0], a[0]);
    vec3cross(a[0], a[1], a[2]);
    vec3nrm(a[2], a[2]);
    vec3cross(a[2], a[0], a[1]);
}

inline void mat4orthonormalize(mat4 a) // for rot/trl matrix only!
{
    mat3 rot;
    vec3 trl;
    mat4split(a, rot, trl);
    mat3orthonormalize(rot);
    mat4combine(rot, trl, a);
}

inline bool intersectRaySphere(vec3 p0, vec3 p1, vec3 center, double rad, double *t0, double *t1)
{
    // Solve (p0 + t*(p1-p0) - center)^2 = rad^2
    vec3 v, w;
    vec3sub(p1, p0, v);
    vec3sub(p0, center, w);
    double vv = vec3dot(v, v);
    double vw = vec3dot(v, w);
    double ww = vec3dot(w, w);

    double radicand = vw * vw - vv * (ww - rad * rad);
    if (radicand < 0)
        return false; // No intersection

    *t0 = (-vw - sqrt(radicand)) / vv;
    *t1 = (-vw + sqrt(radicand)) / vv;

    return true;
}

inline void checkOrthonormality(mat4 m, FILE *fp)
{
    mat3 r;
    vec3 t;
    mat4split(m, r, t);
    mat3 rt, prod;
    mat3trp(r, rt);
    mat3mul(r, rt, prod);
    mat3dump(prod, fp);
}

inline int vec2squareroots(vec2 a, vec2 r)
/*
 *	Solves equation
 *	    1 * x^2 + a[1]*x + a[0] = 0
 *
 *	On output, 
 *	    r[0], r[1] or
 *	    r[0] +- i*r[1] are the roots 
 *	   
 *	returns number of real solutions
 */
{
    double discrim, root;

    discrim = a[1] * a[1] - 4 * a[0];

    if (discrim >= 0)
    {
        root = sqrt(discrim);
        r[0] = (-a[1] - root) / 2.0;
        r[1] = (-a[1] + root) / 2.0;
        return (2);
    }
    else
    {
        root = sqrt(-discrim);
        r[0] = -a[1] / 2.0;
        r[1] = root / 2.0;
        return (0);
    }
}

// Solving equations

inline void fvec3solve(fmat3 a, fvec3 b, fvec3 x)
// Solve a linear 3x3 system
{
    float det = fmat3det(a);
    if (det == 0)
        fvec3zero(x);
    else
    {
        fmat3 a1;
        fmat3invdet(a, det, a1);
        fmat3vec(a1, b, x);
    }
}

inline void vec3solve(mat3 a, vec3 b, vec3 x)
// Solve a linear 3x3 system
{
    double det = mat3det(a);
    if (det == 0)
        vec3zero(x);
    else
    {
        mat3 a1;
        mat3invdet(a, det, a1);
        mat3vec(a1, b, x);
    }
}

inline int vec3cubicroots(vec3 a, vec3 r, bool forceReal = false)
//  Cubic equation (multiple solutions are returned several times)
//
//	Solves equation
//	    1 * x^3 + a[2]*x^2 + a[1]*x + a[0] = 0
//
//	On output,
//	    r[0], r[1], r[2], or
//	    r[0], r[1] +- i*r[2] are the roots
//
//	returns number of real solutions

{
    // Eliminate quadratic term by substituting
    // x = y - a[2] / 3

    double c1 = a[1] - a[2] * a[2] / 3.;
    double c0 = a[0] - a[1] * a[2] / 3. + 2. / 27. * a[2] * a[2] * a[2];

    // Make cubic coefficient 4 and linear coefficient +- 3
    // by substituting y = z*k and multiplying with 4/k^3

    if (c1 == 0)
    {
        if (c0 == 0)
            r[0] = 0;
        else if (c0 > 0)
            r[0] = -pow(c0, 1. / 3.);
        else
            r[0] = pow(-c0, 1. / 3.);
    }
    else
    {
        bool negc1 = c1 < 0;
        double absc1 = negc1 ? -c1 : c1;

        double k = sqrt(4. / 3. * absc1);

        double d0 = c0 * 4. / (k * k * k);

        // Find the first solution

        if (negc1)
        {
            if (d0 > 1)
                r[0] = -cosh(acosh(d0) / 3);
            else if (d0 > -1)
                r[0] = -cos(acos(d0) / 3);
            else
                r[0] = cosh(acosh(-d0) / 3);
        }
        else
        {
            r[0] = -sinh(asinh(d0) / 3);
        }

        // Transform back
        r[0] *= k;
    }
    r[0] -= a[2] / 3;

    // Other two solutions
    double p = r[0] + a[2];
    double q = r[0] * p + a[1];

    double discrim = p * p - 4 * q;
    if (forceReal && discrim < 0.0)
        discrim = 0.0;

    if (discrim >= 0)
    {
        double root = sqrt(discrim);
        r[1] = (-p - root) / 2.;
        r[2] = (-p + root) / 2.;
        return 3;
    }
    else
    {
        double root = sqrt(-discrim);
        r[1] = -p / 2;
        r[2] = root / 2.;
        return 1;
    }
}

// Eigenvalues and related things

inline void mat2invariants(mat2 m, vec2 pqr)
{
    // invariant0 = det(M)
    pqr[0] = mat2det(m);

    // invariant1 = -trace M
    pqr[1] = -(m[0][0] + m[1][1]);
}

inline void mat3invariants(mat3 m, vec3 pqr)
{
    // invariant0 = -det(M)
    pqr[0] = -mat3det(m);

    // invariant1 = det2(M#0) + det2(M#1) + det2(M#2)
    pqr[1] = m[1][1] * m[2][2] - m[1][2] * m[2][1]
             + m[2][2] * m[0][0] - m[2][0] * m[0][2]
             + m[0][0] * m[1][1] - m[0][1] * m[1][0];

    // invariant2 = -trace M
    pqr[2] = -(m[0][0] + m[1][1] + m[2][2]);
}

#ifdef BUGGY_AND_PROBABLY_NEVER_USED
inline bool mat3hasComplexEigenPQR(mat3 /*m TODO: remove? */, vec3 pqr)
{
    //double q = pqr[1] / 3 - pqr[0]*pqr[0] / 9;
    //double r = (pqr[1]*pqr[0] - 3*pqr[2]) / 6 - pqr[0]*pqr[0]*pqr[0] / 27;
    //return (q*q*q + r*r > 0.0);

    // This is according to maple:
    double p = pqr[0];
    double q = pqr[1];
    double r = pqr[2];
    double D = p * p * q * q + 18 * p * q * r - 4 * q * q * q - 4 * p * p * p * r - 27 * r * r;
    return D < 0;
}

inline bool mat3hasComplexEigen(mat3 m)
{
    vec3 pqr;
    mat3invariants(m, pqr);
    return (mat3hasComplexEigenPQR(m, pqr));
}
#endif

inline int mat2eigenvalues(mat2 m, vec2 lambda)
{
    vec2 pqr;

    mat2invariants(m, pqr);

    return (vec2squareroots(pqr, lambda));
}

inline int mat3eigenvalues(mat3 m, vec3 lambda)
// calculate eigenvalues in lambda, return number of real eigenvalues.
// either returnval==1, lambda[0]=real ev, l[1] real part+-l[2] imag part
// or     returnval==3, lambda[0-2] = eigenvalues
{
    vec3 pqr;
    mat3invariants(m, pqr);

    // force real solutions for symmetric matrices
    bool forceReal = false;
    if (m[1][0] == m[0][1] && m[2][0] == m[0][2] && m[2][1] == m[1][2])
        forceReal = true;

    return (vec3cubicroots(pqr, lambda, forceReal));
}

inline bool mat2realEigenvector(mat2 m, double lambda, vec2 ev)
// calculates eigenvector corresponding to real lambda and returns true if ok
{
    mat2 reduced; // matrix minus lambda I

    mat2copy(m, reduced);

    reduced[0][0] -= lambda;
    reduced[1][1] -= lambda;

    if (vec2mag(reduced[1]) > vec2mag(reduced[0]))
    {
        ev[0] = reduced[1][1];
        ev[1] = -reduced[1][0];
    }
    else
    {
        ev[0] = reduced[0][1];
        ev[1] = -reduced[0][0];
    }

    if (vec2mag(ev) == 0)
        return false;

    vec2nrm(ev, ev);
    return true;
}

inline bool mat3realEigenvector(mat3 m, double lambda, vec3 ev)
// calculates eigenvector corresponding to real lambda and returns true if ok
{
    mat3 reduced; // matrix minus lambda I
    mat3 cross;
    vec3 sqr;

    mat3copy(m, reduced);

    reduced[0][0] -= lambda;
    reduced[1][1] -= lambda;
    reduced[2][2] -= lambda;

    vec3cross(reduced[1], reduced[2], cross[0]);
    vec3cross(reduced[2], reduced[0], cross[1]);
    vec3cross(reduced[0], reduced[1], cross[2]);

    sqr[0] = vec3sqr(cross[0]);
    sqr[1] = vec3sqr(cross[1]);
    sqr[2] = vec3sqr(cross[2]);

    // use largest cross product to calculate eigenvector
    int best;
    // ### TODO: divide e.g. sqr[0] by |reduced[1]|^2 * |reduced[2]|^2
    if (sqr[1] > sqr[0])
    {
        if (sqr[2] > sqr[1])
            best = 2;
        else
            best = 1;
    }
    else
    {
        if (sqr[2] > sqr[0])
            best = 2;
        else
            best = 0;
    }

    double len = sqrt(sqr[best]);

    if (len > 0)
    {
        ev[0] = cross[best][0] / len;
        ev[1] = cross[best][1] / len;
        ev[2] = cross[best][2] / len;
        return true; // result ok
    }
    else
    {
        return false; // result not ok: multiple eigenvalue, probably
    }
}

inline bool mat3realOrthogonalEigenvector(mat3 m, vec3 lambdas, int idx, vec3 ev)
// calculates eigenvector corresponding to real lambdas[idx] and returns true if ok
{
    bool res;

    // get eigenvector of largest eigenvalue
    vec3 lambdasSorted;
    vec3 indices = { 0, 1, 2 };
    vec3sortd(lambdas, lambdasSorted, indices);
    vec3 maxEV;
    res = mat3realEigenvector(m, lambdasSorted[0], maxEV);

    // get index of desired eigenvalue in sorted eigenvalues
    int idxSorted;
    if (indices[0] == idx)
        idxSorted = 0;
    else if (indices[1] == idx)
        idxSorted = 1;
    else
        idxSorted = 2;

    // done if desired was largest
    if (idxSorted == 0)
    {
        vec3copy(maxEV, ev);
        return res;
    }

    // get eigenvector of intermediate eigenvalue
    double lambda = lambdasSorted[1];
    mat3 reduced; // matrix minus lambda I
    mat3 cross;
    vec3 sqrr, sqrc;

    mat3copy(m, reduced);

    reduced[0][0] -= lambda;
    reduced[1][1] -= lambda;
    reduced[2][2] -= lambda;

    vec3cross(maxEV, reduced[0], cross[0]);
    vec3cross(maxEV, reduced[1], cross[1]);
    vec3cross(maxEV, reduced[2], cross[2]);

    sqrr[0] = vec3sqr(reduced[0]);
    sqrr[1] = vec3sqr(reduced[1]);
    sqrr[2] = vec3sqr(reduced[2]);

    sqrc[0] = vec3sqr(cross[0]);
    sqrc[1] = vec3sqr(cross[1]);
    sqrc[2] = vec3sqr(cross[2]);

    // use largest cross product to calculate eigenvector
    int best;
    if (sqrc[1] * sqrr[0] > sqrc[0] * sqrr[1])
    {
        if (sqrc[2] * sqrr[1] > sqrc[1] * sqrr[2])
            best = 2;
        else
            best = 1;
    }
    else
    {
        if (sqrc[2] * sqrr[0] > sqrc[0] * sqrr[2])
            best = 2;
        else
            best = 0;
    }

    double len = sqrt(sqrc[best]);

    if (len > 0)
    {
        if (idxSorted == 1)
        {
            ev[0] = cross[best][0] / len;
            ev[1] = cross[best][1] / len;
            ev[2] = cross[best][2] / len;
        }
        else
        {
            vec3 ev2;
            ev2[0] = cross[best][0] / len;
            ev2[1] = cross[best][1] / len;
            ev2[2] = cross[best][2] / len;
            vec3cross(maxEV, ev2, ev);
        }
        return true; // result ok
    }
    else
    {
        return false; // result not ok: multiple eigenvalue, probably
    }
}

inline bool mat3complexEigenplane(mat3 A, double mu, double nu, vec3 nml)
// Calculates plane orthogonal to plane defined by real and imaginary parts of
// eigenvectors (x +- I y) corresponding to eigenvector (mu + I nu).
// Returns true if ok.
// Method: A x = mu x - nu y   <---> Bx = -y    for B = (A - mu Id)/nu
//         A y = nu x + mu y   <---> By = x
//                              ---> -B^2 x = x
{
    mat3 I, B, M;
    mat3ident(I);
    mat3scal(I, mu, B);
    mat3sub(A, B, B);
    mat3scal(B, 1. / nu, B);
    mat3mul(B, B, M);
    mat3add(I, M, M);

#ifdef DEBUG
    // Check that M is of rank 1
    vec3 c0, c1, c2;
    vec3cross(M[0], M[1], c2);
    vec3cross(M[1], M[2], c0);
    vec3cross(M[2], M[0], c1);
    printf("cross products: %15.10f %15.10f %15.10f\n", vec3mag(c0), vec3mag(c1), vec3mag(c2));
#endif

    // Choose largest row
    double m0 = vec3mag(M[0]);
    double m1 = vec3mag(M[1]);
    double m2 = vec3mag(M[2]);

    if (m1 > m0)
    {
        m0 = m1;
        vec3copy(M[1], M[0]);
    }
    if (m2 > m0)
    {
        m0 = m2;
        vec3copy(M[2], M[0]);
    }

    vec3nrm(M[0], nml);
    return true;
}

inline int mat3realEigen(mat3 m, vec3 lambda, mat3 evec, bool evecOk[3])
// Combines mat3eigenvalues and 1 or 3 calls to mat3realEigenvector
{
    vec3 pqr;
    mat3invariants(m, pqr);

    // calc eigenvalues
    int nsol = vec3cubicroots(pqr, lambda);

    // calc real eigenvectors
    for (int i = 0; i < nsol; i++)
    {
        evecOk[i] = mat3realEigenvector(m, lambda[i], evec[i]);
    }
    return nsol;
}

inline void mat3omega(mat3 a, vec3 omega)
// Antisymmetric part of a matrix, (curl from gradient tensor)
{
    omega[0] = a[1][2] - a[2][1];
    omega[1] = a[2][0] - a[0][2];
    omega[2] = a[0][1] - a[1][0];
}

inline void fmat3omega(fmat3 a, fvec3 omega)
// Antisymmetric part of a matrix, (curl from gradient tensor)
{
    omega[0] = a[1][2] - a[2][1];
    omega[1] = a[2][0] - a[0][2];
    omega[2] = a[0][1] - a[1][0];
}

// Matrix norm
inline double mat3magFrobeniusSqr(mat3 m)
{
    mat3 mTm;
    mat3MTM(m, mTm);
    return mat3trace(mTm);
}

// also called Euclidean norm
inline double mat3magFrobenius(mat3 m)
{
    return sqrt(mat3magFrobeniusSqr(m));
}

inline double mat3magSpectral(mat3 m)
{
    mat3 mTm;
    mat3MTM(m, mTm);
    vec3 lambda;
    // FIX 2007-08-25: force symmetric
    mat3symm(mTm, mTm);
    mat3eigenvalues(mTm, lambda);
    // ATA symmetric -> real eigenvalues
    return sqrt(linalg_max(lambda[0], linalg_max(lambda[1], lambda[2])));
}

// Square root of diagonalizable matrix
inline bool mat3sqrt(mat3 m, mat3 r)
{
    mat3zero(r);

    // get eigenvalues
    vec3 eigenvals;
    if (mat3eigenvalues(m, eigenvals) != 3)
    {
        return false;
    }

    // get eigenvectors
    vec3 eigenvects[3];
    for (int ev = 0; ev < 3; ev++)
    {
        if (!mat3realOrthogonalEigenvector(m, eigenvals, ev, eigenvects[ev]))
        {
            return false;
        }
    }

    // get diagonalization matrix v and its inverse
    mat3 v, vi;
    mat3setcols(v, eigenvects[0], eigenvects[1], eigenvects[2]);
    mat3inv(v, vi);

    // construct square root of diagonal matrix
    mat3 dr;
    mat3zero(dr);
    dr[0][0] = sqrt(eigenvals[0]);
    dr[1][1] = sqrt(eigenvals[1]);
    dr[2][2] = sqrt(eigenvals[2]);

    // transform back
    mat3mul(v, dr, r);
    mat3mul(r, vi, r);

    return true;
}

#endif // __LINALG_H__
