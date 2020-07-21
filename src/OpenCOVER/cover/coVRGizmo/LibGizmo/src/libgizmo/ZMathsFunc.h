///////////////////////////////////////////////////////////////////////////////////////////////////
// LibGizmo
// File Name : 
// Creation : 10/01/2012
// Author : Cedric Guillemet
// Description : LibGizmo
//
///Copyright (C) 2012 Cedric Guillemet
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
//of the Software, and to permit persons to whom the Software is furnished to do
///so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
///FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// 


#ifndef ZMATHSFUNCS_H
#define ZMATHSFUNCS_H

// Includes ///////////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdlib.h>
#include "ZBaseDefs.h"

// Constants //////////////////////////////////////////////////////////////////////////////////////

const float ZPI    =  3.14159265358979323846f;

const float PI_MUL_2 =  6.28318530717958647692f;
const float PI_DIV_2 =  1.57079632679489655800f;
const float PI_DIV_4 =  0.78539816339744827900f;
const float INV_PI   =  0.31830988618379069122f;
const float DEGTORAD =  0.01745329251994329547f;
const float RADTODEG = 57.29577951308232286465f;
const float SQRT2    =  1.41421356237309504880f;
const float SQRT3    =  1.73205080756887729352f;

#define Rad2Deg(x)		((x)*180.f*INV_PI)
#define Deg2Rad(x)		((x)*ZPI/180.f)


inline bool MathFloatIsVeryClose(float flt1, float flt2)
{
    return (fabsf(flt1-flt2)<0.001f);
}

#define PSwap(a, b) do {\
    char c[sizeof(a)]; \
    memcpy((void *)&c, (void *)&a, sizeof(c)); \
    memcpy((void *)&a, (void *)&b, sizeof(a)); \
    memcpy((void *)&b, (void *)&c, sizeof(b)); \
} while (0)

#define PMax(a,b) ((a>b)?a:b)
#define PMin(a,b) ((a<b)?a:b)

#define LERP(x,y,z) (x+(y-x)*z)

///////////////////////////////////////////////////////////////////////////////////////////////////

const tulong PSM_CLIP_OUTSIDE = 0;
const tulong PSM_CLIP_PARTIAL = 1;
const tulong PSM_CLIP_INSIDE  = 2;

// Inlines ////////////////////////////////////////////////////////////////////////////////////////
#if (defined(LINUX) || defined(MAC_OS))
#define __forceinline inline
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline float MathSqrt(const float f)
{
    #ifdef PSM_GCC
        return sqrt(f);
    #else
        return sqrtf(f);
    #endif
}

inline float MathRandom(const tlong x)
{
    if(x == 0)
    {
        return 0;
    }

    float res = ((float)rand() / (float)RAND_MAX) * x;

    // If the number found is too near of x, decrease him to avoid error
    if(((float)(x) - res) <= RealEpsilon)
    {
        res -= 0.2f;
    }

    return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathFloatRandom(const float x)
{
    float res = ((float)rand() / (float)RAND_MAX) * x;

    // If the number found is too near of x, decrease him to avoid error
    if((x - res) <= RealEpsilon)
    {
        res -= 0.001f;
    }

    return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathFloatCenterRandom(const float x, const float )
{
    float res = ((float)rand() / (float)RAND_MAX) * x;

    // If the number found is too near of x, decrease him to avoid error
    if((x - res) <= RealEpsilon)
    {
        res -= 0.001f;
    }

    // Center and scale result
    res = res*2-x;

    return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathInfSupRandom(const float borneInf, const float borneSup)
{
    float res = ((borneSup - borneInf) * MathFloatRandom(1)) + borneInf;

    return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void MathFloatTolong(tlong * long_podlonger, const float f)
{
#if defined(PSM_MSVC) && defined(PSM_ASM)
    __asm  fld  f
    __asm  mov  edx, long_podlonger
    __asm  FRNDINT
    __asm  fistp dword ptr [edx];
#else
    *long_podlonger = (tlong) f;
#endif
}
///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathFloatAbs(const float value)
{

    if(value >= 0.0f)
    {
        return value;
    }

    return -value;

}


///////////////////////////////////////////////////////////////////////////////////////////////////

inline tlong MathFloatRound(const float value)
{
    tlong newValue;
    double fract, ent;

    fract = modf(value, &ent);

    newValue = (tlong)ent;

    if(fract > 0.5f)
    {
        newValue++;
    }
    else
    {
        if(fract < -0.5f)
        {
            newValue--;
        }
    }

    return newValue;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool MathFloatIsEqual(const float value,const float value2)
{
    if(MathFloatAbs(value-value2)>RealEpsilon)
    {
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool MathFloatIsGreater(const float value, const float value2)
{

    if(MathFloatIsEqual(value, value2))
    {
        return false;
    }

    if((value-value2)>0.0f)
    {
        return true;
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool MathFloatIsGreaterOrEqual(const float value, const float value2)
{

    if(MathFloatIsEqual(value, value2))
    {
        return true;
    }

    if((value-value2)>0.0f)
    {
        return true;
    }

    return false;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool MathQuadraticFormula(float A, float B, float C, float * u0, float * u1)
{
    float q = B*B - 4*A*C;
    if(q>=0)
    {
        float sq = (float)MathSqrt(q);
        float d = 1/(2*A);
        *u0 = ( -B + sq ) * d;
        *u1 = ( -B - sq ) * d;
        return true;
    }
    else
    {
        return false;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////

inline tdouble MathDoubleAbs(const tdouble value)
{

    if(value >= 0.0)
    {
        return value;
    }

    return -value;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

float MathFloatBezierCubic(const float &v1, const float &v2, const float& v3, const float& v4, const float s);

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathParQuadSplineF1(float t)
{
    return t*(2.0f*t-1.0f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathParQuadSplineF2(float t)
{
    return 4.0f*t*(1.0f-t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathParQuadSplineF3(float t)
{
    return (t-1.0f)*(2.0f*t-1.0f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathParCubicSplineF1(float t)
{
    return t*(4.5f*t*(t-1.0f)+1.0f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathParCubicSplineF2(float t)
{
    return 4.5f*t*((t-1.0f)*(-3.0f*t+1.0f));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathParCubicSplineF3(float t)
{
    return 4.5f*t*((t-1.0f)*(3.0f*t-2.0f));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathParCubicSplineF4(float t)
{
    return 4.5f*t*(t*(2.0f-t))-5.5f*t+1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathBezierQuadSplineF1(float t)
{
    return t*t;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathBezierQuadSplineF2(float t)
{
    return 2.0f*t*(1.0f-t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathBezierQuadSplineF3(float t)
{
    return (1.0f-t)*(1.0f-t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathBezierCubicSplineF1(float t)
{
    return t*t*t;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathBezierCubicSplineF2(float t)
{
    return 3.0f*t*t*(1.0f-t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathBezierCubicSplineF3(float t)
{
    return 3.0f*t*(1.0f-t)*(1.0f-t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathBezierCubicSplineF4(float t)
{
    return (1.0f-t)*(1.0f-t)*(1.0f-t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathCoonsQuadSplineF1(float t)
{
    return (1.0f-t)*(1.0f+t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathCoonsQuadSplineF2(float t)
{
    return t*t;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathCoonsQuadSplineF3(float t)
{
    return t*(1.0f-t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathCoonsCubicSplineF1(float t)
{
    return t*t*(2.0f*t-3.0f)+1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathCoonsCubicSplineF2(float t)
{
    return t*t*(3.0f-2.0f*t);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathCoonsCubicSplineF3(float t)
{
    return t*((t-1.0f)*(t-1.0f));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float MathCoonsCubicSplineF4(float t)
{
    return t*t*(t-1.0f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline float MathCos(const float f)
{
    #ifdef PSM_GCC
        return (float)cos(f);
    #else
        return cosf(f);
    #endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline float MathSin(const float f)
{
    #ifdef PSM_GCC
        return (float)sin(f);
    #else
        return sinf(f);
    #endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline float MathTan(const float f)
{
    #ifdef PSM_GCC
        return (float)tan(f);
    #else
        return tanf(f);
    #endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline float MathACos(const float f)
{
    #ifdef PSM_GCC
        return (float)acos(f);
    #else
        if(f> -1.0)
        {
            if( f < 1.0 )
            {
                return (float)(acos(f));
            }
            else
            {
                return 0.0f;
            }
        }
        else
        {
            return ZPI;
        }
    #endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline float MathASin(const float f)
{
    #ifdef PSM_GCC
        return (float)asin(f);
    #else
        return asinf(f);
    #endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline float MathATan(const float f)
{
    #ifdef PSM_GCC
        return atan(f);
    #else
        return atanf(f);
    #endif
}


///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool GetLowestRoot(float a, float b, float c, float maxR, float* root)
{
    // Check if a solution exists
    float determinant = b*b - 4.0f*a*c;

    // If determinant is negative it means no solutions.
    if (determinant < 0.0f)
    {
        return false;
    }

    // calculate the two roots: (if determinant == 0 then
    // x1==x2 but let's disregard that slight optimization)
    float sqrtD = (float)MathSqrt(determinant);
    float r1 = (-b - sqrtD) / (2*a);
    float r2 = (-b + sqrtD) / (2*a);

    // Sort so x1 <= x2
    if (r1 > r2)
    {
        float temp = r2;
        r2 = r1;
        r1 = temp;
    }

    // Get lowest root:
    if (r1 > 0 && r1 < maxR)
    {
        *root = r1;
        return true;
    }

    // It is possible that we want x2 - this can happen
    // if x1 < 0
    if (r2 > 0 && r2 < maxR)
    {
        *root = r2;
        return true;
    }

    // No (valid) solutions
    return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float Clamp(float val, float inf, float sup)
{
    if (val<inf) return inf;
    if (val>sup) return sup;
    return val;
}

#endif


