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


#include "stdafx.h"
#include "ZBaseMaths.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

const tvector3 tvector3::zero    (0, 0, 0);
const tvector3 tvector3::up    (0, 1, 0);
const tvector3 tvector3::one (1,1,1);
const tvector3 tvector3::XAxis(1,0,0);
const tvector3 tvector3::YAxis(0,1,0);
const tvector3 tvector3::ZAxis(0,0,1);
const tvector3 tvector3::opXAxis(-1,0,0);
const tvector3 tvector3::opYAxis(0,-1,0);
const tvector3 tvector3::opZAxis(0,0,-1);
const tvector3 tvector3::Epsilon(RealEpsilon,RealEpsilon,RealEpsilon);

///////////////////////////////////////////////////////////////////////////////////////////////////

const tvector4 tvector4::zero = vector4(0,0,0,0);
const tvector4 tvector4::one = vector4(1,1,1,1);
const tvector4 tvector4::XAxis = vector4(1,0,0,0);
const tvector4 tvector4::YAxis = vector4(0,1,0,0);
const tvector4 tvector4::ZAxis = vector4(0,0,1,0);
const tvector4 tvector4::WAxis = vector4(0,0,0,1);
const tvector4 tvector4::opXAxis = vector4(-1,0,0,0);
const tvector4 tvector4::opYAxis = vector4(0,-1,0,0);
const tvector4 tvector4::opZAxis = vector4(0,0,-1,0);
const tvector4 tvector4::opWAxis = vector4(0,0,0,-1);

///////////////////////////////////////////////////////////////////////////////////////////////////

const tcolor tcolor::black    (0, 0, 0, 0);
const tcolor tcolor::white    (1, 1, 1, 1);
const tcolor tcolor::red      (1, 0, 0, 1);
const tcolor tcolor::green    (0, 1, 0, 1);
const tcolor tcolor::blue     (0, 0, 1, 1);


tmatrix tmatrix::identity = tmatrix(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
tquaternion tquaternion::identity = tquaternion(0,0,0,1);

///////////////////////////////////////////////////////////////////////////////////////////////////

tvector3 tvector3::vecLimitDeviationAngleUtility (const bool insideOrOutside,
                                          const tvector3& source,
                                          const float cosineOfConeAngle,
                                          const tvector3& basis)
{
    // immediately return zero length input vectors
    float sourceLength = source.Length();
    if (sourceLength == 0) return source;

    // measure the angular diviation of "source" from "basis"
    const tvector3 direction = source / sourceLength;
    float cosineOfSourceAngle = direction.Dot (basis);

    // Simply return "source" if it already meets the angle criteria.
    // (note: we hope this top "if" gets compiled out since the flag
    // is a constant when the function is inlined into its caller)
    if (insideOrOutside)
    {
    // source vector is already inside the cone, just return it
    if (cosineOfSourceAngle >= cosineOfConeAngle) return source;
    }
    else
    {
    // source vector is already outside the cone, just return it
    if (cosineOfSourceAngle <= cosineOfConeAngle) return source;
    }

    // find the portion of "source" that is perpendicular to "basis"
    const tvector3 perp = source.perpendicularComponent (basis);

    // normalize that perpendicular
    tvector3 unitPerp = perp;
    unitPerp.Normalize ();

    // construct a new vector whose length equals the source vector,
    // and lies on the intersection of a plane (formed the source and
    // basis vectors) and a cone (whose axis is "basis" and whose
    // angle corresponds to cosineOfConeAngle)
    float perpDist = (float)sqrt (1 - (cosineOfConeAngle * cosineOfConeAngle));
    const tvector3 c0 = basis * cosineOfConeAngle;
    const tvector3 c1 = unitPerp * perpDist;
    return (c0 + c1) * sourceLength;
}


tvector3 vecLimitDeviationAngleUtility (const bool insideOrOutside,
                                          const tvector3& source,
                                          const float cosineOfConeAngle,
                                          const tvector3& basis)
{
    // immediately return zero length input vectors
    float sourceLength = source.Length();
    if (sourceLength == 0) return source;

    // measure the angular diviation of "source" from "basis"
    const tvector3 direction = source / sourceLength;
    float cosineOfSourceAngle = direction.Dot (basis);

    // Simply return "source" if it already meets the angle criteria.
    // (note: we hope this top "if" gets compiled out since the flag
    // is a constant when the function is inlined into its caller)
    if (insideOrOutside)
    {
    // source vector is already inside the cone, just return it
    if (cosineOfSourceAngle >= cosineOfConeAngle) return source;
    }
    else
    {
    // source vector is already outside the cone, just return it
    if (cosineOfSourceAngle <= cosineOfConeAngle) return source;
    }

    // find the portion of "source" that is perpendicular to "basis"
    const tvector3 perp = source.perpendicularComponent (basis);

    // normalize that perpendicular
    tvector3 unitPerp = perp;
    unitPerp.Normalize ();

    // construct a new vector whose length equals the source vector,
    // and lies on the intersection of a plane (formed the source and
    // basis vectors) and a cone (whose axis is "basis" and whose
    // angle corresponds to cosineOfConeAngle)
    float perpDist = (float)sqrt (1 - (cosineOfConeAngle * cosineOfConeAngle));
    const tvector3 c0 = basis * cosineOfConeAngle;
    const tvector3 c1 = unitPerp * perpDist;
    return (c0 + c1) * sourceLength;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

void tvector3::ClosestPointOnSegment(const tvector3 & point, const tvector3 & point1, const tvector3 & point2 )
{
    tvector3 c, segment;
    tvector3 v;

    c = point - point1;

    tvector3 tmp = point2 - point1;
   segment.Normalize(tmp);

    float t = segment.Dot(c);

    if(t < 0)
    {
      x = point1.x;
      y = point1.y;
      z = point1.z;
    }
   else
   {
       tmp = point2 - point1;
       float distance = tmp.Length();

       if(t > distance)
       {
         x = point2.x;
         y = point2.y;
         z = point2.z;
       }
      else
      {
         x = point1.x + segment.x * t;
         y = point1.y + segment.y * t;
         z = point1.z + segment.z * t;
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void tvector3::ClosestPointOnTriangle(const tvector3 & point, const tvector3 & point1, const tvector3 & point2, const tvector3 & point3 )
{
    tvector3 pointSegAB;
    tvector3 pointSegBC;
    tvector3 pointSegCA;

    pointSegAB.ClosestPointOnSegment( point, point1, point2 );
    pointSegBC.ClosestPointOnSegment( point, point2, point3 );
    pointSegCA.ClosestPointOnSegment( point, point3, point1 );

    tvector3 tmp = point - pointSegAB;
    float dsegAB = tmp.Length();

    tmp = point - pointSegBC;
    float dsegBC = tmp.Length();

    tmp = point - pointSegCA;
    float dsegCA = tmp.Length();

    if( (dsegAB <  dsegBC) && ( dsegAB < dsegCA ))
    {
      x = pointSegAB.x;
      y = pointSegAB.y;
      z = pointSegAB.z;
    }
    else{
        if( (dsegBC < dsegAB) && ( dsegBC < dsegCA ))
        {
         x = pointSegBC.x;
         y = pointSegBC.y;
         z = pointSegBC.z;
        }
        else
      {
         x = pointSegCA.x;
         y = pointSegCA.y;
         z = pointSegCA.z;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////

void tvector3::ParametricQuadratic(const tvector3& v1, const tvector3& v2, const tvector3& v3, const float s)
{
    x = v1.x * MathParQuadSplineF3(s) + v2.x * MathParQuadSplineF2(s) + v3.x * MathParQuadSplineF1(s);
    y = v1.y * MathParQuadSplineF3(s) + v2.y * MathParQuadSplineF2(s) + v3.y * MathParQuadSplineF1(s);
    z = v1.z * MathParQuadSplineF3(s) + v2.z * MathParQuadSplineF2(s) + v3.z * MathParQuadSplineF1(s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void tvector3::ParametricCubic(const tvector3& v1, const tvector3& v2, const tvector3& v3, const tvector3& v4, const float s)
{
    x = v1.x * MathParCubicSplineF4(s) + v2.x * MathParCubicSplineF3(s) + v4.x * MathParCubicSplineF2(s) + v3.x * MathParCubicSplineF1(s);
    y = v1.y * MathParCubicSplineF4(s) + v2.y * MathParCubicSplineF3(s) + v4.y * MathParCubicSplineF2(s) + v3.y * MathParCubicSplineF1(s);
    z = v1.z * MathParCubicSplineF4(s) + v2.z * MathParCubicSplineF3(s) + v4.z * MathParCubicSplineF2(s) + v3.z * MathParCubicSplineF1(s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void tvector3::BezierQuadratic(const tvector3& v1, const tvector3& v2, const tvector3& v3, const float s)
{
    x = v1.x * MathBezierQuadSplineF3(s) + v2.x * MathBezierQuadSplineF2(s) + v3.x * MathBezierQuadSplineF1(s);
    y = v1.y * MathBezierQuadSplineF3(s) + v2.y * MathBezierQuadSplineF2(s) + v3.y * MathBezierQuadSplineF1(s);
    z = v1.z * MathBezierQuadSplineF3(s) + v2.z * MathBezierQuadSplineF2(s) + v3.z * MathBezierQuadSplineF1(s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void tvector3::BezierCubic(const tvector3& v1, const tvector3& v2, const tvector3& v3, const tvector3& v4, const float s)
{
    x = v1.x * MathBezierCubicSplineF4(s) + v2.x * MathBezierCubicSplineF3(s) + v4.x * MathBezierCubicSplineF2(s) + v3.x * MathBezierCubicSplineF1(s);
    y = v1.y * MathBezierCubicSplineF4(s) + v2.y * MathBezierCubicSplineF3(s) + v4.y * MathBezierCubicSplineF2(s) + v3.y * MathBezierCubicSplineF1(s);
    z = v1.z * MathBezierCubicSplineF4(s) + v2.z * MathBezierCubicSplineF3(s) + v4.z * MathBezierCubicSplineF2(s) + v3.z * MathBezierCubicSplineF1(s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void tvector3::CoonsQuadratic(const tvector3& v1, const tvector3& t1, const tvector3& v2, const float s)
{
    x = v1.x * MathCoonsQuadSplineF1(s) + v2.x * MathCoonsQuadSplineF2(s) + t1.x * MathCoonsQuadSplineF3(s);
    y = v1.y * MathCoonsQuadSplineF1(s) + v2.y * MathCoonsQuadSplineF2(s) + t1.y * MathCoonsQuadSplineF3(s);
    z = v1.z * MathCoonsQuadSplineF1(s) + v2.z * MathCoonsQuadSplineF2(s) + t1.z * MathCoonsQuadSplineF3(s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void tvector3::CoonsCubic(const tvector3& v1, const tvector3& t1, const tvector3& v2, const tvector3& t2, const float s)
{
    x = v1.x * MathCoonsCubicSplineF1(s) + v2.x * MathCoonsCubicSplineF2(s) + t1.x * MathCoonsCubicSplineF3(s) + t2.x * MathCoonsCubicSplineF4(s);
    y = v1.y * MathCoonsCubicSplineF1(s) + v2.y * MathCoonsCubicSplineF2(s) + t1.y * MathCoonsCubicSplineF3(s) + t2.y * MathCoonsCubicSplineF4(s);
    z = v1.z * MathCoonsCubicSplineF1(s) + v2.z * MathCoonsCubicSplineF2(s) + t1.z * MathCoonsCubicSplineF3(s) + t2.z * MathCoonsCubicSplineF4(s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void tvector3::CatmullRom(const tvector3& v1, const tvector3& v2, const tvector3& v3, const tvector3& v4, const float s)
{
    x = 0.5f*((-v2.x + 3 * v1.x - 3 * v4.x + v3.x) * s * s * s + (2 * v2.x - 5 * v1.x + 4 * v4.x - v3.x) * s * s+(-v2.x + v4.x) * s + 2 * v1.x);
    y = 0.5f*((-v2.y + 3 * v1.y - 3 * v4.y + v3.y) * s * s * s + (2 * v2.y - 5 * v1.y + 4 * v4.y - v3.y) * s * s+(-v2.y + v4.y) * s + 2 * v1.y);
    z = 0.5f*((-v2.z + 3 * v1.z - 3 * v4.z + v3.z) * s * s * s + (2 * v2.z - 5 * v1.z + 4 * v4.z - v3.z) * s * s+(-v2.z + v4.z) * s + 2 * v1.z);

}

///////////////////////////////////////////////////////////////////////////////////////////////////

float tmatrix::Inverse(const tmatrix &srcMatrix, bool affine )
{
    float det = 0;

    if(affine)
    {
        det = GetDeterminant();
        float s = 1 / det;
        m[0][0] = (srcMatrix.m[1][1]*srcMatrix.m[2][2] - srcMatrix.m[1][2]*srcMatrix.m[2][1]) * s;
        m[0][1] = (srcMatrix.m[2][1]*srcMatrix.m[0][2] - srcMatrix.m[2][2]*srcMatrix.m[0][1]) * s;
        m[0][2] = (srcMatrix.m[0][1]*srcMatrix.m[1][2] - srcMatrix.m[0][2]*srcMatrix.m[1][1]) * s;
        m[1][0] = (srcMatrix.m[1][2]*srcMatrix.m[2][0] - srcMatrix.m[1][0]*srcMatrix.m[2][2]) * s;
        m[1][1] = (srcMatrix.m[2][2]*srcMatrix.m[0][0] - srcMatrix.m[2][0]*srcMatrix.m[0][2]) * s;
        m[1][2] = (srcMatrix.m[0][2]*srcMatrix.m[1][0] - srcMatrix.m[0][0]*srcMatrix.m[1][2]) * s;
        m[2][0] = (srcMatrix.m[1][0]*srcMatrix.m[2][1] - srcMatrix.m[1][1]*srcMatrix.m[2][0]) * s;
        m[2][1] = (srcMatrix.m[2][0]*srcMatrix.m[0][1] - srcMatrix.m[2][1]*srcMatrix.m[0][0]) * s;
        m[2][2] = (srcMatrix.m[0][0]*srcMatrix.m[1][1] - srcMatrix.m[0][1]*srcMatrix.m[1][0]) * s;
        m[3][0] = -(m[0][0]*srcMatrix.m[3][0] + m[1][0]*srcMatrix.m[3][1] + m[2][0]*srcMatrix.m[3][2]);
        m[3][1] = -(m[0][1]*srcMatrix.m[3][0] + m[1][1]*srcMatrix.m[3][1] + m[2][1]*srcMatrix.m[3][2]);
        m[3][2] = -(m[0][2]*srcMatrix.m[3][0] + m[1][2]*srcMatrix.m[3][1] + m[2][2]*srcMatrix.m[3][2]);
    }
    else
    {
        // transpose matrix
        float src[16];
        for ( int i=0; i<4; ++i )
        {
            src[i]      = srcMatrix.m16[i*4];
            src[i + 4]  = srcMatrix.m16[i*4 + 1];
            src[i + 8]  = srcMatrix.m16[i*4 + 2];
            src[i + 12] = srcMatrix.m16[i*4 + 3];
        }

        // calculate pairs for first 8 elements (cofactors)
        float tmp[12]; // temp array for pairs
        tmp[0]  = src[10] * src[15];
        tmp[1]  = src[11] * src[14];
        tmp[2]  = src[9]  * src[15];
        tmp[3]  = src[11] * src[13];
        tmp[4]  = src[9]  * src[14];
        tmp[5]  = src[10] * src[13];
        tmp[6]  = src[8]  * src[15];
        tmp[7]  = src[11] * src[12];
        tmp[8]  = src[8]  * src[14];
        tmp[9]  = src[10] * src[12];
        tmp[10] = src[8]  * src[13];
        tmp[11] = src[9]  * src[12];

        // calculate first 8 elements (cofactors)
        m16[0] = (tmp[0] * src[5] + tmp[3] * src[6] + tmp[4]  * src[7]) - (tmp[1] * src[5] + tmp[2] * src[6] + tmp[5]  * src[7]);
        m16[1] = (tmp[1] * src[4] + tmp[6] * src[6] + tmp[9]  * src[7]) - (tmp[0] * src[4] + tmp[7] * src[6] + tmp[8]  * src[7]);
        m16[2] = (tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7]) - (tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7]);
        m16[3] = (tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6]) - (tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6]);
        m16[4] = (tmp[1] * src[1] + tmp[2] * src[2] + tmp[5]  * src[3]) - (tmp[0] * src[1] + tmp[3] * src[2] + tmp[4]  * src[3]);
        m16[5] = (tmp[0] * src[0] + tmp[7] * src[2] + tmp[8]  * src[3]) - (tmp[1] * src[0] + tmp[6] * src[2] + tmp[9]  * src[3]);
        m16[6] = (tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3]) - (tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3]);
        m16[7] = (tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2]) - (tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2]);

        // calculate pairs for second 8 elements (cofactors)
        tmp[0]  = src[2] * src[7];
        tmp[1]  = src[3] * src[6];
        tmp[2]  = src[1] * src[7];
        tmp[3]  = src[3] * src[5];
        tmp[4]  = src[1] * src[6];
        tmp[5]  = src[2] * src[5];
        tmp[6]  = src[0] * src[7];
        tmp[7]  = src[3] * src[4];
        tmp[8]  = src[0] * src[6];
        tmp[9]  = src[2] * src[4];
        tmp[10] = src[0] * src[5];
        tmp[11] = src[1] * src[4];

        // calculate second 8 elements (cofactors)
        m16[8]  = (tmp[0]  * src[13] + tmp[3]  * src[14] + tmp[4]  * src[15]) - (tmp[1]  * src[13] + tmp[2]  * src[14] + tmp[5]  * src[15]);
        m16[9]  = (tmp[1]  * src[12] + tmp[6]  * src[14] + tmp[9]  * src[15]) - (tmp[0]  * src[12] + tmp[7]  * src[14] + tmp[8]  * src[15]);
        m16[10] = (tmp[2]  * src[12] + tmp[7]  * src[13] + tmp[10] * src[15]) - (tmp[3]  * src[12] + tmp[6]  * src[13] + tmp[11] * src[15]);
        m16[11] = (tmp[5]  * src[12] + tmp[8]  * src[13] + tmp[11] * src[14]) - (tmp[4]  * src[12] + tmp[9]  * src[13] + tmp[10] * src[14]);
        m16[12] = (tmp[2]  * src[10] + tmp[5]  * src[11] + tmp[1]  * src[9])  - (tmp[4]  * src[11] + tmp[0]  * src[9]  + tmp[3]  * src[10]);
        m16[13] = (tmp[8]  * src[11] + tmp[0]  * src[8]  + tmp[7]  * src[10]) - (tmp[6]  * src[10] + tmp[9]  * src[11] + tmp[1]  * src[8]);
        m16[14] = (tmp[6]  * src[9]  + tmp[11] * src[11] + tmp[3]  * src[8])  - (tmp[10] * src[11] + tmp[2]  * src[8]  + tmp[7]  * src[9]);
        m16[15] = (tmp[10] * src[10] + tmp[4]  * src[8]  + tmp[9]  * src[9])  - (tmp[8]  * src[9]  + tmp[11] * src[10] + tmp[5]  * src[8]);

        // calculate determinant
        float det = src[0]*m16[0]+src[1]*m16[1]+src[2]*m16[2]+src[3]*m16[3];

        // calculate matrix inverse
        float invdet = 1 / det;
        for ( int j=0; j<16; ++j )
        {
            m16[j] *= invdet;
        }
    }

    return det;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

float tmatrix::Inverse(bool affine)
{
    float det = 0;

    if(affine)
    {
        det = GetDeterminant();
        float s = 1 / det;
        float v00 = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * s;
        float v01 = (m[2][1]*m[0][2] - m[2][2]*m[0][1]) * s;
        float v02 = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * s;
        float v10 = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * s;
        float v11 = (m[2][2]*m[0][0] - m[2][0]*m[0][2]) * s;
        float v12 = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * s;
        float v20 = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * s;
        float v21 = (m[2][0]*m[0][1] - m[2][1]*m[0][0]) * s;
        float v22 = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * s;
        float v30 = -(v00*m[3][0] + v10*m[3][1] + v20*m[3][2]);
        float v31 = -(v01*m[3][0] + v11*m[3][1] + v21*m[3][2]);
        float v32 = -(v02*m[3][0] + v12*m[3][1] + v22*m[3][2]);
        m[0][0] = v00;
        m[0][1] = v01;
        m[0][2] = v02;
        m[1][0] = v10;
        m[1][1] = v11;
        m[1][2] = v12;
        m[2][0] = v20;
        m[2][1] = v21;
        m[2][2] = v22;
        m[3][0] = v30;
        m[3][1] = v31;
        m[3][2] = v32;
    }
    else
    {
        // transpose matrix
        float src[16];
        for ( int i=0; i<4; ++i )
        {
            src[i]      = m16[i*4];
            src[i + 4]  = m16[i*4 + 1];
            src[i + 8]  = m16[i*4 + 2];
            src[i + 12] = m16[i*4 + 3];
        }

        // calculate pairs for first 8 elements (cofactors)
        float tmp[12]; // temp array for pairs
        tmp[0]  = src[10] * src[15];
        tmp[1]  = src[11] * src[14];
        tmp[2]  = src[9]  * src[15];
        tmp[3]  = src[11] * src[13];
        tmp[4]  = src[9]  * src[14];
        tmp[5]  = src[10] * src[13];
        tmp[6]  = src[8]  * src[15];
        tmp[7]  = src[11] * src[12];
        tmp[8]  = src[8]  * src[14];
        tmp[9]  = src[10] * src[12];
        tmp[10] = src[8]  * src[13];
        tmp[11] = src[9]  * src[12];

        // calculate first 8 elements (cofactors)
        m16[0] = (tmp[0] * src[5] + tmp[3] * src[6] + tmp[4]  * src[7]) - (tmp[1] * src[5] + tmp[2] * src[6] + tmp[5]  * src[7]);
        m16[1] = (tmp[1] * src[4] + tmp[6] * src[6] + tmp[9]  * src[7]) - (tmp[0] * src[4] + tmp[7] * src[6] + tmp[8]  * src[7]);
        m16[2] = (tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7]) - (tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7]);
        m16[3] = (tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6]) - (tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6]);
        m16[4] = (tmp[1] * src[1] + tmp[2] * src[2] + tmp[5]  * src[3]) - (tmp[0] * src[1] + tmp[3] * src[2] + tmp[4]  * src[3]);
        m16[5] = (tmp[0] * src[0] + tmp[7] * src[2] + tmp[8]  * src[3]) - (tmp[1] * src[0] + tmp[6] * src[2] + tmp[9]  * src[3]);
        m16[6] = (tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3]) - (tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3]);
        m16[7] = (tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2]) - (tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2]);

        // calculate pairs for second 8 elements (cofactors)
        tmp[0]  = src[2] * src[7];
        tmp[1]  = src[3] * src[6];
        tmp[2]  = src[1] * src[7];
        tmp[3]  = src[3] * src[5];
        tmp[4]  = src[1] * src[6];
        tmp[5]  = src[2] * src[5];
        tmp[6]  = src[0] * src[7];
        tmp[7]  = src[3] * src[4];
        tmp[8]  = src[0] * src[6];
        tmp[9]  = src[2] * src[4];
        tmp[10] = src[0] * src[5];
        tmp[11] = src[1] * src[4];

        // calculate second 8 elements (cofactors)
        m16[8]  = (tmp[0]  * src[13] + tmp[3]  * src[14] + tmp[4]  * src[15]) - (tmp[1]  * src[13] + tmp[2]  * src[14] + tmp[5]  * src[15]);
        m16[9]  = (tmp[1]  * src[12] + tmp[6]  * src[14] + tmp[9]  * src[15]) - (tmp[0]  * src[12] + tmp[7]  * src[14] + tmp[8]  * src[15]);
        m16[10] = (tmp[2]  * src[12] + tmp[7]  * src[13] + tmp[10] * src[15]) - (tmp[3]  * src[12] + tmp[6]  * src[13] + tmp[11] * src[15]);
        m16[11] = (tmp[5]  * src[12] + tmp[8]  * src[13] + tmp[11] * src[14]) - (tmp[4]  * src[12] + tmp[9]  * src[13] + tmp[10] * src[14]);
        m16[12] = (tmp[2]  * src[10] + tmp[5]  * src[11] + tmp[1]  * src[9])  - (tmp[4]  * src[11] + tmp[0]  * src[9]  + tmp[3]  * src[10]);
        m16[13] = (tmp[8]  * src[11] + tmp[0]  * src[8]  + tmp[7]  * src[10]) - (tmp[6]  * src[10] + tmp[9]  * src[11] + tmp[1]  * src[8]);
        m16[14] = (tmp[6]  * src[9]  + tmp[11] * src[11] + tmp[3]  * src[8])  - (tmp[10] * src[11] + tmp[2]  * src[8]  + tmp[7]  * src[9]);
        m16[15] = (tmp[10] * src[10] + tmp[4]  * src[8]  + tmp[9]  * src[9])  - (tmp[8]  * src[9]  + tmp[11] * src[10] + tmp[5]  * src[8]);

        // calculate determinant
        float det = src[0]*m16[0]+src[1]*m16[1]+src[2]*m16[2]+src[3]*m16[3];

        // calculate matrix inverse
        float invdet = 1 / det;
        for ( int j=0; j<16; ++j )
        {
            m16[j] *= invdet;
        }
    }

    return det;

}

