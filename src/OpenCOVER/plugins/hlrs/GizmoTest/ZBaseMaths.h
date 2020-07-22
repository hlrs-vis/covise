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


#ifndef ZBASEMATHS_H__
#define ZBASEMATHS_H__

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <memory.h>
#include "ZBaseDefs.h"
#include "ZMathsFunc.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

struct tmatrix;
struct tquaternion;
struct tvector4;

///////////////////////////////////////////////////////////////////////////////////////////////////

struct tvector2
{

public:
	float x, y;

public:
	tvector2();
	tvector2( const float * );
	tvector2( float x, float y );

	// casting
	operator float* ();
	operator const float* () const;

	// assignment operators
	tvector2& operator += ( const tvector2& );
	tvector2& operator -= ( const tvector2& );
	tvector2& operator *= ( float );
	tvector2& operator /= ( float );

	// unary operators
	tvector2 operator + () const;
	tvector2 operator - () const;

	// binary operators
	tvector2 operator + ( const tvector2& ) const;
	tvector2 operator - ( const tvector2& ) const;
	tvector2 operator * ( float ) const;
	tvector2 operator / ( float ) const;

	// Comparaison operators
	bool operator == ( const tvector2& ) const;
	bool operator != ( const tvector2& ) const;

	// Member Functions
	float Length( ) const;
	float LengthSq( ) const;
	float Dot( const tvector2 &v) const;
	float CCW( const tvector2 &v ) const;

	void Add(const tvector2 &v);
	void Add(const tvector2 &v1, const tvector2 &v2);

	void Subtract(const tvector2 &v);
	void Subtract(const tvector2 &v1, const tvector2 &v2);

	void Minimize(const tvector2 &v);
	void Minimize(const tvector2 &v1, const tvector2 &v2);

	void Maximize(const tvector2 &v);
	void Maximize(const tvector2 &v1, const tvector2 &v2);

	void Scale(const float s);
	void Lerp(const tvector2 &v1, const tvector2 &v2, float s );

	void Normalize();
	void Normalize(const tvector2 &v);

	void BaryCentric(const tvector2 &v1, const tvector2 &v2, const tvector2 &v3, float f, float g);
	void TransformPoint(const tmatrix &matrix );
	void TransformPoint(const tvector2 &v, const tmatrix &matrix );
	void TransformVector(const tmatrix &matrix );
	void TransformVector(const tvector2 &v, const tmatrix &matrix );
};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct tvector3
{

public:
	float x, y, z;

public:
	// constructor
	tvector3();
	tvector3(const float * coord);
	tvector3(float x, float y, float z );
	tvector3(const tvector3& v1,  const tvector3& v2);
	tvector3(const tvector4& v);
	tvector3(float v) {x=y=z=v;}
	// casting
	operator float* ();
	operator const float* () const;

	// assignment operators
	tvector3& operator += ( const tvector3& );
	tvector3& operator -= ( const tvector3& );
	tvector3& operator *= ( float );
	tvector3& operator *= ( tvector3 );
	tvector3& operator /= ( float );
	tvector3& operator /= ( tvector3 );

	// unary operators
	tvector3 operator + () const;
	tvector3 operator - () const;

	// binary operators
	tvector3 operator + ( const tvector3& ) const;
	tvector3 operator - ( const tvector3& ) const;
	tvector3 operator * ( float ) const;
	tvector3 operator * ( tvector3 ) const;
	tvector3 operator / ( float ) const;
	tvector3 operator / ( tvector3 ) const;

	// Comparaison operators
	bool operator == ( const tvector3& ) const;
	bool operator != ( const tvector3& ) const;

	// Member Functions
	float Length( ) const;
	float LengthSq( ) const;
	float Dot( const tvector3 &v) const;

	bool IsVeryClose(tvector3 &v);

	void Cross(const tvector3 &v);
	void Cross(const tvector3 &v1, const tvector3 &v2);

	void Add(const tvector3 &v);
	void Add(const tvector3 &v1, const tvector3 &v2);

	void Subtract(const tvector3 &v);
	void Subtract(const tvector3 &v1, const tvector3 &v2);

	void Minimize(const tvector3 &v);
	void Minimize(const tvector3 &v1, const tvector3 &v2);

	void Maximize(const tvector3 &v);
	void Maximize(const tvector3 &v1, const tvector3 &v2);

	void Scale(const float s);
	void Lerp(const tvector3 &v1, const tvector3 &v2, float s );

	tvector3 InterpolateHermite(const tvector3 &nextKey, const tvector3 &nextKeyP1, const tvector3 &prevKey, float ratio)
	{
		//((tvector3*)res)->Lerp(m_Value, nextKey.m_Value, ratio );
		//return *((tvector3*)res);
		float t = ratio;
		float t2 = t*t;
		float t3 = t2*t;
		float h1 = 2*t3 - 3*t2 + 1.0f;
		float h2 = -2*t3 + 3*t2;
		float h3 = (t3 - 2*t2 + t) * .5f;
		float h4 = (t3 - t2) *.5f;

		tvector3 res;
		res = (*this) * h1;
		res += nextKey *h2;
		res += (nextKey - prevKey) * h3;
		res += (nextKeyP1 - (*this)) * h4;

		return  res;
	}

	tvector3 InterpolationCubique(const tvector3 & y0, const tvector3 & y1, const tvector3 & y2, const tvector3 & y3, float mu)
	{
		tvector3 a0,a1,a2,a3;
		float mu2;

		mu2 = mu*mu;
		a0 = y3 - y2 - y0 + y1;
		a1 = y0 - y1 - a0;
		a2 = y2 - y0;
		a3 = y1;

		return (a0*mu*mu2+a1*mu2+a2*mu+a3);
	}

	void Normal(const tvector3 & point1, const tvector3 & point2, const tvector3 & point3);

	void ParametricQuadratic(const tvector3& v1, const tvector3& v2, const tvector3& v3, const float s);
	void ParametricCubic(const tvector3& v1, const tvector3& v2, const tvector3& v3, const tvector3& v4, const float s);
	void BezierQuadratic(const tvector3& v1, const tvector3& v2, const tvector3& v3, const float s);
	void BezierCubic(const tvector3& v1, const tvector3& v2, const tvector3& v3, const tvector3& v4, const float s);
	void CoonsQuadratic(const tvector3& v1, const tvector3& t1, const tvector3& v2, const float s);
	void CoonsCubic(const tvector3& v1, const tvector3& t1, const tvector3& v2, const tvector3& t2, const float s);
	void CatmullRom(const tvector3& v1, const tvector3& v2, const tvector3& v3, const tvector3& v4, const float s);

	void Normalize();
	void Normalize(const tvector3 &v);

	void BaryCentric(const tvector3 &v1, const tvector3 &v2, const tvector3 &v3, float f, float g);
	void TransformPoint(const tmatrix &matrix );
	void TransformPoint(const tvector3 &v, const tmatrix &matrix );
	void TransformVector(const tmatrix &matrix );
	void TransformVector(const tvector3 &v, const tmatrix &matrix );

	void ClosestPointOnSegment(const tvector3 & point, const tvector3 & point1, const tvector3 & point2 );
	void ClosestPointOnTriangle(const tvector3 & point, const  tvector3 & point1, const tvector3 & point2, const tvector3 & point3 );

	tvector3 truncateLength (const float maxLength) const;
	float lengthSquared (void) const {return this->Dot (*this);}

	// return component of vector parallel to a unit basis vector
	// (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))

	inline tvector3 parallelComponent (const tvector3& unitBasis) const
	{
		const float projection = this->Dot (unitBasis);
		return unitBasis * projection;
	}

	// return component of vector perpendicular to a unit basis vector
	// (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))

	inline tvector3 perpendicularComponent (const tvector3& unitBasis) const
	{
		return (*this) - parallelComponent (unitBasis);
	}

	tvector3 vecLimitDeviationAngleUtility (const bool insideOrOutside,
		const tvector3& source,
		const float cosineOfConeAngle,
		const tvector3& basis);

	void set(float fx,float fy,float fz);

	static float Distance(const tvector3 &src,const tvector3 &dst)
	{
		return (float)sqrt( (src.x-dst.x)*(src.x-dst.x)+(src.y-dst.y)*(src.y-dst.y)+(src.z-dst.z)*(src.z-dst.z));
	}

	void Reflect(const tvector3 &v)
	{
		/*
		float fac = 2.0f * (normal.x * x + normal.y * y + normal.z * z);
		*this = tvector3(fac * normal.x-x,
			fac * normal.y-y,
			fac * normal.z-z);
			*/
		
		tvector3 mx, my, mz;

		//GLfloat dot = p[0]*v[0] + p[1]*v[1] + p[2]*v[2];
		mx.x = 1 - 2*v.x*v.x;
		my.x = - 2*v.x*v.y;
		mz.x = - 2*v.x*v.z;
		//m[3].x = 2*dot*v.x;
		mx.y = - 2*v.y*v.x;
		my.y = 1 - 2*v.y*v.y;
		mz.y = - 2*v.y*v.z;
		//m[3].y = 2*dot*v.y;
		mx.z = - 2*v.z*v.x;
		my.z = - 2*v.z*v.y;
		mz.z = 1 - 2*v.z*v.z;

		//m[3][2] = 2*dot*v[2];
		*this = tvector3(mx.Dot(*this), my.Dot(*this), mz.Dot(*this));
		
	}
	// static
	static const tvector3 zero;
	static const tvector3 one;
	static const tvector3 up;
	static const tvector3 XAxis;
	static const tvector3 YAxis;
	static const tvector3 ZAxis;
	static const tvector3 opXAxis;
	static const tvector3 opYAxis;
	static const tvector3 opZAxis;
	static const tvector3 Epsilon;

};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct tvector4
{

public:
	float x, y, z, w;

public:
	//inline tvector4() {};
	/*

	inline tvector4(float v) {x=y=z=w=v;}
	inline tvector4( const float* pf)
	{
	x = pf[0];
	y = pf[1];
	z = pf[2];
	w = pf[3];
	}
	inline tvector4( float _x, float _y, float _z, float _w ) : x(_x), y(_y), z(_z), w(_w) {}
	inline tvector4( float _x, float _y, float _z) : x(_x), y(_y), z(_z), w(0) {}


	inline tvector4(const tvector3 &normal, float distance);
	inline tvector4(const tvector3 & point1, const tvector3 & point2, const tvector3 & point3);
	inline tvector4(const tvector3 & point1, const tvector3 & normal);
	inline tvector4(const tvector4 & pt) { x = pt.x; y = pt.y; z = pt.z; w = pt.w; }
	*/
	//        inline tvector4( const tvector4& plane );


	// Member Function
	inline void Normalize();
	inline float DotCoord(const tvector3 & vector);
	inline float DotNormal(const tvector3 & vector);
	inline void Init(const tvector3 & p_point1, const tvector3 & p_normal);
	inline bool IsFrontFacingTo(const tvector3& direction) const;
	inline float SignedDistanceTo(const tvector3& point) const;
	inline bool RayInter( tvector3 & interPoint, const tvector3 & position, const tvector3 & direction);

	// casting
	operator float* () { return (float *) &x; }
	operator const float* () const;

	// assignment operators
	tvector4& operator += ( const tvector4& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;
		return (*this);
	}
	tvector4& operator -= ( const tvector4& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;
		return (*this);
	}
	tvector4& operator = ( const tvector3& );
	tvector4& operator *= ( float );
	tvector4& operator /= ( float );

	// unary operators
	tvector4 operator + () const;
	tvector4 operator - () const;

	// binary operators
	/*
	tvector4 operator + ( const tvector4& ) const;
	tvector4 operator - ( const tvector4& ) const;
	*/
	tvector4 operator * ( float ) const;
	/*

	tvector4 operator / ( float ) const;
	*/
	// Comparaison operators
	bool operator == ( const tvector4& vt) const
	{
		return (( x==vt.x)&&(y==vt.y)&&(z==vt.z)&&(w==vt.w));
	}
	bool operator != ( const tvector4& ) const;

	// Member Functions
	float Length( ) const;
	float LengthSq( ) const;
	float Dot( const tvector4 &v) const;
	float Dot( const tvector3 &v) const;

	void Cross(const tvector4 &v);
	void Cross(const tvector4 &v1, const tvector4 &v2);

	void Add(const tvector4 &v);
	void Add(const tvector4 &v1, const tvector4 &v2);

	void Subtract(const tvector4 &v);
	void Subtract(const tvector4 &v1, const tvector4 &v2);

	void Minimize(const tvector4 &v);
	void Minimize(const tvector4 &v1, const tvector4 &v2);

	void Maximize(const tvector4 &v);
	void Maximize(const tvector4 &v1, const tvector4 &v2);

	void Scale(const float s);
	void Lerp(const tvector4 &v1, const tvector4 &v2, float s );

	void Cross(tvector4 &v1, const tvector4 &v2, const tvector4 &v3);

	//      void Normalize();
	void Normalize(const tvector4 &v);

	void BaryCentric(const tvector4 &v1, const tvector4 &v2, const tvector4 &v3, float f, float g);
	void Transform(const tmatrix &matrix );
	void Transform(const tvector4 &v, const tmatrix &matrix );

	/**
	merge and get smallest volume for the tvector4 sphere and another as parameter
	*/
	void MergeBSphere(const tvector4& sph);
	/**
	is a sphere fit totaly inside another
	*/
	bool CanFitIn(const tvector4 & wsph) const;

	static const tvector4 zero;
	static const tvector4 one;
	static const tvector4 XAxis;
	static const tvector4 YAxis;
	static const tvector4 ZAxis;
	static const tvector4 WAxis;
	static const tvector4 opXAxis;
	static const tvector4 opYAxis;
	static const tvector4 opZAxis;
	static const tvector4 opWAxis;
};

inline tvector4 vector4(float v)
{
	tvector4 ret;
	ret.x=ret.y=ret.z=ret.w=v;
	return ret;
}
inline tvector4 vector4( const tvector3& pf)
{
	tvector4 ret;
	ret.x = pf.x;
	ret.y = pf.y;
	ret.z = pf.z;
	ret.w = 0;
	return ret;
}

inline tvector4 vector4( const float* pf)
{
	tvector4 ret;
	ret.x = pf[0];
	ret.y = pf[1];
	ret.z = pf[2];
	ret.w = pf[3];
	return ret;
}
inline tvector4 vector4( float _x, float _y, float _z, float _w )
{
	tvector4 ret;
	ret.x=_x;
	ret.y=_y;
	ret.z=_z;
	ret.w=_w;
	return ret;
}
inline tvector4 vector4( float _x, float _y, float _z)
{
	tvector4 ret;
	ret.x = _x;
	ret.y = _y;
	ret.z = _z;
	ret.w = 0;
	return ret;
}


inline tvector4 vector4(const tvector3 &normal, float distance)
{
	tvector4 ret;
	ret.x = normal.x;
	ret.y = normal.y;
	ret.z = normal.z;
	ret.w = distance;
	return ret;
}
inline tvector4 vector4(const tvector3 & point1, const tvector3 & point2, const tvector3 & point3)
{
	tvector4 ret;
	tvector3 normal;
	normal.Normal(point1, point2, point3);
	normal.Normalize();
	ret.w = normal.Dot(point1);
	ret.x = normal.x;
	ret.y = normal.y;
	ret.z = normal.z;
	return ret;
}

inline tvector4 vector4(const tvector4 & pt)
{
	tvector4 ret;
	ret.x = pt.x; ret.y = pt.y; ret.z = pt.z; ret.w = pt.w;
	return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct tmatrix
{

public:
	union
	{
		float m[4][4];
		float m16[16];
		struct 
		{
			tvector4 right,up,dir,position;
		} V4;
	};
public:
	// constructor
	tmatrix();
	tmatrix(const tvector3& right, const tvector3& up, const tvector3& dir, const tvector3& pos);

	tmatrix( const float * );
	tmatrix( const tmatrix& mat );

	tmatrix( float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33 );
	// casting operators
	operator float* ();
	operator const float* () const;

	// assignment operators
	tmatrix& operator *= ( const tmatrix& );
	tmatrix& operator += ( const tmatrix& );
	tmatrix& operator -= ( const tmatrix& );
	tmatrix& operator *= ( float );
	tmatrix& operator /= ( float );

	// unary operators
	tmatrix operator + () const;
	tmatrix operator - () const;

	// binary operators
	tmatrix operator * ( const tmatrix& mat) const;
	tmatrix operator + ( const tmatrix& mat) const;
	tmatrix operator - ( const tmatrix& mat) const;
	tmatrix operator * ( float ) const;
	tmatrix operator / ( float ) const;

	// Comparaison operators
	bool operator == ( const tmatrix& mat) const;
	bool operator != ( const tmatrix& mat) const;

	// Casting function
	void operator = (const tquaternion & q);

	// Member Functions
	inline void Set( const float m11, const float m12, const float m13, const float m14,
		const float m21, const float m22, const float m23, const float m24,
		const float m31, const float m32, const float m33, const float m34,
		const float m41, const float m42, const float m43, const float m44);

	inline void Set(tmatrix &matrix);

	tvector3 GetCol(tulong col);
	tvector3 GetLine(tulong line);

	void SetCol(tulong col, const tvector3& vec);


	void SetLine(tulong line, const tvector3& vec);



	bool    IsIdentity() const;
	void Identity();

	void Multiply( const tmatrix &matrix);
	void Multiply( const tmatrix &m1, const tmatrix &m2 );

	void Transpose();
	void Transpose( const tmatrix &matrix );

	float GetDeterminant();
	float Inverse(const tmatrix &srcMatrix, bool affine = false );
	float Inverse(bool affine=false);

	void Scaling(const float sx, const float sy, const float sz );
	void Scaling(tvector3 scale);
	void Translation(const float x, const float y, const float z );
	void Translation(const tvector3 & v);
	void Lerp(const tmatrix &matA, const tmatrix &matB, float t);



	tvector3 GetTranslation();
	// remove translation
	void NoTrans();

	void RotationX(const float angle );
	void RotationY(const float angle );
	void RotationZ(const float angle );
	void RotationAxis(const tvector3 &v, float angle );
	void RotationQuaternion(const tquaternion &q);
	void RotationYawPitchRoll(const float yaw, const float pitch, const float roll );
	void Transformation(const tvector3 &scalingCenter, const tquaternion &scalingRotation, const tvector3 &scaling,
		const tvector3 &rotationCenter, const tquaternion &rotation, const tvector3 &translation);
	void AffineTransformation(float scaling, const tvector3 &rotationCenter, const tquaternion &rotation, const tvector3 &translation);
	void LookAtRH(const tvector3 &eye, const tvector3 &at, const tvector3 &up );
	void LookAtLH(const tvector3 &eye, const tvector3 &at, const tvector3 &up );
	void LookAt(const tvector3 &eye, const tvector3 &at, const tvector3 &up );

	void PerspectiveRH(const float w, const float h, const float zn, const float zf );
	void PerspectiveLH(const float w, const float h, const float zn, const float zf );
	void PerspectiveFovRH(const float fovy, const float aspect, const float zn, const float zf );
	void PerspectiveFovLH(const float fovy, const float aspect, const float zn, const float zf );
	void PerspectiveFovLH2(const float fovy, const float aspect, const float zn, const float zf );

	void PerspectiveOffCenterRH(const float l, const float r, const float b, const float t, const float zn, const float zf );
	void PerspectiveOffCenterLH(const float l, const float r, const float b, const float t, const float zn, const float zf );
	void OrthoRH(const float w, const float h, const float zn, const float zf );
	void OrthoLH(const float w, const float h, const float zn, const float zf );
	void OrthoOffCenterRH(const float l, const float r, const float b, const float t, const float zn, const float zf );
	void OrthoOffCenterLH(const float l, const float r, const float b, const float t, const float zn, const float zf );
	/*
	void Shadow(const tvector4 &light, const tplane &plane );
	void Reflect(const tplane &plane );
	*/
	void OrthoNormalize();
	static tmatrix identity;

	inline	tmatrix&	PreMul(const tmatrix& aM);
	inline	tmatrix&	PostMul(const tmatrix& aM);
	inline	tmatrix&	PostRotate(const tvector3& aAxis, const float aAngle);
	inline	tmatrix&	PreRotate(const tvector3& aAxis, const float aAngle);

} tmatrix ;

///////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct tcolor
{

public:
	float r, g, b, a;

public:
	// Constructor
	tcolor();
	tcolor(float v) { r = g = b = a = v; }
	tcolor( tulong argb );
	tcolor( const float * );
	tcolor( float r, float g, float b, float a );

	inline void SwapRB()
	{
		float tmpc = r;
		r = b;
		b = tmpc;
	}

	// casting
	operator tulong () const;
	operator float* ();
	operator const float* () const;

	// assignment operators
	tcolor& operator += ( const tcolor& );
	tcolor& operator -= ( const tcolor& );
	tcolor& operator *= ( float );
	tcolor& operator /= ( float );

	// unary operators
	tcolor operator + () const;
	tcolor operator - () const;

	// binary operators
	tcolor operator + ( const tcolor& ) const;
	tcolor operator - ( const tcolor& ) const;
	tcolor operator * ( float ) const;
	tcolor operator / ( float ) const;

	// Comparaison operators
	bool  operator == ( const tcolor& ) const;
	bool  operator != ( const tcolor& ) const;

	// Member Functions
	bool IsVeryClose(tcolor &srcQuat)
	{
		if(MathFloatIsVeryClose(r, srcQuat.r) &&
			MathFloatIsVeryClose(g, srcQuat.g)  &&
			MathFloatIsVeryClose(b, srcQuat.b) &&
			MathFloatIsVeryClose(a, srcQuat.a))
			return true;
		return false;
	}

	tulong    ConvToBGR() const;
	tulong    ConvToBGRA() const;
	tulong	  ConvToRGBA() const;
	void        Negative()
	{
		r = -r;
		g = -g;
		b = -b;
		a = -a;
	}

	void        Add(const tcolor &pC1)
	{
		r += pC1.r;
		g += pC1.g;
		b += pC1.b;
		a += pC1.a;
	}

	void        Subtract(const tcolor &pC1)
	{
		r -= pC1.r;
		g -= pC1.g;
		b -= pC1.b;
		a -= pC1.a;
	}

	void        Scale(const float s)
	{
		r *= s;
		g *= s;
		b *= s;
		a *= s;
	}

	void        Modulate(const tcolor &pC1)
	{
		r *= pC1.r;
		g *= pC1.g;
		b *= pC1.b;
		a *= pC1.a;
	}

	void        Lerp(const tcolor &pC1, const tcolor &pC2, const float s)
	{
		r = pC1.r + s * (pC2.r - pC1.r);
		g = pC1.g + s * (pC2.g - pC1.g);
		b = pC1.b + s * (pC2.b - pC1.b);
		a = pC1.a + s * (pC2.a - pC1.a);
	}

	void        AdjustSaturation(const float s)
	{
		float grey = r * 0.2125f + g * 0.7154f + b * 0.0721f;

		r = grey + s * (r - grey);
		g = grey + s * (g - grey);
		b = grey + s * (b - grey);
	}

	void        AdjustContrast(const float c)
	{

		r = 0.5f + c * (r - 0.5f);
		g = 0.5f + c * (g - 0.5f);
		b = 0.5f + c * (b - 0.5f);

	}
	static const tcolor black;
	static const tcolor white;
	static const tcolor green;
	static const tcolor red;
	static const tcolor blue;

} tcolor;



struct tquaternion
{

public:
	float x, y, z, w;


public:
	tquaternion();
	tquaternion( const float * );
	tquaternion( float x, float y, float z, float w );
	tquaternion( float yaw, float pitch, float roll);
	tquaternion( const tmatrix &mat);

	void ToEuler(float &heading, float &attitude, float &bank);
	//        tquaternion(tquaternion & p_q);

	// casting
	operator float* ();
	operator const float* () const;

	// assignment operators
	tquaternion& operator = (const tquaternion& rkQ);
	tquaternion& operator += ( const tquaternion& );
	tquaternion& operator -= ( const tquaternion& );
	tquaternion& operator *= ( const tquaternion& );
	tquaternion& operator *= ( float );
	tquaternion& operator /= ( float );

	// unary operators
	tquaternion  operator + () const;
	tquaternion  operator - () const;

	// binary operators
	tquaternion operator + ( const tquaternion& ) const;
	tquaternion operator - ( const tquaternion& ) const;
	tquaternion operator * ( const tquaternion& ) const;
	tquaternion operator * ( float ) const;
	tquaternion operator / ( float ) const;

	friend tquaternion operator * (float, const tquaternion& );

	// Comparaison operators
	bool operator == ( const tquaternion& ) const;
	bool operator != ( const tquaternion& ) const;

	// Member Functions
	//float Length() const;
	//float LengthSq() const;
	float Dot( const tquaternion &q) const;

	void Identity();
	bool    IsIdentity() const
	{
		return ((x==0)&&(y==0)&&(z==0)&&(w==1));
	}
	bool IsVeryClose(tquaternion &srcQuat);
	/*
	void ToAxisAngle(tvector3 &pAxis, float &pAngle );

	void RotationMatrix(const tmatrix &pM);
	*/
	void RotationAxis(const tvector3 &v, const float angle );
	/*
	void RotationYawPitchRoll(const float yaw, const float pitch, const float roll );
	*/
	void Multiply(const tquaternion &q );
	void Multiply(const tquaternion &q1, const tquaternion &q2 );

	void Normalize();
	float Norm() const;
	//void Normalize(const tquaternion &q );

	bool Inverse();
	bool Inverse(const tquaternion &q );

	// Conjugate of a quaternion
	void UnitInverse();
	void UnitInverse(const tquaternion &q );
	/*
	void Ln();
	void Ln(const tquaternion &q );
	void LnDif(const tquaternion &q );
	void MakeClosest(const tquaternion &q );


	void Exp();
	void Exp(const tquaternion &q );
	*/
	void Slerp(const tquaternion &q1, const tquaternion &q2, float t );

	static tquaternion identity;
	/*
	static void SquadSetup(const tquaternion &q1, const tquaternion &q2, const tquaternion &q3, tquaternion &A, tquaternion &B);
	static void ComputeAParam(const tquaternion &prev, const tquaternion &current, const tquaternion &next, tquaternion &A);
	void SquadWithSetup(const tquaternion &prev, const tquaternion &current, const tquaternion &next, const tquaternion &nextnext, float t );
	void Squad(const tquaternion &q1, const tquaternion &q2, const tquaternion &q3, const tquaternion &q4, float t );
	void BaryCentric(const tquaternion &q1, const tquaternion &q2, const tquaternion &q3, float f, float g );
	*/
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "ZBaseMaths.inl"

#endif
