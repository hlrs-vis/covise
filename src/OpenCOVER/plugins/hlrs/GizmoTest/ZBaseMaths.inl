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



// Inlines ////////////////////////////////////////////////////////////////////////////////////////

inline tvector2::tvector2()
{

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2::tvector2( const float *pf )
{
	x = pf[0];
	y = pf[1];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2::tvector2( float fx, float fy )
{
	x = fx;
	y = fy;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// casting

inline tvector2::operator float* ()
{
	return (float *) &x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2::operator const float* () const
{
	return (const float *) &x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// assignment operators

inline tvector2& tvector2::operator += ( const tvector2& v )
{
	x += v.x;
	y += v.y;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2& tvector2::operator -= ( const tvector2& v )
{
	x -= v.x;
	y -= v.y;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2& tvector2::operator *= ( float f )
{
	x *= f;
	y *= f;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2& tvector2::operator /= ( float f )
{
	float fInv = 1.0f / f;
	x *= fInv;
	y *= fInv;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// unary operators

inline tvector2 tvector2::operator + () const
{
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2 tvector2::operator - () const
{
	return tvector2(-x, -y);
}
///////////////////////////////////////////////////////////////////////////////////////////////////


inline tvector4 tvector4::operator + () const
{
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector4 tvector4::operator - () const
{
	return vector4(-x, -y, -z, -w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// binary operators

inline tvector2 tvector2::operator + ( const tvector2& v ) const
{
	return tvector2(x + v.x, y + v.y);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2 tvector2::operator - ( const tvector2& v ) const
{
	return tvector2(x - v.x, y - v.y);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2 tvector2::operator * ( float f ) const
{
	return tvector2(x * f, y * f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2 tvector2::operator / ( float f ) const
{
	float fInv = 1.0f / f;
	return tvector2(x * fInv, y * fInv);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector2 operator * ( float f, const tvector2& v )
{
	return tvector2(f * v.x, f * v.y);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tvector2::operator == ( const tvector2& v ) const
{
	return x == v.x && y == v.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tvector2::operator != ( const tvector2& v ) const
{
	return x != v.x || y != v.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector2::Length( ) const
{
	return MathSqrt( (x * x) + (y * y));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector2::LengthSq( ) const
{
	return (x * x) + (y * y);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector2::Dot( const tvector2 &v) const
{
	return (x * v.x) + (y * v.y);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector2::CCW( const tvector2 &v ) const
{
	return (x * v.y) - (y * v.x);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Add(const tvector2 &v)
{
	x += v.x;
	y += v.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Add(const tvector2 &v1, const tvector2 &v2)
{
	x = v1.x + v2.x;
	y = v1.y + v2.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Subtract(const tvector2 &v)
{
	x -= v.x;
	y -= v.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Subtract(const tvector2 &v1, const tvector2 &v2)
{
	x = v1.x - v2.x;
	y = v1.y - v2.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Minimize(const tvector2 &v)
{
	x = x < v.x ? x : v.x;
	y = y < v.y ? y : v.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Minimize(const tvector2 &v1, const tvector2 &v2)
{
	x = v1.x < v2.x ? v1.x : v2.x;
	y = v1.y < v2.y ? v1.y : v2.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Maximize(const tvector2 &v)
{
	x = x > v.x ? x : v.x;
	y = y > v.y ? y : v.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Maximize(const tvector2 &v1, const tvector2 &v2)
{
	x = v1.x > v2.x ? v1.x : v2.x;
	y = v1.y > v2.y ? v1.y : v2.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Scale(const float s)
{
	x *= s;
	y *= s;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Lerp(const tvector2 &v1, const tvector2 &v2, float s )
{
	x = v1.x + s * (v2.x - v1.x);
	y = v1.y + s * (v2.y - v1.y);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Normalize()
{
	float lenght = MathSqrt( (x * x) + (y * y) );

	if( lenght != 0.0f )
	{
		x /= lenght;
		y /= lenght;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::Normalize(const tvector2 &v)
{
	float lenght = MathSqrt( (v.x * v.x) + (v.y * v.y) );

	if (lenght != 0.0f)
	{
		x = v.x / lenght;
		y = v.y / lenght;
	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::BaryCentric(const tvector2 &/*v1*/, const tvector2 &/*v2*/, const tvector2 &/*v3*/, float /*f*/, float /*g*/)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::TransformPoint(const tmatrix &matrix )
{
	tvector2 out;

	out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + matrix.m[2][0];
	out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + matrix.m[2][1];

	x = out.x;
	y = out.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::TransformPoint(const tvector2 &v, const tmatrix &matrix )
{
	x = v.x * matrix.m[0][0] + v.y * matrix.m[1][0] + matrix.m[2][0];
	y = v.x * matrix.m[0][1] + v.y * matrix.m[1][1] + matrix.m[2][1];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::TransformVector(const tmatrix &matrix )
{
	tvector2 out;

	out.x = x * matrix.m[0][0] + y * matrix.m[1][0];
	out.y = x * matrix.m[0][1] + y * matrix.m[1][1];

	x = out.x;
	y = out.y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector2::TransformVector(const tvector2 &v, const tmatrix &matrix )
{
	x = v.x * matrix.m[0][0] + v.y * matrix.m[1][0];
	y = v.x * matrix.m[0][1] + v.y * matrix.m[1][1];
}

///////////////////////////////////////////////////////////////////////////////////////////////////











///////////////////////////////////////////////////////////////////////////////////////////////////
// constructor

inline tvector3::tvector3()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3::tvector3( const float * coord)
{
	x = coord[0];
	y = coord[1];
	z = coord[2];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3::tvector3( float p_x, float p_y, float p_z )
{
	x = p_x;
	y = p_y;
	z = p_z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Casting

inline tvector3::operator float * ()
{
	return (float *) &x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3::operator const float* () const
{
	return (const float *) &x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// assignment operators

inline tvector3& tvector3::operator += ( const tvector3& v )
{
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3& tvector3::operator -= ( const tvector3& v )
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3& tvector3::operator *= ( float f )
{
	x *= f;
	y *= f;
	z *= f;
	return *this;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector4& tvector4::operator *= ( float f )
{
	x *= f;
	y *= f;
	z *= f;
	w *= f;
	return *this;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3& tvector3::operator *= ( tvector3 vec )
{
	x *= vec.x;
	y *= vec.y;
	z *= vec.z;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3& tvector3::operator /= ( float f )
{
	float fInv = 1.0f / f;
	x *= fInv;
	y *= fInv;
	z *= fInv;
	return *this;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector4& tvector4::operator /= ( float f )
{
	float fInv = 1.0f / f;
	x *= fInv;
	y *= fInv;
	z *= fInv;
	w *= fInv;
	return *this;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3& tvector3::operator /= ( tvector3 vec )
{
	x /= vec.x;
	y /= vec.y;
	z /= vec.z;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// unary operators

inline tvector3 tvector3::operator + () const
{
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 tvector3::operator - () const
{
	return tvector3(-x, -y, -z);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// binary operators

inline tvector3 tvector3::operator + ( const tvector3& v ) const
{
	return tvector3(x + v.x, y + v.y, z + v.z);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 tvector3::operator - ( const tvector3& v ) const
{
	return tvector3(x - v.x, y - v.y, z - v.z);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 tvector3::operator * ( float f ) const
{
	return tvector3(x * f, y * f, z * f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector4 tvector4::operator * ( float f ) const
{
	return vector4(x * f, y * f, z * f, w * f);
}
///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 tvector3::operator * ( tvector3 vec ) const
{
	return tvector3(x * vec.x, y * vec.y, z * vec.z);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 tvector3::operator / ( float f ) const
{
	float fInv = 1.0f / f;
	return tvector3(x * fInv, y * fInv, z * fInv);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 tvector3::operator / ( tvector3 vec ) const
{
	return tvector3(x / vec.x, y / vec.y, z / vec.z);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 operator * ( float f, const tvector3& v )
{
	return tvector3(f * v.x, f * v.y, f * v.z);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tvector3::operator == ( const tvector3& v ) const
{
	return x == v.x && y == v.y && z == v.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tvector3::operator != ( const tvector3& v ) const
{
	return x != v.x || y != v.y || z != v.z;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

tvector3 vecLimitDeviationAngleUtility (const bool insideOrOutside,
										const tvector3& source,
										const float cosineOfConeAngle,
										const tvector3& basis);
///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 limitMaxDeviationAngle (const tvector3& source,
										const float cosineOfConeAngle,
										const tvector3& basis)
{
	return vecLimitDeviationAngleUtility (true, // force source INSIDE cone
		source,
		cosineOfConeAngle,
		basis);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float DotProduct (const tvector3& v1, const tvector3& v2)
{
	return v1.x*v2.x + v1.y * v2.y + v1.z*v2.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 CrossProduct (const tvector3& v1, const tvector3& v2)
{
	tvector3 result;

	result.x = v1.y * v2.z - v1.z * v2.y;
	result.y = v1.z * v2.x - v1.x * v2.z;
	result.z = v1.x * v2.y - v1.y * v2.x;

	return result;
}
///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3::tvector3(const tvector3& v1,  const tvector3& v2)
{
	x = v1.x - v2.x;
	y = v1.y - v2.y;
	z = v1.z - v2.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tvector3::IsVeryClose(tvector3 & srcVec)
{
	if(MathFloatIsVeryClose(x, srcVec.x) && MathFloatIsVeryClose(y, srcVec.y)  && MathFloatIsVeryClose(z, srcVec.z))
	{
		return true;
	}

	return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector3::Length( ) const
{
	return MathSqrt( (x * x) + (y * y) + (z * z) );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector3::LengthSq( ) const
{
	return (x * x) + (y * y) + (z * z);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector3::Dot( const tvector3 &v) const
{
	return (x * v.x) + (y * v.y) + (z * v.z);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Cross(const tvector3 &v)
{
	tvector3 res;

	res.x = y * v.z - z * v.y;
	res.y = z * v.x - x * v.z;
	res.z = x * v.y - y * v.x;

	x = res.x;
	y = res.y;
	z = res.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Cross(const tvector3 &v1, const tvector3 &v2)
{
	x = v1.y * v2.z - v1.z * v2.y;
	y = v1.z * v2.x - v1.x * v2.z;
	z = v1.x * v2.y - v1.y * v2.x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Add(const tvector3 &v)
{
	x += v.x;
	y += v.y;
	z += v.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Add(const tvector3 &v1, const tvector3 &v2)
{
	x = v1.x + v2.x;
	y = v1.y + v2.y;
	z = v1.z + v2.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Subtract(const tvector3 &v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Subtract(const tvector3 &v1, const tvector3 &v2)
{
	x = v1.x - v2.x;
	y = v1.y - v2.y;
	z = v1.z - v2.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Minimize(const tvector3 &v)
{
	x = x < v.x ? x : v.x;
	y = y < v.y ? y : v.y;
	z = z < v.z ? z : v.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Minimize(const tvector3 &v1, const tvector3 &v2)
{
	x = v1.x < v2.x ? v1.x : v2.x;
	y = v1.y < v2.y ? v1.y : v2.y;
	z = v1.z < v2.z ? v1.z : v2.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Maximize(const tvector3 &v)
{
	x = x > v.x ? x : v.x;
	y = y > v.y ? y : v.y;
	z = z > v.z ? z : v.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Maximize(const tvector3 &v1, const tvector3 &v2)
{
	x = v1.x > v2.x ? v1.x : v2.x;
	y = v1.y > v2.y ? v1.y : v2.y;
	z = v1.z > v2.z ? v1.z : v2.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Scale(const float s)
{
	x *= s;
	y *= s;
	z *= s;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Lerp(const tvector3 &v1, const tvector3 &v2, float s )
{
	x = v1.x + s * (v2.x - v1.x);
	y = v1.y + s * (v2.y - v1.y);
	z = v1.z + s * (v2.z - v1.z);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Normal(const tvector3 & point1, const tvector3 & point2, const tvector3 & point3)
{
	tvector3 tmp1 = point1 - point3;
	tvector3 tmp2 = point2 - point3;
	this->Cross(tmp1, tmp2);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline float InvSqrt_Lomont(float x)
{
	float xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f375a84 - (i>>1); // hidden initial guess, fast
	x = *(float*)&i;
	x = x*(1.5f-xhalf*x*x);
	return x;
}
inline void tvector3::Normalize()
{

	float oneOlength = 1.f/MathSqrt( (x * x) + (y * y) + (z * z) );
	//float oneOlength = InvSqrt_Lomont( (x * x) + (y * y) + (z * z) );
	x *= oneOlength;
	y *= oneOlength;
	z *= oneOlength;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::Normalize(const tvector3 &v)
{
	float oneOlength = 1.f/MathSqrt( (v.x * v.x) + (v.y * v.y) + (v.z * v.z) );
	//float oneOlength = InvSqrt_Lomont( (v.x * v.x) + (v.y * v.y) + (v.z * v.z) );

	x = v.x * oneOlength;
	y = v.y * oneOlength;
	z = v.z * oneOlength;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::BaryCentric(const tvector3 &/*v1*/, const tvector3 &/*v2*/, const tvector3 &/*v3*/, float /*f*/, float /*g*/)
{

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::TransformPoint(const tmatrix &matrix )
{
	tvector3 out;

	out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + z * matrix.m[2][0] + matrix.m[3][0];
	out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + z * matrix.m[2][1] + matrix.m[3][1];
	out.z = x * matrix.m[0][2] + y * matrix.m[1][2] + z * matrix.m[2][2] + matrix.m[3][2];

	x = out.x;
	y = out.y;
	z = out.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::TransformPoint(const tvector3 &v, const tmatrix &matrix )
{
	x = v.x * matrix.m[0][0] + v.y * matrix.m[1][0] + v.z * matrix.m[2][0] + matrix.m[3][0];
	y = v.x * matrix.m[0][1] + v.y * matrix.m[1][1] + v.z * matrix.m[2][1] + matrix.m[3][1];
	z = v.x * matrix.m[0][2] + v.y * matrix.m[1][2] + v.z * matrix.m[2][2] + matrix.m[3][2];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::TransformVector(const tmatrix &matrix )
{
	tvector3 out;

	out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + z * matrix.m[2][0];
	out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + z * matrix.m[2][1];
	out.z = x * matrix.m[0][2] + y * matrix.m[1][2] + z * matrix.m[2][2];

	x = out.x;
	y = out.y;
	z = out.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::TransformVector(const tvector3 &v, const tmatrix &matrix )
{
	x = v.x * matrix.m[0][0] + v.y * matrix.m[1][0] + v.z * matrix.m[2][0];
	y = v.x * matrix.m[0][1] + v.y * matrix.m[1][1] + v.z * matrix.m[2][1];
	z = v.x * matrix.m[0][2] + v.y * matrix.m[1][2] + v.z * matrix.m[2][2];
}


///////////////////////////////////////////////////////////////////////////////////////////////////
inline tvector3 tvector3::truncateLength (const float maxLength) const
{
	const float maxLengthSquared = maxLength * maxLength;
	const float vecLengthSquared = this->lengthSquared ();
	if (vecLengthSquared <= maxLengthSquared)
		return *this;
	else
		return (*this) * (maxLength / (float)sqrt (vecLengthSquared));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector3::set(float fx,float fy,float fz)
{
	this->x=fx;
	this->y=fy;
	this->z=fz;
}




// Inlines ////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructors

inline tmatrix::tmatrix()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix::tmatrix( const float* pf )
{
	memcpy(&m[0][0], pf, sizeof(tmatrix));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix::tmatrix( const tmatrix& mat )
{
	memcpy(&m[0][0], &mat, sizeof(tmatrix));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix::tmatrix(float m11, float m12, float m13, float m14,
						float m21, float m22, float m23, float m24,
						float m31, float m32, float m33, float m34,
						float m41, float m42, float m43, float m44)
{
	m[0][0]=m11; m[0][1]=m12; m[0][2]=m13; m[0][3]=m14;
	m[1][0]=m21; m[1][1]=m22; m[1][2]=m23; m[1][3]=m24;
	m[2][0]=m31; m[2][1]=m32; m[2][2]=m33; m[2][3]=m34;
	m[3][0]=m41; m[3][1]=m42; m[3][2]=m43; m[3][3]=m44;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// casting operators

inline tmatrix::operator float* ()
{
	return (float *) &m[0][0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix::operator const float* () const
{
	return (const float *) &m[0][0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// assignment operators

inline tmatrix& tmatrix::operator *= ( const tmatrix& mat )
{
	tmatrix tmpMat;
	tmpMat = *this;
	tmpMat.Multiply(mat);
	*this = tmpMat;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix& tmatrix::operator += ( const tmatrix& mat )
{
	m[0][0] += mat.m[0][0] ; m[0][1] += mat.m[0][1] ; m[0][2] += mat.m[0][2] ; m[0][3] += mat.m[0][3];
	m[1][0] += mat.m[1][0] ; m[1][1] += mat.m[1][1] ; m[1][2] += mat.m[1][2] ; m[1][3] += mat.m[1][3];
	m[2][0] += mat.m[2][0] ; m[2][1] += mat.m[2][1] ; m[2][2] += mat.m[2][2] ; m[2][3] += mat.m[2][3];
	m[3][0] += mat.m[3][0] ; m[3][1] += mat.m[3][1] ; m[3][2] += mat.m[3][2] ; m[3][3] += mat.m[3][3];

	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix& tmatrix::operator -= ( const tmatrix& mat )
{
	m[0][0] -= mat.m[0][0] ; m[0][1] -= mat.m[0][1] ; m[0][2] -= mat.m[0][2] ; m[0][3] -= mat.m[0][3];
	m[1][0] -= mat.m[1][0] ; m[1][1] -= mat.m[1][1] ; m[1][2] -= mat.m[1][2] ; m[1][3] -= mat.m[1][3];
	m[2][0] -= mat.m[2][0] ; m[2][1] -= mat.m[2][1] ; m[2][2] -= mat.m[2][2] ; m[2][3] -= mat.m[2][3];
	m[3][0] -= mat.m[3][0] ; m[3][1] -= mat.m[3][1] ; m[3][2] -= mat.m[3][2] ; m[3][3] -= mat.m[3][3];

	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix& tmatrix::operator *= ( float f )
{
	m[0][0] *= f ; m[0][1] *= f ; m[0][2] *= f ; m[0][3] *= f;
	m[1][0] *= f ; m[1][1] *= f ; m[1][2] *= f ; m[1][3] *= f;
	m[2][0] *= f ; m[2][1] *= f ; m[2][2] *= f ; m[2][3] *= f;
	m[3][0] *= f ; m[3][1] *= f ; m[3][2] *= f ; m[3][3] *= f;

	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix& tmatrix::operator /= ( float f )
{
	float fInv = 1.0f / f;

	m[0][0] *= fInv ; m[0][1] *= fInv ; m[0][2] *= fInv ; m[0][3] *= fInv;
	m[1][0] *= fInv ; m[1][1] *= fInv ; m[1][2] *= fInv ; m[1][3] *= fInv;
	m[2][0] *= fInv ; m[2][1] *= fInv ; m[2][2] *= fInv ; m[2][3] *= fInv;
	m[3][0] *= fInv ; m[3][1] *= fInv ; m[3][2] *= fInv ; m[3][3] *= fInv;

	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// unary operators

inline tmatrix tmatrix::operator + () const
{
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix tmatrix::operator - () const
{
	return tmatrix(   -m[0][0], -m[0][1], -m[0][2], -m[0][3],
		-m[1][0], -m[1][1], -m[1][2], -m[1][3],
		-m[2][0], -m[2][1], -m[2][2], -m[2][3],
		-m[3][0], -m[3][1], -m[3][2], -m[3][3]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// binary operators

inline tmatrix tmatrix::operator * ( const tmatrix& mat ) const
{
	tmatrix matT;
	matT.Multiply(*this, mat);
	return matT;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix tmatrix::operator + ( const tmatrix& mat ) const
{
	return tmatrix(   m[0][0] + mat.m[0][0], m[0][1] + mat.m[0][1], m[0][2] + mat.m[0][2], m[0][3] + mat.m[0][3],
		m[1][0] + mat.m[1][0], m[1][1] + mat.m[1][1], m[1][2] + mat.m[1][2], m[1][3] + mat.m[1][3],
		m[2][0] + mat.m[2][0], m[2][1] + mat.m[2][1], m[2][2] + mat.m[2][2], m[2][3] + mat.m[2][3],
		m[3][0] + mat.m[3][0], m[3][1] + mat.m[3][1], m[3][2] + mat.m[3][2], m[3][3] + mat.m[3][3]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix tmatrix::operator - ( const tmatrix& mat ) const
{
	return tmatrix(m[0][0] - mat.m[0][0], m[0][1] - mat.m[0][1], m[0][2] - mat.m[0][2], m[0][3] - mat.m[0][3],
		m[1][0] - mat.m[1][0], m[1][1] - mat.m[1][1], m[1][2] - mat.m[1][2], m[1][3] - mat.m[1][3],
		m[2][0] - mat.m[2][0], m[2][1] - mat.m[2][1], m[2][2] - mat.m[2][2], m[2][3] - mat.m[2][3],
		m[3][0] - mat.m[3][0], m[3][1] - mat.m[3][1], m[3][2] - mat.m[3][2], m[3][3] - mat.m[3][3]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix tmatrix::operator * ( float f ) const
{
	return tmatrix(m[0][0] * f, m[0][1] * f, m[0][2] * f, m[0][3] * f,
		m[1][0] * f, m[1][1] * f, m[1][2] * f, m[1][3] * f,
		m[2][0] * f, m[2][1] * f, m[2][2] * f, m[2][3] * f,
		m[3][0] * f, m[3][1] * f, m[3][2] * f, m[3][3] * f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix tmatrix::operator / ( float f ) const
{
	float fInv = 1.0f / f;
	return tmatrix(m[0][0] * fInv, m[0][1] * fInv, m[0][2] * fInv, m[0][3] * fInv,
		m[1][0] * fInv, m[1][1] * fInv, m[1][2] * fInv, m[1][3] * fInv,
		m[2][0] * fInv, m[2][1] * fInv, m[2][2] * fInv, m[2][3] * fInv,
		m[3][0] * fInv, m[3][1] * fInv, m[3][2] * fInv, m[3][3] * fInv);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix operator * ( float f, const tmatrix& mat )
{
	return tmatrix(f * mat.m[0][0], f * mat.m[0][1], f * mat.m[0][2], f * mat.m[0][3],
		f * mat.m[1][0], f * mat.m[1][1], f * mat.m[1][2], f * mat.m[1][3],
		f * mat.m[2][0], f * mat.m[2][1], f * mat.m[2][2], f * mat.m[2][3],
		f * mat.m[3][0], f * mat.m[3][1], f * mat.m[3][2], f * mat.m[3][3]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tmatrix::operator == ( const tmatrix& mat ) const
{
	return 0 == memcmp(this, &mat, sizeof(tmatrix));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tmatrix::operator != ( const tmatrix& mat ) const
{
	return 0 != memcmp(this, &mat, sizeof(tmatrix));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Set(
						 const float m11, const float m12, const float m13, const float m14,
						 const float m21, const float m22, const float m23, const float m24,
						 const float m31, const float m32, const float m33, const float m34,
						 const float m41, const float m42, const float m43, const float m44)
{
	m[0][0]=m11; m[0][1]=m12; m[0][2]=m13; m[0][3]=m14;
	m[1][0]=m21; m[1][1]=m22; m[1][2]=m23; m[1][3]=m24;
	m[2][0]=m31; m[2][1]=m32; m[2][2]=m33; m[2][3]=m34;
	m[3][0]=m41; m[3][1]=m42; m[3][2]=m43; m[3][3]=m44;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Set(tmatrix &matrix)
{
	m[0][0] = matrix.m[0][0] ; m[0][1] = matrix.m[0][1] ; m[0][2] = matrix.m[0][2] ; m[0][3] = matrix.m[0][3];
	m[1][0] = matrix.m[1][0] ; m[1][1] = matrix.m[1][1] ; m[1][2] = matrix.m[1][2] ; m[1][3] = matrix.m[1][3];
	m[2][0] = matrix.m[2][0] ; m[2][1] = matrix.m[2][1] ; m[2][2] = matrix.m[2][2] ; m[2][3] = matrix.m[2][3];
	m[3][0] = matrix.m[3][0] ; m[3][1] = matrix.m[3][1] ; m[3][2] = matrix.m[3][2] ; m[3][3] = matrix.m[3][3];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tmatrix::tmatrix(const tvector3& right, const tvector3& up, const tvector3& dir, const tvector3& pos)
{
	m[0][0] = right.x;
	m[0][1] = right.y;
	m[0][2] = right.z;
	m[0][3] = 0;

	m[1][0] = up.x;
	m[1][1] = up.y;
	m[1][2] = up.z;
	m[1][3] = 0;

	m[2][0] = dir.x;
	m[2][1] = dir.y;
	m[2][2] = dir.z;
	m[2][3] = 0;

	m[3][0] = pos.x;
	m[3][1] = pos.y;
	m[3][2] = pos.z;
	m[3][3] = 1;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::operator = (const tquaternion & q)
{
	float xx = q.x*q.x;
	float xy = q.x*q.y;
	float xz = q.x*q.z;
	float xw = q.x*q.w;

	float yy = q.y*q.y;
	float yz = q.y*q.z;
	float yw = q.y*q.w;

	float zz = q.z*q.z;
	float zw = q.z*q.w;

	m[0][0] = 1.0f-2.0f*(yy+zz);
	m[0][1] = 2.0f*(xy+zw);
	m[0][2] = 2.0f*(xz-yw);
	m[0][3] = 0.0f;

	m[1][0] = 2.0f*(xy-zw);
	m[1][1] = 1.0f-2.0f*(xx+zz);
	m[1][2] = 2.0f*(yz+xw);
	m[1][3] = 0.0f;

	m[2][0] = 2.0f*(xz+yw);
	m[2][1] = 2.0f*(yz-xw);
	m[2][2] = 1.0f-2.0f*(xx+yy);
	m[2][3] = 0.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = 0.0f;
	m[3][3] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool    tmatrix::IsIdentity() const
{
	return   (m[0][0] == 1.0f) && (m[0][1] == 0.0f) && (m[0][2] == 0.0f) && (m[0][3] == 0.0f) &&
		(m[0][0] == 0.0f) && (m[0][1] == 1.0f) && (m[0][2] == 0.0f) && (m[0][3] == 0.0f) &&
		(m[0][0] == 0.0f) && (m[0][1] == 0.0f) && (m[0][2] == 1.0f) && (m[0][3] == 0.0f) &&
		(m[0][0] == 0.0f) && (m[0][1] == 0.0f) && (m[0][2] == 0.0f) && (m[0][3] == 1.0f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
inline tvector3 tmatrix::GetCol(tulong col)
{
	return tvector3(m[0][col],m[1][col],m[2][col]);
}
inline tvector3 tmatrix::GetLine(tulong line)
{
	return tvector3(m[line][0],m[line][1],m[line][2]);
}


inline void tmatrix::SetCol(tulong col, const tvector3& vec)
{
	m[0][col] = vec.x;
	m[1][col] = vec.y;
	m[2][col] = vec.z;
	m[3][col] = 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::SetLine(tulong line, const tvector3& vec)
{
	m[line][0] = vec.x;
	m[line][1] = vec.y;
	m[line][2] = vec.z;
	m[line][3] = 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
inline void tmatrix::Identity()
{
	Set(1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1);
}
#ifdef WIN32
inline void SSE_MatrixF_x_MatrixF(const float *matA, const float *matB, float *result)
{
	__asm
	{
		mov         ecx, matA
			mov         edx, matB
			mov         eax, result

			movss       xmm0, [edx]
		movups      xmm1, [ecx]
		shufps      xmm0, xmm0, 0
			movss       xmm2, [edx+4]
		mulps       xmm0, xmm1
			shufps      xmm2, xmm2, 0
			movups      xmm3, [ecx+10h]
		movss       xmm7, [edx+8]
		mulps       xmm2, xmm3
			shufps      xmm7, xmm7, 0
			addps       xmm0, xmm2
			movups      xmm4, [ecx+20h]
		movss       xmm2, [edx+0Ch]
		mulps       xmm7, xmm4
			shufps      xmm2, xmm2, 0
			addps       xmm0, xmm7
			movups      xmm5, [ecx+30h]
		movss       xmm6, [edx+10h]
		mulps       xmm2, xmm5
			movss       xmm7, [edx+14h]
		shufps      xmm6, xmm6, 0
			addps       xmm0, xmm2
			shufps      xmm7, xmm7, 0
			movlps      [eax], xmm0
			movhps      [eax+8], xmm0
			mulps       xmm7, xmm3
			movss       xmm0, [edx+18h]
		mulps       xmm6, xmm1
			shufps      xmm0, xmm0, 0
			addps       xmm6, xmm7
			mulps       xmm0, xmm4
			movss       xmm2, [edx+24h]
		addps       xmm6, xmm0
			movss       xmm0, [edx+1Ch]
		movss       xmm7, [edx+20h]
		shufps      xmm0, xmm0, 0
			shufps      xmm7, xmm7, 0
			mulps       xmm0, xmm5
			mulps       xmm7, xmm1
			addps       xmm6, xmm0
			shufps      xmm2, xmm2, 0
			movlps      [eax+10h], xmm6
			movhps      [eax+18h], xmm6
			mulps       xmm2, xmm3
			movss       xmm6, [edx+28h]
		addps       xmm7, xmm2
			shufps      xmm6, xmm6, 0
			movss       xmm2, [edx+2Ch]
		mulps       xmm6, xmm4
			shufps      xmm2, xmm2, 0
			addps       xmm7, xmm6
			mulps       xmm2, xmm5
			movss       xmm0, [edx+34h]
		addps       xmm7, xmm2
			shufps      xmm0, xmm0, 0
			movlps      [eax+20h], xmm7
			movss       xmm2, [edx+30h]
		movhps      [eax+28h], xmm7
			mulps       xmm0, xmm3
			shufps      xmm2, xmm2, 0
			movss       xmm6, [edx+38h]
		mulps       xmm2, xmm1
			shufps      xmm6, xmm6, 0
			addps       xmm2, xmm0
			mulps       xmm6, xmm4
			movss       xmm7, [edx+3Ch]
		shufps      xmm7, xmm7, 0
			addps       xmm2, xmm6
			mulps       xmm7, xmm5
			addps       xmm2, xmm7
			movups      [eax+30h], xmm2
	}
}

#endif
inline void FPU_MatrixF_x_MatrixF(const float *a, const float *b, float *r)
{
	r[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]  + a[3]*b[12];
	r[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]  + a[3]*b[13];
	r[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10] + a[3]*b[14];
	r[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]*b[15];

	r[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]  + a[7]*b[12];
	r[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]  + a[7]*b[13];
	r[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10] + a[7]*b[14];
	r[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]*b[15];

	r[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8] + a[11]*b[12];
	r[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9] + a[11]*b[13];
	r[10]= a[8]*b[2] + a[9]*b[6] + a[10]*b[10]+ a[11]*b[14];
	r[11]= a[8]*b[3] + a[9]*b[7] + a[10]*b[11]+ a[11]*b[15];

	r[12]= a[12]*b[0]+ a[13]*b[4]+ a[14]*b[8] + a[15]*b[12];
	r[13]= a[12]*b[1]+ a[13]*b[5]+ a[14]*b[9] + a[15]*b[13];
	r[14]= a[12]*b[2]+ a[13]*b[6]+ a[14]*b[10]+ a[15]*b[14];
	r[15]= a[12]*b[3]+ a[13]*b[7]+ a[14]*b[11]+ a[15]*b[15];
}
inline void tmatrix::Multiply( const tmatrix &matrix)
{
	tmatrix tmp;
	tmp = *this;
	/*
	for(tulong l = 0; l < 4; l++)
	{
	for(tulong c = 0; c < 4; c++)
	{
	s = 0;
	for(tulong i = 0; i < 4; i++)
	{
	s += tmp.m[l][i] * matrix.m[i][c];
	}
	m[l][c] = s;
	}
	}
	*/
	FPU_MatrixF_x_MatrixF( (float*)&tmp, (float*)&matrix, (float*)this);
	//SSE_MatrixF_x_MatrixF( (float*)&tmp, (float*)&matrix, (float*)this);


}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Multiply( const tmatrix &m1, const tmatrix &m2 )
{
	/*
	float s;

	for(tulong l = 0; l < 4; l++)
	{
	for(tulong c = 0; c < 4; c++)
	{
	s = 0;
	for(tulong i = 0; i < 4; i++)
	{
	s += m1.m[l][i] * m2.m[i][c];
	}
	m[l][c] = s;
	}
	}
	*/
	FPU_MatrixF_x_MatrixF( (float*)&m1, (float*)&m2, (float*)this);
	//SSE_MatrixF_x_MatrixF( (float*)&m1, (float*)&m2, (float*)this);

}
///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Transpose()
{
	tmatrix tmpm;
	for (int l = 0; l < 4; l++)
	{
		for (int c = 0; c < 4; c++)
		{
			tmpm.m[l][c] = m[c][l];
		}
	}
	(*this) = tmpm;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Transpose( const tmatrix &matrix )
{
	for (int l = 0; l < 4; l++)
	{
		for (int c = 0; c < 4; c++)
		{
			m[l][c] = matrix.m[c][l];
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tmatrix::GetDeterminant()
{
	return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] +    m[0][2] * m[1][0] * m[2][1] -
		m[0][2] * m[1][1] * m[2][0] - m[0][1] * m[1][0] * m[2][2] -    m[0][0] * m[1][2] * m[2][1];
}


///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Scaling(const float sx, const float sy, const float sz )
{
	Set(sx, 0 , 0 , 0,
		0 , sy, 0 , 0,
		0 , 0 , sz, 0,
		0 , 0 , 0 , 1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Scaling(tvector3 s)
{
	Set(s.x, 0  , 0  , 0,
		0  , s.y, 0  , 0,
		0  , 0  , s.z, 0,
		0  , 0  , 0  , 1);

}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Translation matrix
//
//   1 0 0 0
//   0 1 0 0
//   0 0 1 0
//   x y z 1

inline void tmatrix::Translation(const float x, const float y, const float z )
{
	Set(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		x, y, z, 1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Translation(const tvector3 &v)
{
	Set( 1 ,  0 ,  0 , 0,
		0 ,  1 ,  0 , 0,
		0 ,  0 ,  1 , 0,
		v.x, v.y, v.z, 1 );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::NoTrans()
{
	m16[12]=0;
	m16[13]=0;
	m16[14]=0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector3 tmatrix::GetTranslation()
{
	return tvector3(m16[12],m16[13],m16[14]);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// X rotation matrix
//
//   1 0       0      0
//   0 cos(a)  sin(a) 0
//   0 -sin(a) cos(a) 0
//   0 0       0      1

inline void tmatrix::RotationX(const float angle )
{
	float c = (float)MathCos(angle);
	float s = (float)MathSin(angle);

	Set(1, 0 , 0, 0,
		0, c , s, 0,
		0, -s, c, 0,
		0, 0 , 0, 1);

}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Y rotation matrix
//
//      cos(a) 0 -sin(a) 0
//   0      1 0       0
//   sin(a) 0 cos(a)  0
//      0      0 0       1


inline void tmatrix::RotationY(const float angle )
{
	float c = (float)MathCos(angle);
	float s = (float)MathSin(angle);

	Set(c, 0, -s, 0,
		0, 1, 0 , 0,
		s, 0, c , 0,
		0, 0, 0 , 1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Z rotation matrix
//
//   cos(a)  sin(a) 0 0
//   -sin(a) cos(a) 0 0
//   0       0      1 0
//   0       0      0 1

inline void tmatrix::RotationZ(const float angle )
{
	float c = (float)MathCos(angle);
	float s = (float)MathSin(angle);

	Set(c , s, 0, 0,
		-s, c, 0, 0,
		0 , 0, 1, 0,
		0 , 0, 0, 1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::RotationAxis(const tvector3 & axis, float angle )
{
	float length2 = axis.LengthSq();
	if ( length2 == 0 )
	{
		Identity();
		return;
	}

	tvector3 n = axis / (float)MathSqrt(length2);
	float s = (float)MathSin(angle);
	float c = (float)MathCos(angle);
	float k = 1 - c;

	float xx = n.x * n.x * k + c;
	float yy = n.y * n.y * k + c;
	float zz = n.z * n.z * k + c;
	float xy = n.x * n.y * k;
	float yz = n.y * n.z * k;
	float zx = n.z * n.x * k;
	float xs = n.x * s;
	float ys = n.y * s;
	float zs = n.z * s;

	m[0][0] = xx;
	m[0][1] = xy + zs;
	m[0][2] = zx - ys;
	m[0][3] = 0;
	m[1][0] = xy - zs;
	m[1][1] = yy;
	m[1][2] = yz + xs;
	m[1][3] = 0;
	m[2][0] = zx + ys;
	m[2][1] = yz - xs;
	m[2][2] = zz;
	m[2][3] = 0;
	m[3][0] = 0;
	m[3][1] = 0;
	m[3][2] = 0;
	m[3][3] = 1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::RotationQuaternion(const tquaternion &q)
{
	float xx = q.x*q.x;
	float xy = q.x*q.y;
	float xz = q.x*q.z;
	float xw = q.x*q.w;

	float yy = q.y*q.y;
	float yz = q.y*q.z;
	float yw = q.y*q.w;

	float zz = q.z*q.z;
	float zw = q.z*q.w;

	m[0][0] = 1.0f-2.0f*(yy+zz);
	m[0][1] = 2.0f*(xy+zw);
	m[0][2] = 2.0f*(xz-yw);
	m[0][3] = 0.0f;

	m[1][0] = 2.0f*(xy-zw);
	m[1][1] = 1.0f-2.0f*(xx+zz);
	m[1][2] = 2.0f*(yz+xw);
	m[1][3] = 0.0f;

	m[2][0] = 2.0f*(xz+yw);
	m[2][1] = 2.0f*(yz-xw);
	m[2][2] = 1.0f-2.0f*(xx+yy);
	m[2][3] = 0.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = 0.0f;
	m[3][3] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::RotationYawPitchRoll(const float yaw, const float pitch, const float roll )
{
	float cy = (float)MathCos(yaw);
	float sy = (float)MathSin(yaw);

	float cp = (float)MathCos(pitch);
	float sp = (float)MathSin(pitch);

	float cr = (float)MathCos(roll);
	float sr = (float)MathSin(roll);

	float spsy = sp * sy;
	float spcy = sp * cy;

	m[0][0] = cr * cp;
	m[0][1] = sr * cp;
	m[0][2] = -sp;
	m[0][3] = 0;
	m[1][0] = cr * spsy - sr * cy;
	m[1][1] = sr * spsy + cr * cy;
	m[1][2] = cp * sy;
	m[1][3] = 0;
	m[2][0] = cr * spcy + sr * sy;
	m[2][1] = sr * spcy - cr * sy;
	m[2][2] = cp * cy;
	m[2][3] = 0;
	m[3][0] = 0;
	m[3][1] = 0;
	m[3][2] = 0;
	m[3][3] = 1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Transformation(const tvector3 &p_scalingCenter, const tquaternion &p_scalingRotation,const tvector3 &p_scaling,
									const tvector3 &p_rotationCenter, const tquaternion &p_rotation, const tvector3 &p_translation)
{
	tmatrix  rotMat;
	tmatrix  transMat;
	tmatrix  scaleMat;

	tmatrix  stretchRotMat;
	tmatrix  invStretchRotMat;

	tmatrix  rotationCenter;
	tmatrix  rotationCenterInv;

	tmatrix  scaleCenter;
	tmatrix  scaleCenterInv;

	rotMat.RotationQuaternion(p_rotation);
	transMat.Translation(p_translation);
	scaleMat.Scaling(p_scaling.x, p_scaling.y, p_scaling.z);

	stretchRotMat.RotationQuaternion(p_scalingRotation);
	invStretchRotMat.Inverse(stretchRotMat);

	rotationCenter.Translation(p_rotationCenter);
	rotationCenterInv.Inverse(rotationCenter,true);

	scaleCenter.Translation(p_scalingCenter);
	rotationCenterInv.Inverse(scaleCenter,true);

	*this = scaleCenterInv * invStretchRotMat * scaleMat * stretchRotMat * scaleCenter * rotationCenterInv * rotMat * rotationCenter * transMat;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::AffineTransformation(float p_scaling, const tvector3 &p_rotationCenter, const tquaternion &p_rotation, const tvector3 &p_translation)
{
	tmatrix  rotMat;
	tmatrix  transMat;
	tmatrix  scaleMat;

	tmatrix  stretchRotMat;
	tmatrix  invStretchRotMat;

	tmatrix  rotationCenter;
	tmatrix  rotationCenterInv;

	tmatrix  scaleCenter;
	tmatrix  scaleCenterInv;

	rotMat.RotationQuaternion(p_rotation);
	transMat.Translation(p_translation);
	scaleMat.Scaling(p_scaling, p_scaling, p_scaling);

	rotationCenter.Translation(p_rotationCenter);
	rotationCenterInv.Inverse(rotationCenter,true);

	*this = scaleMat * rotationCenterInv * rotMat * rotationCenter  * transMat;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::LookAtRH(const tvector3 &eye, const tvector3 &at, const tvector3 &up )
{
	/*
	D3DXMATRIX matRes;
	D3DXVECTOR3 pEye;
	D3DXVECTOR3 pAt;
	D3DXVECTOR3 pUp;


	pEye.x = eye.x;
	pEye.y = eye.y;
	pEye.z = eye.z;

	pAt.x = at.x;
	pAt.y = at.y;
	pAt.z = at.z;

	pUp.x = up.x;
	pUp.y = up.y;
	pUp.z = up.z;


	D3DXMatrixLookAtRH(&matRes, &pEye, &pAt, &pUp);

	memcpy(&m[0], &matRes, sizeof(m));
	*/

	tvector3 X, Y, Z, tmp;

	Z.Normalize(eye - at);
	Y.Normalize(up);

	tmp.Cross(Y, Z);
	X.Normalize(tmp);

	tmp.Cross(Z, X);
	Y.Normalize(tmp);

	m[0][0] = X.x;
	m[0][1] = Y.x;
	m[0][2] = Z.x;
	m[0][3] = 0.0f;

	m[1][0] = X.y;
	m[1][1] = Y.y;
	m[1][2] = Z.y;
	m[1][3] = 0.0f;

	m[2][0] = X.z;
	m[2][1] = Y.z;
	m[2][2] = Z.z;
	m[2][3] = 0.0f;

	m[3][0] = -X.Dot(eye);
	m[3][1] = -Y.Dot(eye);
	m[3][2] = -Z.Dot(eye);
	m[3][3] = 1.0f;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::LookAtLH(const tvector3 &eye, const tvector3 &at, const tvector3 &up )
{
	tvector3 X, Y, Z, tmp;

	Z.Normalize(at - eye);
	Y.Normalize(up);

	tmp.Cross(Y, Z);
	X.Normalize(tmp);

	tmp.Cross(Z, X);
	Y.Normalize(tmp);

	m[0][0] = X.x;
	m[0][1] = Y.x;
	m[0][2] = Z.x;
	m[0][3] = 0.0f;

	m[1][0] = X.y;
	m[1][1] = Y.y;
	m[1][2] = Z.y;
	m[1][3] = 0.0f;

	m[2][0] = X.z;
	m[2][1] = Y.z;
	m[2][2] = Z.z;
	m[2][3] = 0.0f;

	m[3][0] = -X.Dot(eye);
	m[3][1] = -Y.Dot(eye);
	m[3][2] = -Z.Dot(eye);
	m[3][3] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::LookAt(const tvector3 &eye, const tvector3 &at, const tvector3 &up )
{

	tvector3 X, Y, Z, tmp;

	Z.Normalize(at - eye);
	Y.Normalize(up);

	tmp.Cross(Y, Z);
	X.Normalize(tmp);

	tmp.Cross(Z, X);
	Y.Normalize(tmp);

	m[0][0] = X.x;
	m[0][1] = X.y;
	m[0][2] = X.z;
	m[0][3] = 0.0f;

	m[1][0] = Y.x;
	m[1][1] = Y.y;
	m[1][2] = Y.z;
	m[1][3] = 0.0f;

	m[2][0] = Z.x;
	m[2][1] = Z.y;
	m[2][2] = Z.z;
	m[2][3] = 0.0f;

	m[3][0] = eye.x;
	m[3][1] = eye.y;
	m[3][2] = eye.z;
	m[3][3] = 1.0f;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::PerspectiveRH(const float w, const float h, const float zn, const float zf )
{
	m[0][0] = (2 * zn) / w;
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = (2 * zn) / h;
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = zf / (zn - zf);
	m[2][3] = -1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = zn * zf / (zn - zf);
	m[3][3] = 0.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::PerspectiveLH(const float w, const float h, const float zn, const float zf )
{
	m[0][0] = (2 * zn) / w;
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = (2 * zn) / h;
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = zf / (zf - zn);
	m[2][3] = 1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = zn * zf / (zn - zf);
	m[3][3] = 0.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::PerspectiveFovRH(const float fovy, const float aspect, const float zn, const float zf )
{
	float h = (float)MathCos(fovy/2) / (float)MathSin(fovy/2);
	float w = h / aspect;

	m[0][0] = 2 * zn / w;
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 2 * zn / h;
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = zf / (zn - zf);
	m[2][3] = -1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = zn * zf / (zn - zf);
	m[3][3] = 0.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::PerspectiveFovLH(const float fovy, const float aspect, const float zn, const float zf )
{/*
 float h = (float)MathCos(fovy/2) / (float)MathSin(fovy/2);
 float w = h / aspect;

 m[0][0] = 2 * zn / w;
 m[0][1] = 0.0f;
 m[0][2] = 0.0f;
 m[0][3] = 0.0f;

 m[1][0] = 0.0f;
 m[1][1] = 2 * zn / h;
 m[1][2] = 0.0f;
 m[1][3] = 0.0f;

 m[2][0] = 0.0f;
 m[2][1] = 0.0f;
 m[2][2] = zf / (zf - zn);
 m[2][3] = 1.0f;

 m[3][0] = 0.0f;
 m[3][1] = 0.0f;
 m[3][2] = zn * zf / (zn - zf);
 m[3][3] = 0.0f;
 */

	float yscale = cosf(fovy*0.5f);
	float xscale = yscale / aspect;

	m[0][0] = xscale;
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = yscale;
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = zf / (zf - zn);
	m[2][3] = 1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = -zn * zf / (zf - zn);
	m[3][3] = 0.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::PerspectiveFovLH2(const float fovy, const float aspect, const float zn, const float zf )
{
/*
	xScale     0          0               0
0        yScale       0               0
0          0       zf/(zf-zn)         1
0          0       -zn*zf/(zf-zn)     0
where:
*/
/*
+    pout->m[0][0] =3D 1.0f / (aspect * tan(fovy/2.0f));
+    pout->m[1][1] =3D 1.0f / tan(fovy/2.0f);
+    pout->m[2][2] =3D zf / (zf - zn);
+    pout->m[2][3] =3D 1.0f;
+    pout->m[3][2] =3D (zf * zn) / (zn - zf);
+    pout->m[3][3] =3D 0.0f;



float yscale = cosf(fovy*0.5f);

float xscale = yscale / aspect;

*/
	m[0][0] = 1.0f / (aspect * tan(fovy*0.5f));
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 1.0f / tan(fovy*0.5f);
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = zf / (zf - zn);
	m[2][3] = 1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = (zf * zn) / (zn - zf);
	m[3][3] = 0.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::PerspectiveOffCenterRH(const float l, const float r, const float b, const float t, const float zn, const float zf )
{
	m[0][0] = 2 * zn / (r-l);
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 2 * zn * (t-b);
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = (l+r)/(r-l);
	m[2][1] = (t+b)/(t-b);
	m[2][2] = zf / (zn - zf);
	m[2][3] = -1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = zn * zf / (zn - zf);
	m[3][3] = 0.0f;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::PerspectiveOffCenterLH(const float l, const float r, const float b,const float t, const float zn, const float zf )
{
	m[0][0] = 2 * zn / (r-l);
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 2 * zn * (t-b);
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = (l+r)/(r-l);
	m[2][1] = (t+b)/(b-t);
	m[2][2] = zf / (zf - zn);
	m[2][3] = 1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = zn * zf / (zn - zf);
	m[3][3] = 0.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::OrthoRH(const float w, const float h, const float zn, const float zf )
{
	m[0][0] = 2 / w;
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 2 / h;
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = 1.0f / (zn - zf);
	m[2][3] = 0.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = zn / (zn - zf);
	m[3][3] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::OrthoLH(const float w, const float h, const float zn, const float zf )
{
	m[0][0] = 2 / w;
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 2 / h;
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = 1.0f / (zf - zn);
	m[2][3] = 0.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = zn / (zn - zf);
	m[3][3] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::OrthoOffCenterRH(const float l, const float r, const float b, const float t, const float zn, const float zf )
{
	m[0][0] = 2 / (r-l);
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 2 / (t-b);
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = 1.0f / (zn - zf);
	m[2][3] = 0.0f;

	m[3][0] = (l+r)/(l-r);
	m[3][1] = (t+b)/(b-t);
	m[3][2] = zn / (zn - zf);
	m[3][3] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::OrthoOffCenterLH(const float l, float r, float b, const float t, float zn, const float zf )
{
	m[0][0] = 2 / (r-l);
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 2 / (t-b);
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = 1.0f / (zf - zn);
	m[2][3] = 0.0f;

	m[3][0] = (l+r)/(l-r);
	m[3][1] = (t+b)/(b-t);
	m[3][2] = zn / (zn - zf);
	m[3][3] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/*
inline void tmatrix::Shadow(const tvector4 &light, const tplane &plane )
{
// TODO : tmatrix::Shadow(const tvector4 &light, const tplane &plane )

}
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
inline void tmatrix::Reflect(const tplane &plane )
{
// components
tvector3 x = GetLine(0);
tvector3 y = GetLine(1);
tvector3 z = GetLine(2);
tvector3 t = GetLine(3);
tvector3 n2 = plane.normal * -2;

// mirror translation
tvector3 mt = t + n2 * (t.Dot(plane.normal) - plane.distance);

// mirror x rotation
x += t;
x += n2 * (x.Dot(plane.normal) - plane.distance);
x -= mt;

// mirror y rotation
y += t;
y += n2 * (y.Dot(plane.normal) - plane.distance);
y -= mt;

// mirror z rotation
z += t;
z += n2 * (z.Dot(plane.normal) - plane.distance);
z -= mt;

// write result
SetLine(0, x);
SetLine(1, y);
SetLine(2, z);
SetLine(3, mt);

m[0][3] = 0;
m[1][3] = 0;
m[2][3] = 0;
m[3][3] = 1;

}
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::OrthoNormalize()
{
	((tvector3*)&m[0][0])->Normalize();
	((tvector3*)&m[1][0])->Normalize();
	((tvector3*)&m[2][0])->Normalize();
}
///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector4::Length( ) const
{
	return MathSqrt( (x * x) + (y * y) + (z * z) + (w * w) );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector4::LengthSq( ) const
{
	return (x * x) + (y * y) + (z * z) + (w * w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector4::Dot( const tvector4 &v) const
{
	return (x * v.x) + (y * v.y) + (z * v.z) + (w * v.w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector4::Dot( const tvector3 &v) const
{
	return (x * v.x) + (y * v.y) + (z * v.z) ;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Cross(const tvector4 &v)
{
	tvector4 result;

	result.x = y * v.z - z * v.y;
	result.y = z * v.x - x * v.z;
	result.z = x * v.y - y * v.x;
	result.w = 0;

	*this = result;
	//return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Cross(const tvector4 &v1, const tvector4 &v2)
{
	x = v1.y * v2.z - v1.z * v2.y;
	y = v1.z * v2.x - v1.x * v2.z;
	z = v1.x * v2.y - v1.y * v2.x;
	w = 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Add(const tvector4 &v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	w += v.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Add(const tvector4 &v1, const tvector4 &v2)
{
	x = v1.x + v2.x;
	y = v1.y + v2.y;
	z = v1.z + v2.z;
	w = v1.w + v2.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Subtract(const tvector4 &v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	w -= v.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Subtract(const tvector4 &v1, const tvector4 &v2)
{
	x = v1.x - v2.x;
	y = v1.y - v2.y;
	z = v1.z - v2.z;
	w = v1.w - v2.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Minimize(const tvector4 &v)
{
	x = x < v.x ? x : v.x;
	y = y < v.y ? y : v.y;
	z = z < v.z ? z : v.z;
	w = w < v.w ? w : v.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Minimize(const tvector4 &v1, const tvector4 &v2)
{
	x = v1.x < v2.x ? v1.x : v2.x;
	y = v1.y < v2.y ? v1.y : v2.y;
	z = v1.z < v2.z ? v1.z : v2.z;
	w = v1.w < v2.w ? v1.w : v2.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Maximize(const tvector4 &v)
{
	x = x > v.x ? x : v.x;
	y = y > v.y ? y : v.y;
	z = z > v.z ? z : v.z;
	w = w > v.w ? w : v.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Maximize(const tvector4 &v1, const tvector4 &v2)
{
	x = v1.x > v2.x ? v1.x : v2.x;
	y = v1.y > v2.y ? v1.y : v2.y;
	z = v1.z > v2.z ? v1.z : v2.z;
	w = v1.w > v2.w ? v1.w : v2.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Scale(const float s)
{
	x *= s;
	y *= s;
	z *= s;
	w *= s;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Lerp(const tvector4 &v1, const tvector4 &v2, float s )
{
	x = v1.x + s * (v2.x - v1.x);
	y = v1.y + s * (v2.y - v1.y);
	z = v1.z + s * (v2.z - v1.z);
	w = v1.w + s * (v2.w - v1.w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Cross(tvector4 &v1, const tvector4 &v2, const tvector4 &v3)
{
	//tvector4 result;

	v1.x = v2.y * v3.z - v2.z * v3.y;
	v1.y = v2.z * v3.x - v2.x * v3.z;
	v1.z = v2.x * v3.y - v2.y * v3.x;
	v1.w = 0;

	//return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Normalize()
{
	float lenght = MathSqrt( (x * x) + (y * y) + (z * z) );

	if (lenght != 0.0f)
	{
		x /= lenght;
		y /= lenght;
		z /= lenght;
	}
	w = 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Normalize(const tvector4 &/*v*/)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::BaryCentric(const tvector4 &/*v1*/, const tvector4 &/*v2*/, const tvector4 &/*v3*/, float /*f*/, float /*g*/)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Transform(const tmatrix &matrix )
{
	tvector4 out;

	out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + z * matrix.m[2][0] + w * matrix.m[3][0];
	out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + z * matrix.m[2][1] + w * matrix.m[3][1];
	out.z = x * matrix.m[0][2] + y * matrix.m[1][2] + z * matrix.m[2][2] + w * matrix.m[3][2];
	out.w = x * matrix.m[0][3] + y * matrix.m[1][3] + z * matrix.m[2][3] + w * matrix.m[3][3];

	x = out.x;
	y = out.y;
	z = out.z;
	w = out.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Transform(const tvector4 &v, const tmatrix &matrix )
{
	tvector4 out;

	out.x = v.x * matrix.m[0][0] + v.y * matrix.m[1][0] + v.z * matrix.m[2][0] + v.w * matrix.m[3][0];
	out.y = v.x * matrix.m[0][1] + v.y * matrix.m[1][1] + v.z * matrix.m[2][1] + v.w * matrix.m[3][1];
	out.z = v.x * matrix.m[0][2] + v.y * matrix.m[1][2] + v.z * matrix.m[2][2] + v.w * matrix.m[3][2];
	out.w = v.x * matrix.m[0][3] + v.y * matrix.m[1][3] + v.z * matrix.m[2][3] + v.w * matrix.m[3][3];

	x = out.x;
	y = out.y;
	z = out.z;
	w = out.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

inline tcolor::tcolor()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor::tcolor( tulong dw )
{
	const float f = 1.0f / 255.0f;
	r = f * (float) (uint8) (dw >> 16);
	g = f * (float) (uint8) (dw >>  8);
	b = f * (float) (uint8) (dw >>  0);
	a = f * (float) (uint8) (dw >> 24);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor::tcolor( const float* pf )
{
	r = pf[0];
	g = pf[1];
	b = pf[2];
	a = pf[3];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor::tcolor( float fr, float fg, float fb, float fa )
{
	r = fr;
	g = fg;
	b = fb;
	a = fa;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// casting

inline tcolor::operator tulong () const
{
	tulong dwR = r >= 1.0f ? 0xff : r <= 0.0f ? 0x00 : (tulong) (r * 255.0f + 0.5f);
	tulong dwG = g >= 1.0f ? 0xff : g <= 0.0f ? 0x00 : (tulong) (g * 255.0f + 0.5f);
	tulong dwB = b >= 1.0f ? 0xff : b <= 0.0f ? 0x00 : (tulong) (b * 255.0f + 0.5f);
	tulong dwA = a >= 1.0f ? 0xff : a <= 0.0f ? 0x00 : (tulong) (a * 255.0f + 0.5f);

	return (dwA << 24) | (dwR << 16) | (dwG << 8) | dwB;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tulong tcolor::ConvToBGR() const
{
	tulong dwR = r >= 1.0f ? 0xff : r <= 0.0f ? 0x00 : (tulong) (r * 255.0f + 0.5f);
	tulong dwG = g >= 1.0f ? 0xff : g <= 0.0f ? 0x00 : (tulong) (g * 255.0f + 0.5f);
	tulong dwB = b >= 1.0f ? 0xff : b <= 0.0f ? 0x00 : (tulong) (b * 255.0f + 0.5f);
	tulong dwA = 0;

	return (dwA << 24) | (dwB << 16) | (dwG << 8) | dwR;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tulong tcolor::ConvToBGRA() const
{
	tulong dwR = r >= 1.0f ? 0xff : r <= 0.0f ? 0x00 : (tulong) (r * 255.0f + 0.5f);
	tulong dwG = g >= 1.0f ? 0xff : g <= 0.0f ? 0x00 : (tulong) (g * 255.0f + 0.5f);
	tulong dwB = b >= 1.0f ? 0xff : b <= 0.0f ? 0x00 : (tulong) (b * 255.0f + 0.5f);
	tulong dwA = a >= 1.0f ? 0xff : a <= 0.0f ? 0x00 : (tulong) (a * 255.0f + 0.5f);

	return (dwA << 24) | (dwB << 16) | (dwG << 8) | dwR;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tulong tcolor::ConvToRGBA() const
{
	tulong dwR = r >= 1.0f ? 0xff : r <= 0.0f ? 0x00 : (tulong) (r * 255.0f + 0.5f);
	tulong dwG = g >= 1.0f ? 0xff : g <= 0.0f ? 0x00 : (tulong) (g * 255.0f + 0.5f);
	tulong dwB = b >= 1.0f ? 0xff : b <= 0.0f ? 0x00 : (tulong) (b * 255.0f + 0.5f);
	tulong dwA = a >= 1.0f ? 0xff : a <= 0.0f ? 0x00 : (tulong) (a * 255.0f + 0.5f);

	return (dwA << 24) | (dwR << 16) | (dwG << 8) | dwB;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor::operator float * ()
{
	return (float *) &r;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor::operator const float * () const
{
	return (const float *) &r;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// assignment operators

inline tcolor& tcolor::operator += ( const tcolor& c )
{
	r += c.r;
	g += c.g;
	b += c.b;
	a += c.a;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor& tcolor::operator -= ( const tcolor& c )
{
	r -= c.r;
	g -= c.g;
	b -= c.b;
	a -= c.a;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor& tcolor::operator *= ( float f )
{
	r *= f;
	g *= f;
	b *= f;
	a *= f;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor& tcolor::operator /= ( float f )
{
	float fInv = 1.0f / f;
	r *= fInv;
	g *= fInv;
	b *= fInv;
	a *= fInv;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// unary operators

inline tcolor tcolor::operator + () const
{
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor tcolor::operator - () const
{
	return tcolor(-r, -g, -b, -a);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// binary operators

inline tcolor tcolor::operator + ( const tcolor& c ) const
{
	return tcolor(r + c.r, g + c.g, b + c.b, a + c.a);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor tcolor::operator - ( const tcolor& c ) const
{
	return tcolor(r - c.r, g - c.g, b - c.b, a - c.a);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor tcolor::operator * ( float f ) const
{
	return tcolor(r * f, g * f, b * f, a * f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tcolor tcolor::operator / ( float f ) const
{
	float fInv = 1.0f / f;
	return tcolor(r * fInv, g * fInv, b * fInv, a * fInv);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Comparaison operators

inline bool tcolor::operator == ( const tcolor& c ) const
{
	return r == c.r && g == c.g && b == c.b && a == c.a;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tcolor::operator != ( const tcolor& c ) const
{
	return r != c.r || g != c.g || b != c.b || a != c.a;
}



inline tquaternion::tquaternion( )
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tquaternion::Slerp(const tquaternion &q1, const tquaternion &q2, float t )
{
	/*    float fCos;
	float fAngle;

	fCos = q1.Dot(q2);
	fAngle = MathACos(fCos);

	tquaternion res;

	if( fabs(fAngle) < 1.192092896e-07F )
	{
	res = q1;
	x = res.x;
	y = res.y;
	z = res.z;
	w = res.w;
	}

	float fSin = (float)MathSin(fAngle);
	float fInvSin = (float)(1.0f/fSin);
	float fCoeff0 = (float)MathSin((1.0f-t)*fAngle)*fInvSin;
	float fCoeff1 = (float)MathSin(t*fAngle)*fInvSin;
	res = fCoeff0*q1 + fCoeff1*q2;
	x = res.x;
	y = res.y;
	z = res.z;
	w = res.w;
	*/

	// omega is the 4D angle between q0 and q1.
	float    omega, cosom,sinom;
	float    factq0= 1;
	float    s0,s1;
	tquaternion res;

	cosom = q1.Dot(q2);

	// Make q0 and q1 on the same hemisphere.
	if(cosom<0)
	{
		cosom= -cosom;
		factq0= -1;
	}
	// ????

	if(cosom < 1.0 - 0.000001)
	{
		omega = acos(cosom);
		sinom = MathSin(omega);
		s0 = MathSin((1.0f - t)*omega) / sinom;
		s1 = MathSin(t*omega) / sinom;
	}
	else
	{    // q0 and q1 are nearly the same => sinom nearly 0. We can't slerp.
		// just linear interpolate.
		s0 = 1.0f - t;
		s1 = t;
	}

	res = (factq0*s0)*q1 + s1*q2;
	x = res.x;
	y = res.y;
	z = res.z;
	w = res.w;


	/*FIXME
	D3DXQUATERNION xq1;
	D3DXQUATERNION xq2;
	D3DXQUATERNION xq3;
	D3DXQUATERNION xq4;
	D3DXQUATERNION res;


	xq1.x = q1.x;
	xq1.y = q1.y;
	xq1.z = q1.z;
	xq1.w = q1.w;

	xq2.x = q2.x;
	xq2.y = q2.y;
	xq2.z = q2.z;
	xq2.w = q2.w;

	D3DXQuaternionSlerp(&res, &xq1, &xq2, t);
	x = res.x;
	y = res.y;
	z = res.z;
	w = res.w;
	*/

}
///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tquaternion::IsVeryClose(tquaternion &srcQuat)
{
	if(MathFloatIsVeryClose(x, srcQuat.x) && MathFloatIsVeryClose(y, srcQuat.y)  && MathFloatIsVeryClose(z, srcQuat.z)  && MathFloatIsVeryClose(w, srcQuat.w))
	{
		return true;
	}

	return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tquaternion::Dot( const tquaternion &q) const
{
	return (x * q.x) + (y * q.y) + (z * q.z) + (w * q.w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion operator * (float f, const tquaternion& q )
{
	return tquaternion(f * q.x, f * q.y, f * q.z, f * q.w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion::tquaternion( const float* pf )
{
	x = pf[0];
	y = pf[1];
	z = pf[2];
	w = pf[3];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion::tquaternion( float fx, float fy, float fz, float fw )
{
	x = fx;
	y = fy;
	z = fz;
	w = fw;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
inline tquaternion::tquaternion(tquaternion & p_q)
{
x = p_q.x;
y = p_q.y;
z = p_q.z;
w = p_q.w;
}
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
// casting

inline tquaternion::operator float* ()
{
	return (float *) &x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion::operator const float* () const
{
	return (const float *) &x;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// assignment operators

inline tquaternion& tquaternion::operator= (const tquaternion& p_q)
{
	x = p_q.x;
	y = p_q.y;
	z = p_q.z;
	w = p_q.w;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion& tquaternion::operator += ( const tquaternion& q )
{
	x += q.x;
	y += q.y;
	z += q.z;
	w += q.w;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion& tquaternion::operator -= ( const tquaternion& q )
{
	x -= q.x;
	y -= q.y;
	z -= q.z;
	w -= q.w;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion& tquaternion::operator *= ( const tquaternion& q )
{
	this->Multiply(q);
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion& tquaternion::operator *= ( float f )
{
	x *= f;
	y *= f;
	z *= f;
	w *= f;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion& tquaternion::operator /= ( float f )
{
	float fInv = 1.0f / f;
	x *= fInv;
	y *= fInv;
	z *= fInv;
	w *= fInv;
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// unary operators

inline tquaternion tquaternion::operator + () const
{
	return *this;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion tquaternion::operator - () const
{
	return tquaternion(-x, -y, -z, -w);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// binary operators

inline tquaternion tquaternion::operator + ( const tquaternion& q ) const
{
	return tquaternion(x + q.x, y + q.y, z + q.z, w + q.w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion tquaternion::operator - ( const tquaternion& q ) const
{
	return tquaternion(x - q.x, y - q.y, z - q.z, w - q.w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion tquaternion::operator * ( const tquaternion& q ) const
{
	tquaternion qT;
	qT.Multiply(*this, q);
	return qT;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion tquaternion::operator * ( float f ) const
{
	return tquaternion(x * f, y * f, z * f, w * f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion tquaternion::operator / ( float f ) const
{
	float fInv = 1.0f / f;
	return tquaternion(x * fInv, y * fInv, z * fInv, w * fInv);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Comparaison Operator

inline bool tquaternion::operator == ( const tquaternion& q ) const
{
	return x == q.x && y == q.y && z == q.z && w == q.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tquaternion::operator != ( const tquaternion& q ) const
{
	return x != q.x || y != q.y || z != q.z || w != q.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline	tmatrix&	tmatrix::PreMul(const tmatrix& aM)
{ tmatrix tmp; tmp.Multiply(*this,aM); *this=tmp; return(*this); }

inline	tmatrix&	tmatrix::PostMul(const tmatrix& aM)
{ tmatrix tmp; tmp.Multiply(aM,*this); *this=tmp; return(*this); }

inline	tmatrix&	tmatrix::PostRotate(const tvector3& aAxis, const float aAngle)
{ tmatrix tmp; tmp.RotationAxis(aAxis, aAngle); PostMul(tmp); return(*this); }

inline	tmatrix&	tmatrix::PreRotate(const tvector3& aAxis, const float aAngle)
{ tmatrix tmp; tmp.RotationAxis(aAxis, aAngle); PostMul(tmp); return(*this); }



// Prototype //////////////////////////////////////////////////////////////////////////////////////

inline float SegmentPointSquareDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction);
inline float LinePointSquareDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction);
inline float RayPointSquareDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction);

inline float SegmentPointDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction);
inline float LinePointDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction);
inline float RayPointDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction);


inline float SquaredDistance(const tvector3& vt1, const tvector3& vt2)
{
	return ( (vt1.x-vt2.x)*(vt1.x-vt2.x) + (vt1.y-vt2.y)*(vt1.y-vt2.y) + (vt1.z-vt2.z)*(vt1.z-vt2.z) );
}
inline float SquaredDistance(float vt1x, float vt1y, float vt1z, float vt2x, float vt2y, float vt2z)
{
	return ( (vt1x-vt2x)*(vt1x-vt2x) + (vt1y-vt2y)*(vt1y-vt2y) + (vt1z-vt2z)*(vt1z-vt2z) );
}

inline float SquaredDistance(const tvector3& vt1, float vt2x, float vt2y, float vt2z)
{
	return ( (vt1.x-vt2x)*(vt1.x-vt2x) + (vt1.y-vt2y)*(vt1.y-vt2y) + (vt1.z-vt2z)*(vt1.z-vt2z) );
}

inline float		SquaredDistance2D(const tvector3& aP1,const tvector3& aP2)
{ return (aP1.x-aP2.x)*(aP1.x-aP2.x)+(aP1.z-aP2.z)*(aP1.z-aP2.z);}

inline float Distance(const tvector3& vt1, const tvector3& vt2)
{
	return sqrtf( (vt1.x-vt2.x)*(vt1.x-vt2.x) + (vt1.y-vt2.y)*(vt1.y-vt2.y) + (vt1.z-vt2.z)*(vt1.z-vt2.z) );
}

inline float SquaredDistance(const tvector4& vt1, const tvector4& vt2)
{
	return ( (vt1.x-vt2.x)*(vt1.x-vt2.x) + (vt1.y-vt2.y)*(vt1.y-vt2.y) + (vt1.z-vt2.z)*(vt1.z-vt2.z) );
}

inline float Distance(const tvector4& vt1, const tvector4& vt2)
{
	return sqrtf( (vt1.x-vt2.x)*(vt1.x-vt2.x) + (vt1.y-vt2.y)*(vt1.y-vt2.y) + (vt1.z-vt2.z)*(vt1.z-vt2.z) );
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// x---------------x
//
//         O

inline float SegmentPointSquareDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction)
{
	tvector3 diff = p_point - p_position;

	float fT = diff.Dot(p_direction);

	if( fT <= 0.0f )
	{
		fT = 0.0f;
	}
	else
	{
		float fSqrLen = p_direction.LengthSq();

		if( fT >= fSqrLen )
		{
			fT = 1.0f;
			diff -= p_direction;
		}
		else
		{
			fT /= fSqrLen;
			diff -= fT*p_direction;
		}
	}

	return diff.LengthSq();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// ---------------
//
//         O

inline float LinePointSquareDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction)
{

	tvector3 diff = p_point - p_position;
	float fSqrLen = p_direction.LengthSq();
	float fT = diff.Dot(p_direction) / fSqrLen;
	diff -= fT*p_direction;
	return diff.LengthSq();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// x---------------
//
//         O

inline float RayPointSquareDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction)
{
	tvector3 diff = p_point - p_position;

	float fT = diff.Dot(p_direction);

	if( fT <= 0.0f )
	{
		fT = 0.0f;
	}
	else
	{
		fT /= p_direction.LengthSq();
		diff -= fT*p_direction;
	}

	return diff.LengthSq();

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float SegmentPointDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction)
{
	return (float)MathSqrt(SegmentPointSquareDist(p_point,p_position,p_direction));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float LinePointDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction)
{
	return (float)MathSqrt(LinePointSquareDist(p_point,p_position,p_direction));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float RayPointDist(const tvector3 &p_point, const tvector3 &p_position, const tvector3 &p_direction)
{
	return (float)MathSqrt(RayPointSquareDist(p_point,p_position,p_direction));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tquaternion::RotationAxis(const tvector3 &v, const float angle )
{
	tvector3 axis;

	axis.Normalize(v);

	float sin_a = (float)MathSin(angle/2);
	float cos_a = (float)MathCos(angle/2);

	x = axis.x * sin_a;
	y = axis.y * sin_a;
	z = axis.z * sin_a;
	w = cos_a;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
/*
inline tvector4::tvector4(const tvector3 & p_normal,  float p_distance)
{
x = p_normal.x;
y = p_normal.y;
z = p_normal.z;
w = p_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector4::tvector4(const tvector3 & point1, const tvector3 & point2, const tvector3 & point3)
{
tvector3 normal;
normal.Normal(point1, point2, point3);
normal.Normalize();
w = normal.Dot(point1);
x = normal.x;
y = normal.y;
z = normal.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector4::tvector4(const tvector3 & p_point1, const tvector3 & p_normal)
{
tvector3 normal;
normal.Normalize(p_normal);
w = normal.Dot(p_point1);
x = normal.x;
y = normal.y;
z = normal.z;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector4::tvector4( const tvector4& plane )
{
x = plane.x;
y = plane.y;
z = plane.z;
w = plane.w;
}
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::Init(const tvector3 & p_point1, const tvector3 & p_normal)
{
	tvector3 normal;
	normal.Normalize(p_normal);
	x = normal.x;
	y = normal.y;
	z = normal.z;
	w = -normal.Dot(p_point1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/*
inline void tvector4::Normalize()
{
float ln = 1.f/tvector3(x,y,z).Length();
x *= ln;
y *= ln;
z *= ln;
}
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
// Return the signed distance between the point and the plane

inline float tvector4::DotCoord(const tvector3 & point)
{
	return tvector3(x,y,z).Dot(point) + w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Return the angle between the plane Normal and another normal

inline float tvector4::DotNormal(const tvector3 & pvector)
{
	return tvector3(x,y,z).Dot(pvector);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tvector4::IsFrontFacingTo(const tvector3& direction) const
{
	float dot = tvector3(x,y,z).Dot(direction);
	return (dot <=0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline float tvector4::SignedDistanceTo(const tvector3& point) const
{
	tvector3 dist;
	return (point.Dot(tvector3(x,y,z))) - w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////


bool tvector4::RayInter( tvector3 & interPoint, const tvector3 & position, const tvector3 & direction)
{
	float den = tvector3(x,y,z).Dot(direction);

	if( MathFloatAbs(den) < 0.00001 )
	{
		return false;
	}

	tvector3 tmp = (tvector3(x,y,z) * w) - position;
	interPoint = position + (tvector3(x,y,z).Dot(tmp) / den) * direction;

	return true;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tquaternion::tquaternion( const tmatrix &mat)
{
	float T = mat.m[0][0] + mat.m[1][1] + mat.m[2][2];
	if (T<=0)
	{
		if ((mat.m[0][0] > mat.m[1][1] ) && ( mat.m[0][0] > mat.m[2][2] ) )
		{
			float S = sqrtf(1+mat.m[0][0] - mat.m[1][1] - mat.m[2][2]) * 2;
			x = 1.f/(2*S);
			S = 1.f/S;
			y = (mat.m[0][1] - mat.m[1][0]) * S;
			z = (mat.m[0][2] - mat.m[2][0]) * S;
			w = (mat.m[1][2] - mat.m[2][1]) * S;
		}
		else if ((mat.m[1][1] > mat.m[0][0] ) && ( mat.m[1][1] > mat.m[2][2] ) )
		{
			float S = sqrtf(1-mat.m[0][0] + mat.m[1][1] - mat.m[2][2]) * 2;
			y = 1.f/(2*S);
			S = 1.f/S;
			x = (mat.m[0][1] - mat.m[1][0]) * S;
			z = (mat.m[1][2] - mat.m[2][1]) * S;
			w = (mat.m[0][2] - mat.m[2][0]) * S;
		}
		else
		{
			float S = sqrtf(1-mat.m[0][0] - mat.m[1][1] + mat.m[2][2]) * 2;
			z = 1.f/(2*S);
			S = 1.f/S;
			x = (mat.m[0][2] - mat.m[2][0]) * S;
			y = (mat.m[1][2] - mat.m[2][1]) * S;
			w = (mat.m[0][1] - mat.m[1][0]) * S;
		}
	}
	else
	{
		float S = 1.f/(2 * sqrtf(T));
		x = (mat.m[2][1] - mat.m[1][2]) * S;
		y = (mat.m[0][2] - mat.m[2][0]) * S;
		z = (mat.m[1][0] - mat.m[0][1]) * S;
		w = 1.f/(4*S);
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// return an invalid result if norm = 0

inline bool tquaternion::Inverse()
{
	float fNorm = w*w + x*x + y*y + z*z;
	if( fNorm > 0.0f )
	{
		float fInvNorm = 1.0f/fNorm;
		x = -x*fInvNorm;
		y = -y*fInvNorm;
		z = -z*fInvNorm;
		w = w*fInvNorm;
		return true;
	}
	else
	{
		return false;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tquaternion::Inverse(const tquaternion &p_q )
{
	float fNorm = p_q.w*p_q.w + p_q.x*p_q.x + p_q.y*p_q.y + p_q.z*p_q.z;
	if( fNorm > 0.0f )
	{
		float fInvNorm = 1.0f/fNorm;
		x = -p_q.x * fInvNorm;
		y = -p_q.y * fInvNorm;
		z = -p_q.z * fInvNorm;
		w =  p_q.w * fInvNorm;
		return true;
	}
	else
	{
		return false;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tquaternion::UnitInverse()
{
	x = -x;
	y = -y;
	z = -z;
	w = w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tquaternion::UnitInverse(const tquaternion &p_q )
{
	x = -p_q.x;
	y = -p_q.y;
	z = -p_q.z;
	w = p_q.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tquaternion::Identity()
{
	x = y = z = 0.0f;
	w = 1.0f;
}

inline tvector3::tvector3(const tvector4& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
}

inline tvector4& tvector4::operator = ( const tvector3& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	w = 0;
	return (*this);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tquaternion::Normalize()
{
	float len = 1.f/(float)MathSqrt(x*x + y*y + z*z + w*w);

	x = x*len;
	y = y*len;
	z = z*len;
	w = w*len;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
inline float tquaternion::Norm() const
{
	return (x*x + y*y + z*z + w*w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tquaternion::Multiply(const tquaternion &q1 )
{
	tvector3 v2(x, y, z);
	tvector3 v1(q1.x, q1.y, q1.z);
	float w1 = q1.w;
	float w2 = w;
	tvector3 tmp;

	w = w1*w2- v1.Dot(v2);
	tmp.Cross(v1,v2);
	tmp = w1*v2+w2*v1+tmp;
	x = tmp.x;
	y = tmp.y;
	z = tmp.z;
	Normalize();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tquaternion::Multiply(const tquaternion &q1, const tquaternion &q2 )
{
	tvector3 v2(q1.x, q1.y, q1.z);
	tvector3 v1(q2.x, q2.y, q2.z);
	float w1 = q2.w;
	float w2 = q1.w;
	tvector3 tmp;

	w = w1*w2- v1.Dot(v2);
	tmp.Cross(v1,v2);
	tmp = w1*v2+w2*v1+tmp;
	x = tmp.x;
	y = tmp.y;
	z = tmp.z;
	Normalize();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tmatrix::Lerp(const tmatrix &matA, const tmatrix &matB, float t)
{
	for (int i=0;i<16;i++)
	{
		m16[i] = LERP(matA.m16[i], matB.m16[i], t);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////


inline tquaternion::tquaternion( float heading, float attitude, float bank)
{
	float c1 = cosf(heading*0.5f);
	float s1 = sinf(heading*0.5f);
	float c2 = cosf(attitude*0.5f);
	float s2 = sinf(attitude*0.5f);
	float c3 = cosf(bank*0.5f);
	float s3 = sinf(bank*0.5f);
	float c1c2 = c1*c2;
	float s1s2 = s1*s2;
	w =c1c2*c3 - s1s2*s3;
	x =c1c2*s3 + s1s2*c3;
	y =s1*c2*c3 + c1*s2*s3;
	z =c1*s2*c3 - s1*c2*s3;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tquaternion::ToEuler(float &heading, float &attitude, float &bank)
{
	float test = x*y + z*w;
	if (test > 0.499f) { // singularity at north pole
		heading = 2 * atan2(x,w);
		attitude = ZPI*0.5f;
		bank = 0;
		return;
	}
	if (test < -0.499f) { // singularity at south pole
		heading = -2 * atan2(x,w);
		attitude = - ZPI*0.5f;
		bank = 0;
		return;
	}
	float sqx = x*x;
	float sqy = y*y;
	float sqz = z*z;
	heading = atan2(2*y*w-2*x*z , 1 - 2*sqy - 2*sqz);
	attitude = asin(2*test);
	bank = atan2(2*x*w-2*y*z , 1 - 2*sqx - 2*sqz);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline void tvector4::MergeBSphere(const tvector4& sph)
{
	if (w > (Distance(*this, sph)+sph.w) )
		return; // sph is inside sphere
	if (sph.w > (Distance(*this, sph)+w) )
	{
		// we are inside sph
		*this = sph;
		return;
	}
	float dist = Distance(*this, sph);
	if (dist <0.001f)
	{
		w = (sph.w>w)?sph.w:w;
		return;
	}

	tvector3 dir = tvector3(*this)-tvector3(sph);
	dir /= dist;
	tvector3 pt1 = tvector3(*this) + dir*w;
	tvector3 pt2 = tvector3(sph) - dir*sph.w;
	tvector3 pos = (pt1+pt2)*0.5f;
	float r = (pt2-pt1).Length()*0.5f;
	x = pos.x;
	y = pos.y;
	z = pos.z;
	w =r;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline bool tvector4::CanFitIn(const tvector4 & wsph) const
{
	if (w> wsph.w) return false;

	float dist = SquaredDistance(*this, wsph);

	return ((dist + w*w)<=(wsph.w*wsph.w));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline tvector4 vector4(const tvector3 & p_point1, const tvector3 & p_normal)
{
	tvector4 ret;
	tvector3 normal;
	normal.Normalize(p_normal);
	ret.w = normal.Dot(p_point1);
	ret.x = normal.x;
	ret.y = normal.y;
	ret.z = normal.z;
	return ret;
}
