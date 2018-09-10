#ifndef VV_DETAIL_MATH_H
#define VV_DETAIL_MATH_H

#include "../config.h"

#include <virvo/vvmacros.h>

#include <cmath>

#undef min
#undef max
#ifdef WIN32
#pragma warning (disable: 4305)
#endif

namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// Import required math functions from the standard library.
// Enable ADL!
//

using std::sqrt;

template < typename T >
VV_FORCE_INLINE T min(T const& x, T const& y)
{
    return x < y ? x : y;
}

template < typename T >
VV_FORCE_INLINE T max(T const& x, T const& y)
{
    return x < y ? y : x;
}


//--------------------------------------------------------------------------------------------------
// Implement some (useful) functions not defined in <cmath>
//

template < typename T >
VV_FORCE_INLINE T clamp(T const& x, T const& a, T const& b)
{
    return max( a, min(x, b) );
}

template < typename T >
VV_FORCE_INLINE T lerp(T a, T b, T x)
{
    return a + x * (b - a);
}

template < typename T >
VV_FORCE_INLINE T rsqrt(T const& x)
{
    return T(1.0) / sqrt(x);
}

template < typename T >
inline T cot(T x)
{
    return tan(M_PI_2 - x);
}

template < typename T >
inline T det2(T const& m00, T const& m01, T const& m10, T const& m11)
{
    return m00 * m11 - m10 * m01;
}


//--------------------------------------------------------------------------------------------------
// Constants
//

namespace constants
{
template < typename T > T degrees_to_radians()  { return T(1.74532925199432957692369076849e-02); }
template < typename T > T radians_to_degrees()  { return T(5.72957795130823208767981548141e+01); }
template < typename T > T e()                   { return T(2.71828182845904523536028747135e+00); }
template < typename T > T log2_e()              { return T(1.44269504088896338700465094007e+00); }
template < typename T > T pi()                  { return T(3.14159265358979323846264338328e+00); }
template < typename T > T inv_pi()              { return T(3.18309886183790691216444201928e-01); }
} // constants


//--------------------------------------------------------------------------------------------------
// Misc.
//

template < typename T >
inline T select(bool k, T const& a, T const& b)
{
    return k ? a : b;
}

inline bool any(bool b)
{
    return b;
}

inline bool all(bool b)
{
    return b;
}

} // MATH_NAMESPACE


#endif // VV_DETAIL_MATH_H


