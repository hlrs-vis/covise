
#pragma once

//-------------------------------------------------------------------------------------------------
// minimax polynomial approximations for transcendental functions
// cf. David H. Eberly: GPGPU Programming for Games and Science, pp. 120
//


#ifdef WIN32
#pragma warning (disable: 4305)
#endif
#include "sse.h"
#include "../detail/math.h"


#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE2)

namespace MATH_NAMESPACE
{
namespace simd
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// detail::frexp and detail::scalbn do not handle subnormals!
//

inline float4 frexp(const float4 &x, int4* exp)
{
    static const int4 exp_mask(0x7f800000);
    static const int4 inv_exp_mask(~0x7f800000);
    static const int4 man_mask(0x3f000000);

    int4 ptr = reinterpret_as_int(x);
    *exp = (ptr & exp_mask) >> 23;
    mask4 is_zero = (*exp == 0);
    *exp = select( is_zero, int4(0), *exp - 126 ); // IEEE-754 stores a biased exponent
    ptr  = select( is_zero, int4(0), ptr & inv_exp_mask );
    ptr  = select( is_zero, int4(0), ptr | man_mask );
    return reinterpret_as_float(ptr);
}

inline float4 scalbn(const float4 &x, const int4 &exp)
{
    static const int4 exp_mask(0x7f800000);
    static const float4 huge_val = reinterpret_as_float(int4(0x7f800000));
    static const float4 tiny_val = reinterpret_as_float(int4(0x00000000));

    int4 xi = reinterpret_as_int(x);
    float4 sign = reinterpret_as_float(xi & 0x80000000);
    int4 k = (xi & exp_mask) >> 23;
    k += exp;

    // overflow?
    mask4 uoflow = k > int4(0xfe);
    float4 huge_or_tiny = _mm_or_ps( select(uoflow, huge_val, tiny_val), sign );

    // overflow or underflow?
    uoflow |= k < int4(0);
    return select( uoflow, huge_or_tiny, reinterpret_as_float((xi & int4(0x807fffff)) | (k << 23)) );
}


//-------------------------------------------------------------------------------------------------
// Polynomials with degree D
//

template < unsigned D >
struct poly_t
{

    template < typename T >
    static T eval(const T &x, T const* p)
    {

        T result(0.0);
        T y(1.0);

        for (unsigned i = 0; i <= D; ++i)
        {
            result += p[i] * y;
            y *= x;
        }

        return result;

    }
};

template < unsigned D >
struct pow2_t;

template < >
struct pow2_t< 1 > : public poly_t< 1 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(1.0), T(1.0)
        };

        return poly_t< 1 >::eval(x, p);
    }
};

template < >
struct pow2_t< 2 > : public poly_t< 2 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.5571332605741528E-1),
            T(3.4428667394258472E-1)
        };

        return poly_t< 2 >::eval(x, p);
    }
};

template < >
struct pow2_t< 3 > : public poly_t< 3 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9589012084456225E-1),
            T(2.2486494900110188E-1),
            T(7.9244930154334980E-2)
        };

        return poly_t< 3 >::eval(x, p);
    }
};

template < >
struct pow2_t< 4 > : public poly_t< 4 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9300392358459195E-1),
            T(2.4154981722455560E-1),
            T(5.1744260331489045E-2),
            T(1.3701998859367848E-2)
        };

        return poly_t< 4 >::eval(x, p);
    }
};

template < >
struct pow2_t< 5 > : public poly_t< 5 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9315298010274962E-1),
            T(2.4014712313022102E-1),
            T(5.5855296413199085E-2),
            T(8.9477503096873079E-3),
            T(1.8968500441332026E-3)
        };

        return poly_t< 5 >::eval(x, p);
    }
};

template < >
struct pow2_t< 6 > : public poly_t< 6 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9314698914837525E-1),
            T(2.4023013440952923E-1),
            T(5.5481276898206033E-2),
            T(9.6838443037086108E-3),
            T(1.2388324048515642E-3),
            T(2.1892283501756538E-4)
        };

        return poly_t< 6 >::eval(x, p);
    }
};

template < >
struct pow2_t< 7 > : public poly_t< 7 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(1.0),
            T(6.9314718588750690E-1),
            T(2.4022637363165700E-1),
            T(5.5505235570535660E-2),
            T(9.6136265387940512E-3),
            T(1.3429234504656051E-3),
            T(1.4299202757683815E-4),
            T(2.1662892777385423E-5)
        };

        return poly_t< 7 >::eval(x, p);
    }
};


inline float4 pow2(const float4 &x)
{
    float4 xi = floor(x);
    float4 xf = x - xi;
    return detail::scalbn(1.0, int4(xi)) * pow2_t< 7 >::value(xf);
}


//-------------------------------------------------------------------------------------------------
// log2(1 + x), x in [0,1)
//

template < unsigned D >
struct log2_t;

template < >
struct log2_t< 1 > : public poly_t< 1 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(0.0), T(1.0)
        };

        return poly_t< 1 >::eval(x, p);
    }
};

template < >
struct log2_t< 7 > : public poly_t< 7 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(0.0),
            T(+1.4426664401536078),
            T(-7.2055423726162360E-1),
            T(+4.7332419162501083E-1),
            T(-3.2514018752954144E-1),
            T(+1.9302966529095673E-1),
            T(-7.8534970641157997E-2),
            T(+1.5209108363023915E-2)
        };

        return poly_t< 7 >::eval(x, p);
    }
};

template < >
struct log2_t< 8 > : public poly_t< 8 >
{
    template < typename T >
    static T value(const T &x)
    {
        static const T p[] =
        {
            T(0.0),
            T(+1.4426896453621882),
            T(-7.2115893912535967E-1),
            T(+4.7861716616785088E-1),
            T(-3.4699935395019565E-1),
            T(+2.4114048765477492E-1),
            T(-1.3657398692885181E-1),
            T(+5.1421382871922106E-2),
            T(-9.1364020499895560E-3)
        };

        return poly_t< 8 >::eval(x, p);
    }
};


template < typename T >
inline T log2(const T &x)
{
    return log2_t< 8 >::value(x);
}

} // detail


//-------------------------------------------------------------------------------------------------
// API
//

float4 exp(const float4 &x);
float4 log(const float4 &x);
float4 log2(const float4 &x);
float4 pow(const float4 &x, const float4 &y);


//-------------------------------------------------------------------------------------------------
// impl
//

inline float4 exp(const float4 &x)
{
    float4 y = x * constants::log2_e< float4 >();
    return detail::pow2(y);
}

inline float4 log(const float4 &x)
{
    return log2(x) / constants::log2_e< float4 >();
}

inline float4 log2(const float4 &x)
{
    int4 n = 0;
    float4 m = detail::frexp(x, &n);
    m *= 2.0f;
    return float4(n - 1) + detail::log2(m - 1.0f);
}

inline float4 pow(const float4 &x, const float4 &y)
{
#if VV_SIMD_HAS_SVML
    return _mm_pow_ps(x, y);
#else
    return exp( y * log(x) );
#endif
}

} // simd
} // MATH_NAMESPACE

#endif // VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE2)
