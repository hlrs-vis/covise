#ifndef VV_MATH_OUTPUT_H
#define VV_MATH_OUTPUT_H


#include "forward.h"


#include <cstddef>
#include <ostream>
#include <sstream>


namespace MATH_NAMESPACE
{

#if VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE2)

//-------------------------------------------------------------------------------------------------
// simd types
//

template
<
    typename CharT,
    typename Traits
>
std::basic_ostream< CharT, Traits >&
operator<<(std::basic_ostream< CharT, Traits >& out, simd::float4 const& v)
{

    std::basic_ostringstream< CharT, Traits > s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    VV_ALIGN(16) float vals[4];
    store(vals, v);
    s << '(' << vals[0] << ',' << vals[1] << ',' << vals[2] << ',' << vals[3] << ')';

    return out << s.str();

}

template
<
    typename CharT,
    typename Traits
>
std::basic_ostream< CharT, Traits >&
operator<<(std::basic_ostream< CharT, Traits >& out, simd::int4 const& v)
{

    std::basic_ostringstream< CharT, Traits > s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    VV_ALIGN(16) int vals[4];
    store(vals, v);
    s << '(' << vals[0] << ',' << vals[1] << ',' << vals[2] << ',' << vals[3] << ')';

    return out << s.str();

}

#endif // VV_SIMD_ISA_GE(VV_SIMD_ISA_SSE2)


//-------------------------------------------------------------------------------------------------
// vectors
//

template
<
    size_t Dim,
    typename T,
    typename CharT,
    typename Traits
>
std::basic_ostream< CharT, Traits >&
operator<<(std::basic_ostream< CharT, Traits >& out, vector< Dim, T > v)
{

    std::basic_ostringstream< CharT, Traits > s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(';
    for (size_t d = 0; d < Dim; ++d)
    {
        s << v[d];
        if (d < Dim - 1)
        {
            s << ',';
        }
    }
    s << ')';

    return out << s.str();

}

template
<
    typename T,
    typename CharT,
    typename Traits
>
std::basic_ostream< CharT, Traits >&
operator<<(std::basic_ostream< CharT, Traits >& out, vector< 3, T > v)
{

    std::basic_ostringstream< CharT, Traits > s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << v.x << ',' << v.y << ',' << v.z << ')';

    return out << s.str();

}


template
<
    typename T,
    typename CharT,
    typename Traits
>
std::basic_ostream< CharT, Traits >&
operator<<(std::basic_ostream< CharT, Traits >& out, vector< 4, T > v)
{

    std::basic_ostringstream< CharT, Traits > s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << v.x << ',' << v.y << ',' << v.z << ',' << v.w << ')';

    return out << s.str();

}


//-------------------------------------------------------------------------------------------------
// matrices
//

template
<
    typename T,
    typename CharT,
    typename Traits
>
std::basic_ostream< CharT, Traits >&
operator<<(std::basic_ostream< CharT, Traits >& out, matrix< 4, 4, T > const& m)
{

    std::basic_ostringstream< CharT, Traits > s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << m.col0 << ',' << m.col1 << ',' << m.col2 << ',' << m.col3 << ')';

    return out << s.str();

}



//-------------------------------------------------------------------------------------------------
// rects
//

template
<
    typename T,
    typename CharT,
    typename Traits
>
std::basic_ostream< CharT, Traits>&
operator<<(std::basic_ostream< CharT, Traits >& out, rectangle< xywh_layout, T > const& r)
{

    std::basic_ostringstream< CharT, Traits > s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());

    s << '(' << r.x << ',' << r.y << ',' << r.w << ',' << r.h << ')';

    return out << s.str();

}


} // MATH_NAMESPACE


#endif // VV_MATH_OUTPUT_H


