#ifndef VV_MATH_PLANE_H
#define VV_MATH_PLANE_H

#include "forward.h"

#include <cstddef>


namespace MATH_NAMESPACE
{

template < typename T >
class basic_plane< 3, T >
{
public:

    typedef T value_type;
    typedef vector< 3, T > vec_type;

    vec_type normal;
    value_type offset;

    basic_plane();

    // NOTE: n must be normalized!
    basic_plane(vec_type const& n, value_type o);
    basic_plane(vec_type const& n, vec_type const& p);

};

} // MATH_NAMESPACE


#include "detail/plane3.inl"


#endif // VV_MATH_PLANE_H


