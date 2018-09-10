#ifndef VV_MATH_RAY_H
#define VV_MATH_RAY_H

#include <virvo/vvmacros.h>

namespace MATH_NAMESPACE
{

template < typename T >
class basic_ray
{
public:

    typedef T value_type;
    typedef vector< 3, T > vec_type;

    vec_type ori;
    vec_type dir;

    VV_FORCE_INLINE basic_ray() {}
    VV_FORCE_INLINE basic_ray(vec_type const& o, vec_type const& d) : ori(o), dir(d) {}

};

} // MATH_NAMESPACE

#endif // VV_MATH_RAY_H


