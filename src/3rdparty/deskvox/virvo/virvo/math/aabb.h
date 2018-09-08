#ifndef VV_MATH_AABB_H
#define VV_MATH_AABB_H

#include "vector.h"

#include <virvo/vvmacros.h>

#if VV_CXXLIB_HAS_HDR_ARRAY
#include <array>
#else
#include <boost/array.hpp>
#endif

namespace MATH_NAMESPACE
{


template < typename T >
class basic_aabb
{
public:

    typedef T value_type;
    typedef vector< 3, T > vec_type;
#if VV_CXXLIB_HAS_HDR_ARRAY
    typedef std::array< vec_type, 8 > vertex_list;
#else
    typedef boost::array< vec_type, 8 > vertex_list;
#endif

    vec_type min;
    vec_type max;

    basic_aabb();
    basic_aabb(vec_type const& min, vec_type const& max);

    template < typename U >
    explicit basic_aabb(vector< 3, U > const& min, vector< 3, U > const& max);

    vec_type center() const;
    vec_type size() const;

    bool contains(vec_type const& v) const;

};

} // MATH_NAMESPACE

#include "detail/aabb.inl"

#endif // VV_MATH_AABB_H


