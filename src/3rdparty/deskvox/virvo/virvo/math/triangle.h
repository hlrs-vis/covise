#ifndef VV_MATH_TRIANGLE_H
#define VV_MATH_TRIANGLE_H

#include "vector.h"

namespace MATH_NAMESPACE
{

template < typename T /* (unsigned) int type */ >
struct primitive
{
    typedef T id_type;

    id_type geom_id;
    id_type prim_id;
};

template < size_t Dim, typename T, typename P >
class basic_triangle : public primitive< P >
{
public:

    typedef T scalar_type;
    typedef vector< Dim, T > vec_type;

    vec_type v1;
    vec_type e1;
    vec_type e2;

    basic_triangle() {}

};

} // MATH_NAMESPACE

#endif // VV_MATH_TRIANGLE_H


