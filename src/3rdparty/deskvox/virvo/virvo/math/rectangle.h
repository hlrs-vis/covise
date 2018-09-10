#ifndef VV_MATH_RECTANGLE_H
#define VV_MATH_RECTANGLE_H

#include "vector.h"

#include <cstddef>


namespace MATH_NAMESPACE
{


//-------------------------------------------------------------------------------------------------
// Layout policies
//

template < size_t Dim, typename T >
class min_max_layout
{
public:

    typedef vector< Dim, T > vector_type;

    min_max_layout() {}
    min_max_layout(vector_type const& min, vector_type const& max) : min(min), max(max) {}

    vector_type min;
    vector_type max;
};

template < typename T >
class xywh_layout
{
public:

    xywh_layout() {}
    xywh_layout(T x, T y, T w, T h) : x(x), y(y), w(w), h(h) {}

    explicit xywh_layout(T const data[4]) : x(data[0]), y(data[1]), w(data[2]), h(data[3]) {}

    T x;
    T y;
    T w;
    T h;
};


//-------------------------------------------------------------------------------------------------
// rectangle
//

template
<
    template < typename > class L /* data layout */,
    typename T
>
class rectangle;


template < typename T >
class rectangle< xywh_layout, T > : public xywh_layout< T >
{
public:

    typedef T value_type;


    rectangle();
    rectangle(T x, T y, T w, T h);

    explicit rectangle(T const data[4]);

    T* data();
    T const* data() const;

    T& operator[](size_t i);
    T const& operator[](size_t i) const;

};


} // MATH_NAMESPACE


#include "detail/rectangle.inl"


#endif // VV_MATH_RECTANGLE


