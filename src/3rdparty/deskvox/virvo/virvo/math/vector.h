#ifndef VV_MATH_VECTOR_H
#define VV_MATH_VECTOR_H

#include "detail/math.h"

#include "forward.h"

#include <cstddef>


namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// vector2
//

template < typename T >
class vector< 2, T >
{
public:

    typedef T value_type;

    T x;
    T y;

    vector();
    vector(const T &x, const T &y);

    explicit vector(const T &s);
    explicit vector(T const data[2]);

    template < typename U >
    explicit vector(vector< 2, U > const& rhs);

    template < typename U >
    explicit vector(vector< 3, U > const& rhs);

    template < typename U >
    explicit vector(vector< 4, U > const& rhs);

    template < typename U >
    vector& operator=(vector< 2, U > const& rhs);

    T* data();
    T const* data() const;

    T& operator[](size_t i);
    T const& operator[](size_t i) const;

};


//--------------------------------------------------------------------------------------------------
// vector3
//

template < typename T >
class vector< 3, T >
{
public:

    typedef T value_type;

    T x;
    T y;
    T z;

    vector();
    vector(const T &x, const T &y, const T &z);

    explicit vector(const T &s);
    explicit vector(T const data[3]);

    template < typename U >
    explicit vector(vector< 2, U > const& rhs, const U &z);

    template < typename U >
    explicit vector(vector< 3, U > const& rhs);

    template < typename U >
    explicit vector(vector< 4, U > const& rhs);

    template < typename U >
    vector& operator=(vector< 3, U > const& rhs);

    T* data();
    T const* data() const;

    T& operator[](size_t i);
    T const& operator[](size_t i) const;

    vector< 2, T >& xy();
    vector< 2, T > const& xy() const;

};


//--------------------------------------------------------------------------------------------------
// vector4
//

template < typename T >
class vector< 4, T >
{
public:

    typedef T value_type;

    T x;
    T y;
    T z;
    T w;

    vector();
    vector(const T &x, const T &y, const T &z, const T &w);

    explicit vector(const T &s);
    explicit vector(T const data[4]);

    template < typename U >
    explicit vector(vector< 2, U > const& rhs, const U &z, const U &w);

    template < typename U >
    explicit vector(vector< 3, U > const& rhs, const U &w);

    template < typename U >
    explicit vector(vector< 4, U > const& rhs);

    template < typename U >
    vector& operator=(vector< 4, U > const& rhs);

    T* data();
    T const* data() const;

    T& operator[](size_t i);
    T const& operator[](size_t i) const;

    vector< 3, T >& xyz();
    vector< 3, T > const& xyz() const;

};


} // MATH_NAMESPACE


#include "detail/vector.inl"
#include "detail/vector2.inl"
#include "detail/vector3.inl"
#include "detail/vector4.inl"


#endif // VV_MATH_VECTOR_H


