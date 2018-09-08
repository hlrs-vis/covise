#ifndef VV_MATH_MATRIX_H
#define VV_MATH_MATRIX_H


#include "vector.h"

#include <cassert>


namespace MATH_NAMESPACE
{


template < typename T >
class matrix< 4, 4, T >
{
public:

    typedef vector< 4, T > column_type;

    column_type col0;
    column_type col1;
    column_type col2;
    column_type col3;


    matrix();
    matrix(column_type const& c0,
           column_type const& c1,
           column_type const& c2,
           column_type const& c3);
    matrix(T const& m00, T const& m10, T const& m20, T const& m30,
           T const& m01, T const& m11, T const& m21, T const& m31,
           T const& m02, T const& m12, T const& m22, T const& m32,
           T const& m03, T const& m13, T const& m23, T const& m33);
    matrix(T const& m00, T const& m11, T const& m22, T const& m33);

    explicit matrix(T const data[16]);

    T* data();
    T const* data() const;

    column_type& operator()(size_t col);
    column_type const& operator()(size_t col) const;

    T& operator()(size_t row, size_t col);
    T const& operator()(size_t row, size_t col) const;

    static matrix identity();

};


} // MATH_NAMESPACE


#include "detail/matrix4.inl"


#endif // VV_MATH_MATRIX_H


