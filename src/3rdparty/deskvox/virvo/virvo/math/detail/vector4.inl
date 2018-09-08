namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// vector4 members
//

template < typename T >
VV_FORCE_INLINE vector< 4, T >::vector()
{
}

template < typename T >
VV_FORCE_INLINE vector< 4, T >::vector(const T &x, const T &y, const T &z, const T &w)
    : x(x)
    , y(y)
    , z(z)
    , w(w)
{
}

template < typename T >
VV_FORCE_INLINE vector< 4, T >::vector(const T &s)
    : x(s)
    , y(s)
    , z(s)
    , w(s)
{
}

template < typename T >
VV_FORCE_INLINE vector< 4, T >::vector(T const data[4])
    : x(data[0])
    , y(data[1])
    , z(data[2])
    , w(data[3])
{
}

template < typename T >
template < typename U >
VV_FORCE_INLINE vector< 4, T >::vector(vector< 2, U > const& rhs, const U &z, const U &w)
    : x(rhs.x)
    , y(rhs.y)
    , z(z)
    , w(w)
{
}

template < typename T >
template < typename U >
VV_FORCE_INLINE vector< 4, T >::vector(vector< 3, U > const& rhs, const U &w)
    : x(rhs.x)
    , y(rhs.y)
    , z(rhs.z)
    , w(w)
{
}

template < typename T >
template < typename U >
VV_FORCE_INLINE vector< 4, T >::vector(vector< 4, U > const& rhs)
    : x(rhs.x)
    , y(rhs.y)
    , z(rhs.z)
    , w(rhs.w)
{
}

template < typename T >
template < typename U >
VV_FORCE_INLINE vector< 4, T >& vector< 4, T >::operator=(vector< 4, U > const& rhs)
{

    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = rhs.w;
    return *this;

}

template < typename T >
VV_FORCE_INLINE T* vector< 4, T >::data()
{
    return reinterpret_cast< T* >(this);
}

template < typename T >
VV_FORCE_INLINE T const* vector< 4, T >::data() const
{
    return reinterpret_cast< T const* >(this);
}

template < typename T >
VV_FORCE_INLINE T& vector< 4, T >::operator[](size_t i)
{
    return data()[i];
}

template < typename T >
VV_FORCE_INLINE T const& vector< 4, T >::operator[](size_t i) const
{
    return data()[i];
}

template < typename T >
VV_FORCE_INLINE vector< 3, T >& vector< 4, T >::xyz()
{
    return *reinterpret_cast< vector< 3, T >* >( data() );
}

template < typename T >
VV_FORCE_INLINE vector< 3, T > const& vector< 4, T >::xyz() const
{
    return *reinterpret_cast< vector< 3, T > const* >( data() );
}


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator-(vector< 4, T > const& v)
{
    return vector< 4, T >(-v.x, -v.y, -v.z, -v.w);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator+(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return vector< 4, T >(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator-(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return vector< 4, T >(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator*(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return vector< 4, T >(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator/(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return vector< 4, T >(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator+(vector< 4, T > const& v, T const& s)
{
    return vector< 4, T >(v.x + s, v.y + s, v.z + s, v.w + s);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator-(vector< 4, T > const& v, T const& s)
{
    return vector< 4, T >(v.x - s, v.y - s, v.z - s, v.w - s);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator*(vector< 4, T > const& v, T const& s)
{
    return vector< 4, T >(v.x * s, v.y * s, v.z * s, v.w * s);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator/(vector< 4, T > const& v, T const& s)
{
    return vector< 4, T >(v.x / s, v.y / s, v.z / s, v.w / s);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator+(T const& s, vector< 4, T > const& v)
{
    return vector< 4, T >(s + v.x, s + v.y, s + v.z, s + v.w);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator-(T const& s, vector< 4, T > const& v)
{
    return vector< 4, T >(s - v.x, s - v.y, s - v.z, s - v.w);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator*(T const& s, vector< 4, T > const& v)
{
    return vector< 4, T >(s * v.x, s * v.y, s * v.z, s * v.w);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > operator/(T const& s, vector< 4, T > const& v)
{
    return vector< 4, T >(s / v.x, s / v.y, s / v.z, s / v.w);
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template < typename T >
bool operator==(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return u.x == v.x && u.y == v.y && u.z == v.z && u.w == v.w;
}

template < typename T >
bool operator<(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return u.x < v.x || ( (u.x == v.x && u.y < v.y) || ( (u.y == v.y && u.z < v.z) || (u.z == v.z && u.w < v.w) ) );
}

template < typename T >
bool operator!=(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return !(u == v);
}

template < typename T >
bool operator<=(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return !(v < u);
}

template < typename T >
bool operator>(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return v < u;
}

template < typename T >
bool operator>=(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return !(u < v);
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template < typename T >
VV_FORCE_INLINE T dot(vector< 4, T > const& u, vector< 4, T > const& v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

template < typename T >
VV_FORCE_INLINE T norm(vector< 4, T > const& v)
{
    return sqrt( dot(v, v) );
}

template < typename T >
VV_FORCE_INLINE T norm2(vector< 4, T > const& v)
{
    return dot(v, v);
}

template < typename T >
VV_FORCE_INLINE T length(vector< 4, T > const& v)
{
    return norm(v);
}

template < typename T >
VV_FORCE_INLINE vector< 4, T > normalize(vector< 4, T > const& v)
{
    return v * rsqrt( dot(v, v) );
}


} // MATH_NAMESPACE


