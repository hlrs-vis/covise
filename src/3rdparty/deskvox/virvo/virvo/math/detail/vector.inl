namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// Basic arithmetic
//

template < size_t Dim, typename T >
inline vector< Dim, T > operator-(vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = -v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator+(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] + v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator-(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] - v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator*(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] * v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator/(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] / v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator&(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] & v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator|(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] | v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator^(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] ^ v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator<<(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] << v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator>>(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = u[d] >> v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator+(vector< Dim, T > const& v, T s)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] + s;
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator-(vector< Dim, T > const& v, T s)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] - s;
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator*(vector< Dim, T > const& v, T s)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] * s;
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator/(vector< Dim, T > const& v, T s)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] / s;
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator&(vector< Dim, T > const& v, T s)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] & s;
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator|(vector< Dim, T > const& v, T s)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] | s;
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator^(vector< Dim, T > const& v, T s)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] ^ s;
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator<<(vector< Dim, T > const& v, T s)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] << s;
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > operator>>(vector< Dim, T > const& v, T s)
{
    vector< Dim, T > result;

    for (size_t d = 0; d < Dim; ++d)
    {
        result[d] = v[d] >> s;
    }

    return result;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator+=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u + v;
    return u;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator+=(vector< Dim, T >& v, T s)
{
    v = v + s;
    return v;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator-=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u - v;
    return u;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator-=(vector< Dim, T >& v, T s)
{
    v = v - s;
    return v;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator*=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u * v;
    return u;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator*=(vector< Dim, T >& v, T s)
{
    v = v * s;
    return v;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator/=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u / v;
    return u;
}

template < size_t Dim, typename T >
VV_FORCE_INLINE vector< Dim, T >& operator/=(vector< Dim, T >& v, T s)
{
    v = v / s;
    return v;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator&=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u & v;
    return u;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator&=(vector< Dim, T >& v, T s)
{
    v = v & s;
    return v;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator|=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u | v;
    return u;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator|=(vector< Dim, T >& v, T s)
{
    v = v | s;
    return v;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator^=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u ^ v;
    return u;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator^=(vector< Dim, T >& v, T s)
{
    v = v ^ s;
    return v;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator<<=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u << v;
    return u;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator<<=(vector< Dim, T >& v, T s)
{
    v = v << s;
    return v;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator>>=(vector< Dim, T >& u, vector< Dim, T > const& v)
{
    u = u >> v;
    return u;
}

template < size_t Dim, typename T >
inline vector< Dim, T >& operator>>=(vector< Dim, T >& v, T s)
{
    v = v >> s;
    return v;
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template < size_t Dim, typename T >
inline vector< Dim, T > dot(vector< Dim, T > const& u, vector< Dim, T > const& v)
{
    T result(0.0);

    for (size_t d = 0; d < Dim; ++d)
    {
        result += u[d] * v[d];
    }

    return result;
}

template < size_t Dim, typename T >
inline vector< Dim, T > reflect(vector< Dim, T > const& i, vector< Dim, T > const& n)
{
    return i - T(2.0) * dot(n, i) * n;
}

template < size_t Dim, typename T >
inline vector< Dim, T > refract(vector< Dim, T > const& i, vector< Dim, T > const& n, T eta)
{
    T k = T(1.0) - eta * eta * (T(1.0) - dot(n, i) * dot(n, i));
    if (k < T(0.0))
    {
        return vector< Dim, T >(0.0);
    }
    else
    {
        return eta * i - (eta * dot(n, i) + sqrt(k)) * n;
    }
}


} // MATH_NAMESPACE


