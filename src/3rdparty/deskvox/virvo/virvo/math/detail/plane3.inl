namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// plane3 members
//

template < typename T >
inline basic_plane< 3, T >::basic_plane()
{
}

template < typename T >
inline basic_plane< 3, T >::basic_plane(vector< 3, T > const& n, T o)
    : normal(n)
    , offset(o)
{
}

template < typename T >
inline basic_plane< 3, T >::basic_plane(vector< 3, T > const& n, vector< 3, T > const& p)
    : normal(n)
    , offset(dot(n, p))
{
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <size_t Dim, typename T>
inline bool operator==(basic_plane<Dim, T> const& a, basic_plane<Dim, T> const& b)
{
    return a.normal == b.normal && a.offset == b.offset;
}

template <size_t Dim, typename T>
inline bool operator!=(basic_plane<Dim, T> const& a, basic_plane<Dim, T> const& b)
{
    return a.normal != b.normal || a.offset != b.offset;
}

} // MATH_NAMESPACE


