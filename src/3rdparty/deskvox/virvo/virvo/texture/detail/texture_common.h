#ifndef VV_TEXTURE_COMMON_H
#define VV_TEXTURE_COMMON_H

#include "../forward.h"

#include <algorithm>
#include <vector>


namespace virvo
{

namespace detail
{

template < typename S, typename T >
struct cast
{
    float operator()(T val) { return static_cast< S >(val); }
};

} // detail


template < typename T, typename Derived >
class texture_storage
{
public:

    typedef T value_type;


    value_type const* data;

    texture_storage() : address_mode_(Wrap) {}

    void set_address_mode(tex_address_mode mode) { address_mode_ = mode; }
    tex_address_mode get_address_mode() const { return address_mode_; }

protected:

    tex_address_mode address_mode_;

};


template < typename T, typename Derived >
class basic_filterable
{
public:

    basic_filterable() : filter_mode_(Nearest) {}

    void set_filter_mode(tex_filter_mode mode) { filter_mode_ = mode; }
    tex_filter_mode get_filter_mode() const { return filter_mode_; }

protected:

    tex_filter_mode filter_mode_;

};


template < typename T, typename Derived >
class prefilterable
{
public:

    typedef T value_type;
    typedef short element_type;

    element_type* prefiltered_data;


    prefilterable() : prefiltered_data(0), filter_mode_(Nearest) {}

    void set_filter_mode(tex_filter_mode mode)
    {
        if (mode == BSplineInterpol)
        {
            Derived* d = static_cast< Derived* >(this);

            prefiltered_.resize( d->size() );
            prefiltered_data = &prefiltered_[0];
            std::transform( &(d->data)[0], &(d->data)[d->size()], prefiltered_.begin(), detail::cast< element_type, value_type >() );
            convert_for_bspline_interpol( d );
        }

        filter_mode_ = mode;
    }

    tex_filter_mode get_filter_mode() const { return filter_mode_; }

protected:

    tex_filter_mode filter_mode_;
    std::vector< element_type > prefiltered_;

};


} // virvo


#endif // VV_TEXTURE_COMMON_H


