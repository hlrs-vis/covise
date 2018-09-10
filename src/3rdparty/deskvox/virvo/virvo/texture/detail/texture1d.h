#ifndef VV_TEXTURE_TEXTURE1D_H
#define VV_TEXTURE_TEXTURE1D_H


#include "texture_common.h"

#include <stddef.h>


namespace virvo
{


template
<
    typename T,
    tex_read_mode ReadMode
>
class texture< T, ReadMode, 1 > : public texture_storage< T, texture< T, ReadMode, 1 > >,
    public basic_filterable< T, texture< T, ReadMode, 1 > >
{
public:

    typedef texture_storage< T, texture > storage_base;
    typedef T value_type;


    texture() {}

    texture(size_t w)
        : width_(w)
    {
    }


    value_type& operator()(size_t x)
    {
        return storage_base::data[x];
    }

    value_type const& operator()(size_t x) const
    {
        return storage_base::data[x];
    }


    size_t size() const { return width_; }

    size_t width() const { return width_; }

private:

    size_t width_;

};


} // virvo


#endif // VV_TEXTURE_TEXTURE1D_H


