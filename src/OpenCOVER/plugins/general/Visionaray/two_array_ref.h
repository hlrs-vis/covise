/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef VSNRAY_PLUGIN_TWO_ARRAY_REF_H
#define VSNRAY_PLUGIN_TWO_ARRAY_REF_H 1

#include <cstddef>
#include <iterator>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

namespace visionaray
{

    //-------------------------------------------------------------------------------------------------
    // Wrap two dynamic arrays (e.g. normals, colors, etc.)
    // Behave as if they were stored consecutively
    // TODO: implement full RandIt semantics (?)
    //

    template <typename Cont, typename T = typename Cont::value_type>
    struct two_array_ref
    {
        two_array_ref() = default;

        two_array_ref(const Cont &c1, const Cont &c2)
            : data1(c1.data())
            , size1(c1.size())
            , data2(c2.data())
//          , size2(c2.size())
        {
        }

        const T &operator[](size_t i) const
        {
            if (i < size1)
                return data1[i];
            else
                return data2[i - size1];
        }

        const T *data1;
        size_t size1;

        const T *data2;
//      size_t size2;
    };


#ifdef __CUDACC__

    //-------------------------------------------------------------------------------------------------
    // Specialize for thrust::device_vector
    //

    template <typename T>
    struct two_array_ref<thrust::device_vector<T>>
    {
        two_array_ref() = default;

        two_array_ref(
            const thrust::device_vector<T> &c1,
            const thrust::device_vector<T> &c2)
            : data1(thrust::raw_pointer_cast(c1.data()))
            , size1(c1.size())
            , data2(thrust::raw_pointer_cast(c2.data()))
//          , size2(c2.size())
        {
        }

        __host__ __device__
        const T &operator[](size_t i) const
        {
            if (i < size1)
                return data1[i];
            else
                return data2[i - size1];
        }

        const T *data1;
        size_t size1;

        const T *data2;
//      size_t size2;
    };

#endif

    template <typename Cont, typename Func>
    auto make_two_array_ref(const Cont &cont, size_t index1, size_t index2, Func func)
        -> two_array_ref<typename Cont::value_type>
    {
        if (func(index1) && func(index2))
            return two_array_ref<typename Cont::value_type>(cont[index1], cont[index2]);
        else if (func(index1))
            return two_array_ref<typename Cont::value_type>(cont[index1], typename Cont::value_type());
        else if (func(index2))
            return two_array_ref<typename Cont::value_type>(cont[index2], typename Cont::value_type());
        else
            return two_array_ref<typename Cont::value_type>(typename Cont::value_type(), typename Cont::value_type());
    };

} // namespace visionaray


namespace std
{
    template <typename Cont>
    struct iterator_traits<visionaray::two_array_ref<Cont>>
    {
        using value_type = typename Cont::value_type;
    };
}

#endif // VSNRAY_PLUGIN_TWO_ARRAY_REF_H
