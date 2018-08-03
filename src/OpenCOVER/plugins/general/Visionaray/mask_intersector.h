/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifndef VSNRAY_PLUGIN_MASK_INTERSECTOR_H
#define VSNRAY_PLUGIN_MASK_INTERSECTOR_H 1

#include <visionaray/math/simd/simd.h>
#include <visionaray/get_tex_coord.h>
#include <visionaray/intersector.h>

namespace visionaray
{
    //-------------------------------------------------------------------------------------------------
    // TODO: use make_intersector(lambda...) instead
    //

    template <typename TexCoords, typename Texture>
    struct mask_intersector : basic_intersector<mask_intersector<TexCoords, Texture> >
    {
        using basic_intersector<mask_intersector<TexCoords, Texture> >::operator();

        template <typename R, typename S>
        VSNRAY_FUNC auto operator()(R const &ray, basic_triangle<3, S> const &tri)
            -> decltype(intersect(ray, tri))
        {
            auto hr = intersect(ray, tri);

            if (!any(hr.hit))
            {
                return hr;
            }

            auto tex_color = get_tex_color(hr);
            hr.hit &= tex_color.w >= S(0.01);

            return hr;
        }

        TexCoords tex_coords;
        Texture textures;

    private:
        template <typename HR>
        VSNRAY_FUNC
        vector<4, float>
        get_tex_color(HR const &hr)
        {
            auto tc = get_tex_coord(tex_coords, hr);
            auto const &tex = textures[hr.geom_id];
            return tex.width() > 0 && tex.height() > 0
                       ? vector<4, float>(tex2D(tex, tc))
                       : vector<4, float>(1.0);
        }

        template <typename T,
                  typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type>
        VSNRAY_CPU_FUNC
        vector<4, T>
        get_tex_color(hit_record<basic_ray<T>, primitive<unsigned> > const &hr)
        {
            auto tc = get_tex_coord(tex_coords, hr);

            auto hrs = unpack(hr);
            auto tcs = unpack(tc);

            array<vector<4, float>, simd::num_elements<T>::value> tex_colors;

            for (unsigned i = 0; i < simd::num_elements<T>::value; ++i)
            {
                if (!hrs[i].hit)
                {
                    continue;
                }

                auto const &tex = textures[hrs[i].geom_id];
                tex_colors[i] = tex.width() > 0 && tex.height() > 0
                                    ? vector<4, float>(tex2D(tex, tcs[i]))
                                    : vector<4, float>(1.0);
            }

            return simd::pack(tex_colors);
        }
    };
} // visionaray

#endif // VSNRAY_PLUGIN_MASK_INTERSECTOR_H
