/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "bumpmap_to_normalmap.h"

using namespace glh;

namespace

{

typedef unsigned char uchar;

uchar range_compress(const float &f)
{
    return uchar((f + 1) * 127.5);
}

vec3ub range_compress(const vec3f &n)
{
    unsigned char v[3];
    v[0] = range_compress(n[0]);
    v[1] = range_compress(n[1]);
    v[2] = range_compress(n[2]);
    return vec3ub(v);
}

void modulate(vec3f &lhs, const vec3f &rhs)

{
    lhs[0] *= rhs[0];
    lhs[1] *= rhs[1];
    lhs[2] *= rhs[2];
}
}

void bumpmap_to_normalmap(const array2<unsigned char> &src,
                          array2<vec3ub> &dst, vec3f scale)
{
    int w = src.get_width();
    int h = src.get_height();
    if (scale[0] == 0.f || scale[1] == 0.f || scale[2] == 0.f)
    {
        float a = float(w) / float(h);
        if (a < 1.f)
        {
            scale[0] = 1.f;
            scale[1] = 1.f / a;
        }
        else
        {
            scale[0] = a;
            scale[1] = 1.f;
        }
        scale[2] = 1.f;
    }
    dst.set_size(w, h);

    for (int i = 1; i < w - 1; i++)
    {
        for (int j = 1; j < h - 1; j++)
        {
            vec3f dfdi(2.f, 0.f, float(src(i + 1, j) - src(i - 1, j)) / 255.f);
            vec3f dfdj(0.f, 2.f, float(src(i, j + 1) - src(i, j - 1)) / 255.f);
            vec3f n = dfdi.cross(dfdj);
            modulate(n, scale);
            n.normalize();
            dst(i, j) = range_compress(n);
        }
    }
    // microsoft non-ansi c++ scoping concession
    {
        // cheesy boundary cop-out
        for (int i = 0; i < w; i++)
        {
            dst(i, 0) = dst(i, 1);
            dst(i, h - 1) = dst(i, h - 2);
        }
        for (int j = 0; j < h; j++)
        {
            dst(0, j) = dst(1, j);
            dst(w - 1, j) = dst(w - 2, j);
        }
    }
}

void bumpmap_to_normalmap(const array2<unsigned char> &src,
                          array2<vec3f> &dst, vec3f scale)
{
    int w = src.get_width();
    int h = src.get_height();

    if (scale[0] == 0.f || scale[1] == 0.f || scale[2] == 0.f)
    {
        float a = float(w) / float(h);
        if (a < 1.f)
        {
            scale[0] = 1.f;
            scale[1] = 1.f / a;
        }
        else
        {
            scale[0] = a;
            scale[1] = 1.f;
        }
        scale[2] = 1.f;
    }
    dst.set_size(w, h);

    for (int i = 1; i < w - 1; i++)
    {
        for (int j = 1; j < h - 1; j++)
        {
            vec3f dfdi(2.f, 0.f, float(src(i + 1, j) - src(i - 1, j)) / 255.f);
            vec3f dfdj(0.f, 2.f, float(src(i, j + 1) - src(i, j - 1)) / 255.f);
            vec3f n = dfdi.cross(dfdj);
            modulate(n, scale);
            n.normalize();
            dst(i, j) = n;
        }
    }
    // microsoft non-ansi c++ scoping concession
    {
        // cheesy boundary cop-out
        for (int i = 0; i < w; i++)
        {
            dst(i, 0) = dst(i, 1);
            dst(i, h - 1) = dst(i, h - 2);
        }
        for (int j = 0; j < h; j++)
        {
            dst(0, j) = dst(1, j);
            dst(w - 1, j) = dst(w - 2, j);
        }
    }
}

void bumpmap_to_normalmap(const array2<unsigned char> &src,
                          array2<vec4f> &dst, vec3f scale)
{
    int w = src.get_width();
    int h = src.get_height();

    if (scale[0] == 0.f || scale[1] == 0.f || scale[2] == 0.f)
    {
        float a = float(w) / float(h);
        if (a < 1.f)
        {
            scale[0] = 1.f;
            scale[1] = 1.f / a;
        }
        else
        {
            scale[0] = a;
            scale[1] = 1.f;
        }
        scale[2] = 1.f;
    }
    dst.set_size(w, h);

    for (int i = 1; i < w - 1; i++)
    {
        for (int j = 1; j < h - 1; j++)
        {
            vec3f dfdi(2.f, 0.f, float(src(i + 1, j) - src(i - 1, j)) / 255.f);
            vec3f dfdj(0.f, 2.f, float(src(i, j + 1) - src(i, j - 1)) / 255.f);
            vec3f n = dfdi.cross(dfdj);
            modulate(n, scale);
            n.normalize();
            dst(i, j) = vec4f(n[0], n[1], n[2], src(i, j) / 255.0f);
        }
    }
    // microsoft non-ansi c++ scoping concession
    {
        // cheesy boundary cop-out
        for (int i = 0; i < w; i++)
        {
            dst(i, 0) = dst(i, 1);
            dst(i, h - 1) = dst(i, h - 2);
        }
        for (int j = 0; j < h; j++)
        {
            dst(0, j) = dst(1, j);
            dst(w - 1, j) = dst(w - 2, j);
        }
    }
}

void bumpmap_to_mipmap_normalmap(const array2<unsigned char> &src,
                                 array2<vec3f> *&dst, int &nlevels, vec3f scale)
{
    int w = src.get_width();
    int h = src.get_height();

    array2<vec3f> base;

    if (scale[0] == 0.f || scale[1] == 0.f || scale[2] == 0.f)
    {
        float a = float(w) / float(h);
        if (a < 1.f)
        {
            scale[0] = 1.f;
            scale[1] = 1.f / a;
        }
        else
        {
            scale[0] = a;
            scale[1] = 1.f;
        }
        scale[2] = 1.f;
    }
    base.set_size(w, h);

    for (int i = 1; i < w - 1; i++)
    {
        for (int j = 1; j < h - 1; j++)
        {
            vec3f dfdi(2.f, 0.f, float(src(i + 1, j) - src(i - 1, j)) / 255.f);
            vec3f dfdj(0.f, 2.f, float(src(i, j + 1) - src(i, j - 1)) / 255.f);
            vec3f n = dfdi.cross(dfdj);
            modulate(n, scale);
            n.normalize();
            base(i, j) = n;
        }
    }
    // microsoft non-ansi c++ scoping concession
    {
        // cheesy boundary cop-out
        for (int i = 0; i < w; i++)
        {
            base(i, 0) = base(i, 1);
            base(i, h - 1) = base(i, h - 2);
        }
        for (int j = 0; j < h; j++)
        {
            base(0, j) = base(1, j);
            base(w - 1, j) = base(w - 2, j);
        }
    }

    // Now build the mipmap levels.
    if (w != h || (w != 1 && w != 2 && w != 4 && w != 8 && w != 16 && w != 32 && w != 64 && w != 128 && w != 256 && w != 512 && w != 1024))
    {
        nlevels = 1;
        dst = new array2<vec3f>[nlevels];
        dst[0].set_size(w, h);
        dst[0] = base;
    }

    // Compute log base 2 of width.
    int bits = w;
    nlevels = 0;
    for (; bits != 0;)
    {
        bits = bits >> 1;
        nlevels++;
    }

    dst = new array2<vec3f>[nlevels];

    // Fill in each level.
    dst[0] = base;
    for (int l = 1; l < nlevels; l++)
    {
        int res = w >> l;
        array2<vec3f> &curr = dst[l];
        array2<vec3f> &prev = dst[l - 1];
        curr.set_size(res, res);
        for (int j = 0; j < res; j++)
        {
            for (int i = 0; i < res; i++)
            {
                vec3f avg = prev(2 * i, 2 * j) + prev(2 * i + 1, 2 * j) + prev(2 * i + 1, 2 * j + 1) + prev(2 * i, 2 * j + 1);
                avg.normalize();
                curr(i, j) = avg;
            }
        }
    }
}
