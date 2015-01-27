/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// this file: minimal 3D vector base class for data exchange

/******************************************************************************
 *                       CGV Optical Tracking
 *
 *              license: currently no public release, all rights reserved
 *
 *       main developer: Hyosun Kim
 *  assistant developer: Marcel Lancelle
 *                       2006
 *
 *       Computer Graphics & Knowledge Visualization, TU Graz, Austria
 *                       http://www.cgv.tugraz.at/
 *
 ******************************************************************************/

#ifndef VEC3_BASEtr_HEADER_INCLUDED
#define VEC3_BASEtr_HEADER_INCLUDED

#include <iostream>

class CGVVec3
{
public:
    CGVVec3()
        : x(0)
        , y(0)
        , z(0)
    {
    }
    CGVVec3(float a, float b, float c)
        : x(a)
        , y(b)
        , z(c)
    {
    }
    CGVVec3 &assign(float a, float b, float c)
    {
        x = a;
        y = b;
        z = c;
        return *this;
    }
    friend std::ostream &operator<<(std::ostream &os, const CGVVec3 &v);
    friend std::istream &operator>>(std::istream &is, CGVVec3 &v);

    float x, y, z;
};

// streaming operators

inline std::ostream &operator<<(std::ostream &os, const CGVVec3 &v)
{
    return (os << '(' << v.x << ',' << v.y << ',' << v.z << ')');
}

inline std::istream &operator>>(std::istream &is, CGVVec3 &v)
{
    char c;
    float x, y, z;
    is >> c;
    if (c == '(')
    {
        is >> x >> c;
        if (c == ',')
        {
            is >> y >> c;
            if (c == ',')
            {
                is >> z >> c;
                if (c != ')')
                {
                    is.clear(std::ios_base::badbit);
                }
            }
            else
            {
                is.clear(std::ios_base::badbit);
            }
        }
        else
        {
            is.clear(std::ios_base::badbit);
        }
    }
    else
    {
        is.clear(std::ios_base::badbit);
    }
    if (is)
        v = CGVVec3(x, y, z);
    return is;
}

#endif // VEC3_BASEtr_HEADER_INCLUDED
