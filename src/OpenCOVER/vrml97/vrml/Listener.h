/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LISTENER_
#define _LISTENER_

#include <math.h>
#include "vrmlexport.h"

#include <iostream>

#ifdef __GNUC__
#define CERR std::cerr << __FUNCTION__ << "(" << __FILE__ << ":" << __LINE__ << "): "
#else
#define CERR std::cerr << "(" << __FILE__ << ":" << __LINE__ << "): "
#endif

namespace vrml
{

class VRMLEXPORT vec
{
public:
    float x, y, z;
    vec(float x1 = 0.0, float y1 = 0.0, float z1 = 0.0)
    {
        x = x1;
        y = y1;
        z = z1;
    }
    float dot(vec v)
    {
        return x * v.x + y * v.y + z * v.z;
    }
    vec cross(vec v)
    {
        return vec(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    float length()
    {
        return sqrt(x * x + y * y + z * z);
    }
    vec add(vec v)
    {
        return vec(x + v.x, y + v.y, z + v.z);
    }
    vec sub(vec v)
    {
        return vec(x - v.x, y - v.y, z - v.z);
    }
    vec mult(float f)
    {
        return vec(x * f, y * f, z * f);
    }
    vec div(float f)
    {
        return vec(x / f, y / f, z / f);
    }
    vec normalize()
    {
        float l = length();
        if (l != 0.0)
        {
            return this->div(l);
        }
        else
        {
            return *this;
        }
    }
};

class vec;

class VRMLEXPORT Listener
{
public:
    virtual ~Listener()
    {
    }

    virtual vec WCtoOC(vec pos) const = 0;
    virtual vec WCtoVC(vec pos) const = 0;
    virtual vec VCtoOC(vec pos) const = 0;
    virtual vec VCtoWC(vec pos) const = 0;
    virtual vec OCtoWC(vec pos) const = 0;
    virtual vec OCtoVC(vec pos) const = 0;

    virtual vec getPositionOC() const = 0;
    virtual vec getPositionVC() const = 0;
    virtual vec getPositionWC() const = 0;

    virtual vec getVelocity() const = 0;
    virtual void getOrientation(vec *at, vec *up) const = 0;

    virtual double getTime() const = 0;
};
}
#endif
