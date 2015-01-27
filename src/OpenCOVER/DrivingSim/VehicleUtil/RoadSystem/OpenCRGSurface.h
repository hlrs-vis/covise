/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OpenCRGSurface_h
#define OpenCRGSurface_h

#include "RoadSurface.h"
extern "C" {
#include <crgBaseLibPrivate.h>
}

class OpenCRGSurface : public RoadSurface
{
public:
    enum SurfaceOrientation
    {
        SAME = dCrgOrientFwd,
        OPPOSITE = dCrgOrientRev
    };

    enum SurfaceWrapMode
    {
        NONE = dCrgBorderModeNone,
        EX_ZERO = dCrgBorderModeExZero,
        EX_KEEP = dCrgBorderModeExKeep,
        REPEAT = dCrgBorderModeRepeat,
        REFLECT = dCrgBorderModeReflect
    };

    OpenCRGSurface(const std::string &, double, double);
    OpenCRGSurface(const std::string &, double, double, SurfaceOrientation, double = 0.0, double = 0.0, double = 0.0, double = 1.0, double = 0.0);

    virtual double height(double, double);

    double getLength();
    double getWidth();
    double getLongOffset()
    {
        return sOffset;
    }
    double getLatOffset()
    {
        return tOffset;
    }

    SurfaceWrapMode getSurfaceWrapModeU();
    SurfaceWrapMode getSurfaceWrapModeV();

    osg::Image *getParallaxMap();

    osg::Image *getPavementTextureImage();

protected:
    std::string filename;
    int dataSetId;
    int contactPointId;

    double sStart;
    double sEnd;
    int orientation;
    double sOffset;
    double tOffset;
    double zOffset;
    double zScale;

    osg::Image *parallaxMap;
    osg::Image *pavementTextureImage;

    osg::Image *createDiffuseMapTextureImage();
    osg::Image *createParallaxMapTextureImage();
};

template <typename T>
class Lowpass
{
public:
    Lowpass(const double &f_c, const double &dt, const T &init = T())
        : alpha(dt / ((1.0 / (2.0 * M_PI * f_c)) + dt))
        , y(init)
    {
    }

    T operator()(const T &x)
    {
        y = alpha * x + (1.0 - alpha) * y;
        return y;
    }

private:
    double alpha;
    T y;
};

template <typename T>
class Highpass
{
public:
    Highpass(const double &f_c, const double &dt, const T &init = T())
        : alpha((1.0 / (2.0 * M_PI * f_c)) / ((1.0 / (2.0 * M_PI * f_c)) + dt))
        , y(init)
        , lastX(init)
    {
    }

    T operator()(const T &x)
    {
        y = alpha * y + alpha * (x - lastX);
        lastX = x;
        return y;
    }

private:
    double alpha;
    T y;
    T lastX;
};

#endif
