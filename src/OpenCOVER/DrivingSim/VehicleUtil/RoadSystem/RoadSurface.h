/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RoadSurface_h
#define RoadSurface_h

#include "../opencrg/crgSurface.h"

class RoadSurface
{
public:
    virtual ~RoadSurface()
    {
    }
    virtual double height(double, double)
    {
        return 0.0;
    }

protected:
};

class crgSurface : public RoadSurface
{
public:
    enum SurfaceOrientation
    {
        SAME = 1,
        OPPOSITE = -1
    };

    crgSurface(const std::string &, double, double, SurfaceOrientation = SAME, double = 0.0, double = 0.0, double = 0.0, double = 1.0);

    virtual double height(double, double);

    double getLength()
    {
        return surface.getLength();
    }
    double getWidth()
    {
        return surface.getWidth();
    }
    double getLongOffset()
    {
        return sOffset;
    }
    double getLatOffset()
    {
        return tOffset;
    }

    osg::Image *getParallaxMap();

    osg::Image *getPavementTextureImage();

protected:
    opencrg::Surface surface;

    double sStart;
    double sEnd;
    int orientation;
    double sOffset;
    double tOffset;
    double zOffset;
    double zScale;

    osg::Image *parallaxMap;
    osg::Image *pavementTextureImage;
};

#endif
