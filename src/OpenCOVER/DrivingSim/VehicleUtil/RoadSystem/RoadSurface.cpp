/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RoadSurface.h"

crgSurface::crgSurface(const std::string &filename, double setSStart, double setSEnd, SurfaceOrientation setOrient, double setSOff, double setTOff, double setZOff, double setZScale)
    : surface(filename)
    , sStart(setSStart)
    , sEnd(setSEnd)
    , orientation(setOrient)
    , sOffset(setSOff)
    , tOffset(setTOff)
    , zOffset(setZOff)
    , zScale(setZScale)
    , parallaxMap(NULL)
    , pavementTextureImage(NULL)
{
}

osg::Image *crgSurface::getParallaxMap()
{
    if (!parallaxMap)
    {
        parallaxMap = surface.createParallaxMapTextureImage();
        return parallaxMap;
    }
    else
    {
        return parallaxMap;
    }
}

osg::Image *crgSurface::getPavementTextureImage()
{
    if (!pavementTextureImage)
    {
        pavementTextureImage = surface.createPavementTextureImage();
        return pavementTextureImage;
    }
    else
    {
        return pavementTextureImage;
    }
}

double crgSurface::height(double s, double t)
{
    s -= sOffset + sStart;
    t -= tOffset;
    s = (double)((1 - orientation) / 2) * surface.getLength() + (double)orientation * s;
    t = (double)((1 - orientation) / 2) * surface.getWidth() + (double)orientation * t;

    //std::cerr << "crgSurface(): s: " << s << ", t: " << t << ", height: " << surface(s,t) << std::endl;

    return (surface(s, t) - zOffset) * zScale;
}
