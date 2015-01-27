/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_RENDERER_H
#define VR_RENDERER_H

#include <osgViewer/Renderer>

#include <util/common.h>

#include <osg/Vec3>
#include <osg/Matrix>

namespace opencover
{
class COVEREXPORT coVRRenderer : public osgViewer::Renderer
{
public:
public:
    coVRRenderer(osg::Camera *camera, int channel);

    virtual ~coVRRenderer();
};
}
#endif
