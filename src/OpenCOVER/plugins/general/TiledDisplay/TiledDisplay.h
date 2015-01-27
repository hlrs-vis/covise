/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef OPENCOVER_TILED_DISPLAY_PLUGIN_H
#define OPENCOVER_TILED_DISPLAY_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2006 HLRS  **
 **                                                                          **
 ** Description: Sending a tiled display to a composer                       **
 **                                                                          **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
 ** History:                                                                 **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "TiledDisplayDefines.h"

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <osg/Camera>

#include <string>

class TiledDisplayClient;
class TiledDisplayServer;
class TiledDisplayDimension;
class TiledDisplayCompositor;

class TiledDisplay : public coVRPlugin, public osg::Camera::DrawCallback
{
public:
    TiledDisplay();
    virtual ~TiledDisplay();

    void preFrame();

    void operator()(const osg::Camera &cam) const;

private:
    void updateTextures();

    osg::ref_ptr<osg::Camera> camera;

    TiledDisplayClient *client;
    TiledDisplayServer **servers;
    TiledDisplayCompositor **compositors;
    //TiledDisplayDimension * dimensions;

    int tileX;
    int tileY;
    int number;

    unsigned int width;
    unsigned int height;

    bool initPending;
    bool internalCompositor;

    std::string compositor;
};

#endif
