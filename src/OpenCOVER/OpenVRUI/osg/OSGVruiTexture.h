/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_TEXTURE_H
#define OSG_VRUI_TEXTURE_H

#include <OpenVRUI/sginterface/vruiTexture.h>

#include <osg/Texture2D>

namespace vrui
{

class OSGVRUIEXPORT OSGVruiTexture : public vruiTexture
{
public:
    OSGVruiTexture(osg::Texture2D *texture)
    {
        this->texture = texture;
    }

    virtual ~OSGVruiTexture()
    {
    }

    osg::Texture2D *getTexture()
    {
        return texture;
    }

private:
    osg::Texture2D *texture;
};
}
#endif
