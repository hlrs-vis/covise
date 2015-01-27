/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_TEXTURE_H
#define SG_VRUI_TEXTURE_H

#include <OpenVRUI/sginterface/vruiTexture.h>

#include <OpenSG/OSGImage.h>

class SGVRUIEXPORT SGVruiTexture : public vruiTexture
{
public:
    SGVruiTexture(osg::ImagePtr texture)
    {
        this->texture = texture;
    }

    virtual ~SGVruiTexture()
    {
    }

    osg::ImagePtr getTexture()
    {
        return texture;
    }

private:
    osg::ImagePtr texture;
};
#endif
