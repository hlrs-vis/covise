/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <OpenVRUI/sginterface/vruiTexture.h>


namespace vrui
{

class VSGVRUIEXPORT VSGVruiTexture : public vruiTexture
{
public:
    VSGVruiTexture(vsg::Image *texture)
    {
        this->texture = texture;
    }

    virtual ~VSGVruiTexture()
    {
    }

    vsg::Image *getTexture()
    {
        return texture;
    }

private:
    vsg::Image *texture;
};
}
