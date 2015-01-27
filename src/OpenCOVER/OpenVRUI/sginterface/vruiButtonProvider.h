/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_BUTTON_PROVIDER_H
#define VRUI_BUTTON_PROVIDER_H

#include <OpenVRUI/coButtonGeometry.h>

#include <OpenVRUI/util/vruiLog.h>

namespace vrui
{

class vruiTransformNode;

class OPENVRUIEXPORT vruiButtonProvider
{

public:
    vruiButtonProvider(coButtonGeometry *element)
    {
        this->element = element;
    }

    virtual ~vruiButtonProvider()
    {
    }

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active) = 0;

    virtual void createGeometry() = 0;
    virtual void resizeGeometry() = 0;

    virtual float getWidth() const = 0;
    virtual float getHeight() const = 0;

    virtual void setRotation(float /*angle*/)
    {
        VRUILOG("vruiButtonProvider::setRotation warn: stub called")
    }

    virtual vruiTransformNode *getDCS() = 0;

protected:
    coButtonGeometry *element;
};
}
#endif
