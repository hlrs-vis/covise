/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_UIELEMENT_PROVIDER_H
#define VRUI_UIELEMENT_PROVIDER_H

#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiMatrix.h>

#include <OpenVRUI/coUIElement.h>
#include <util/coTypes.h>

#include <OpenVRUI/util/vruiLog.h>

namespace vrui
{

class OPENVRUIEXPORT vruiUIElementProvider
{

public:
    vruiUIElementProvider(coUIElement *element)
    {
        this->element = element;
    }

    virtual ~vruiUIElementProvider();

    virtual void createGeometry() = 0;
    virtual vruiTransformNode *getDCS() = 0;

    //virtual vruiMatrix & getTransformMat() const = 0;

    virtual void resizeGeometry() = 0;

    virtual void setEnabled(bool /*enabled*/)
    {
        VRUILOG("vruiUIElementProvider::setEnabled warn: stub called")
    }

    virtual void setHighlighted(bool /*highlighted*/)
    {
        VRUILOG("vruiUIElementProvider::setHighlighted warn: stub called")
    }

    virtual void update()
    {
        VRUILOG("vruiUIElementProvider::update warn: stub called")
    }

    virtual float getWidth() const
    {
        VRUILOG("vruiUIElementProvider::getWidth warn: stub called")
        return 0.0f;
    }

    virtual float getHeight() const
    {
        VRUILOG("vruiUIElementProvider::getHeight warn: stub called")
        return 0.0f;
    }

    virtual float getDepth() const
    {
        VRUILOG("vruiUIElementProvider::getDepth warn: stub called")
        return 0.0f;
    }

protected:
    coUIElement *element;
};
}
#endif
