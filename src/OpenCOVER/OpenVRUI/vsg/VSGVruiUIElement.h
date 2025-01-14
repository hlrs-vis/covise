/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <OpenVRUI/sginterface/vruiUIElementProvider.h>
#include <OpenVRUI/vsg/VSGVruiMatrix.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>

namespace vrui
{

class coUIElement;

class VSGVRUIEXPORT VSGVruiUIElement : public vruiUIElementProvider
{

public:
    VSGVruiUIElement(coUIElement *element);
    virtual ~VSGVruiUIElement();

    //vruiMatrix & VSGVruiUIElement::getTransformMat() const;
    vruiTransformNode *getDCS();

protected:
    VSGVruiTransformNode *myDCS; ///< main DCS of this UI Element
};
}

