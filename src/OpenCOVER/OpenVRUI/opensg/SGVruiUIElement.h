/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef SG_VRUI_UIELEMENT
#define SG_VRUI_UIELEMENT

#include <OpenVRUI/sginterface/vruiUIElementProvider.h>
#include <OpenVRUI/opensg/SGVruiMatrix.h>
#include <OpenVRUI/opensg/SGVruiTransformNode.h>

class coUIElement;

class SGVruiUIElement : public vruiUIElementProvider
{

public:
    SGVruiUIElement(coUIElement *element);
    virtual ~SGVruiUIElement();

    vruiTransformNode *getDCS();

protected:
    SGVruiTransformNode *myDCS; ///< main DCS of this UI Element
};
#endif
