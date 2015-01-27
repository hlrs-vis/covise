/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_UIELEMENT
#define OSG_VRUI_UIELEMENT

#include <OpenVRUI/sginterface/vruiUIElementProvider.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>

namespace vrui
{

class coUIElement;

class OSGVRUIEXPORT OSGVruiUIElement : public vruiUIElementProvider
{

public:
    OSGVruiUIElement(coUIElement *element);
    virtual ~OSGVruiUIElement();

    //vruiMatrix & OSGVruiUIElement::getTransformMat() const;
    vruiTransformNode *getDCS();

protected:
    OSGVruiTransformNode *myDCS; ///< main DCS of this UI Element
};
}
#endif
