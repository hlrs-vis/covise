/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_H
#define OSG_VRUI_H

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

class OSGVruiNode;

class VRUITest : public vruiRendererInterface
{

public:
    VRUITest();
    virtual ~VRUITest();

    virtual vruiNode *getMenuGroup();
    virtual vruiUIElementProvider *createUIElementProvider(coUIElement *element);
    virtual vruiButtonProvider *createButtonProvider(coButtonGeometry *element);
    virtual vruiPanelGeometryProvider *createPanelGeometryProvider(coPanelGeometry *element);

    virtual vruiTransformNode *createTransformNode();
    virtual vruiMatrix *createMatrix();

private:
    OSGVruiNode *groupNode;
};
#endif
