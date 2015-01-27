/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_DEFAULT_BUTTON_GEOMETRY_H
#define SG_VRUI_DEFAULT_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenSG/OSGNode.h>
#include <OpenSG/OSGSwitch.h>
#include <OpenSG/OSGText.h>

#include <qstring.h>

class VRUIEXPORT SGVruiDefaultButtonGeometry : public vruiButtonProvider
{
public:
    SGVruiDefaultButtonGeometry(coButtonGeometry *geometry);
    virtual ~SGVruiDefaultButtonGeometry();

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

    virtual float getWidth() const;
    virtual float getHeight() const;

protected:
    osg::NodePtr normalNode; ///< normal geometry
    osg::NodePtr pressedNode; ///< pressed normal geometry
    osg::NodePtr highlightNode; ///< highlighted geometry
    osg::NodePtr pressedHighlightNode; ///< pressed highlighted geometry

    osg::NodePtr switchNode;

    osg::Text text;
    QString textString;

    osg::NodePtr createNode(bool pressed, bool highlighted);

    vruiTransformNode *myDCS;
};
#endif
