/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_FLAT_BUTTON_GEOMETRY_H
#define SG_VRUI_FLAT_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <OpenSG/OSGGeometry.h>
#include <OpenSG/OSGNode.h>
#include <OpenSG/OSGSwitch.h>

#include <string>

class coFlatButtonGeometry;
class SGVruiTransformNode;

/**
    this class implements a flat, textured button
*/
class SGVRUIEXPORT SGVruiFlatButtonGeometry : public vruiButtonProvider
{
public:
    SGVruiFlatButtonGeometry(coFlatButtonGeometry *button);
    virtual ~SGVruiFlatButtonGeometry();
    virtual float getWidth() const
    {
        return A;
    }
    virtual float getHeight() const
    {
        return A;
    }

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

protected:
    ///< creates the base button polygon
    osg::NodePtr createBox(const std::string &textureName);
    ///< creates the overlay check polygon
    osg::NodePtr createCheck(const std::string &textureName);
    void createSharedLists(); ///< creates shared coordinate arrays

    osg::NodePtr normalNode; ///< normal geometry
    osg::NodePtr pressedNode; ///< pressed normal geometry
    osg::NodePtr highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    osg::NodePtr pressedHighlightNode;

    osg::SwitchPtr switchCore;

    coFlatButtonGeometry *button;

    SGVruiTransformNode *myDCS;

private:
    //shared coord and color list
    static float A; ///< size parameters
    static float B; ///< size parameters
    static float D; ///< size parameters

    static osg::GeoPositions3fPtr coord1; ///< coordinates
    static osg::GeoPositions3fPtr coord2; ///< coordinates
    static osg::GeoNormals3fPtr normal; ///< normal
    ///< texture coordinates
    static osg::GeoTexCoords2fPtr texCoord;

    osg::NodePtr geode1; ///< base geometry
    osg::NodePtr geode2; ///< overlay geometry
};
#endif
