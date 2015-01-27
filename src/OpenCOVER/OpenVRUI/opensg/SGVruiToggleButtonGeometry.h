/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_TOGGLE_BUTTON_GEOMETRY_H
#define SG_VRUI_TOGGLE_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <OpenSG/OSGGeometry.h>
#include <OpenSG/OSGNode.h>
#include <OpenSG/OSGSwitch.h>
#include <OpenSG/OSGTextureChunk.h>

#include <string>

class coToggleButtonGeometry;
class SGVruiTransformNode;

class SGVRUIEXPORT SGVruiToggleButtonGeometry : public vruiButtonProvider
{
public:
    SGVruiToggleButtonGeometry(coToggleButtonGeometry *button);
    virtual ~SGVruiToggleButtonGeometry();
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
    // create Texture, either normal or checked
    osg::NodePtr createNode(const std::string &textureName, bool checkTexture);

    void createSharedLists();

    // kept for compatibility only! They re-call createTexture()
    osg::NodePtr createBox(const std::string &textureName);
    osg::NodePtr createCheck(const std::string &textureName);

    osg::NodePtr normalNode; ///< normal geometry
    osg::NodePtr pressedNode; ///< pressed normal geometry
    osg::NodePtr highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    osg::NodePtr pressedHighlightNode;

    osg::SwitchPtr switchCore;

    coToggleButtonGeometry *button;

    SGVruiTransformNode *myDCS;

private:
    //shared coord and color list
    static float A;
    static float B;
    static float D;

    static osg::GeoPositions3fPtr coord; ///< coordinates
    static osg::GeoNormals3fPtr normal; ///< normal
    static osg::GeoTexCoords2fPtr texCoord; ///< texture coordinates

    osg::TextureChunkPtr textureChunk;
};
#endif
