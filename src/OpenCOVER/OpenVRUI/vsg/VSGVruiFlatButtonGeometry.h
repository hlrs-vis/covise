/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_FLAT_BUTTON_GEOMETRY_H
#define OSG_VRUI_FLAT_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <vsg/all.h>

#include <string>

namespace vrui
{

class coFlatButtonGeometry;
class VSGVruiTransformNode;

/**
    this class implements a flat, textured button
*/
class VSGVRUIEXPORT VSGVruiFlatButtonGeometry : public vruiButtonProvider
{
public:
    VSGVruiFlatButtonGeometry(coFlatButtonGeometry *button);
    virtual ~VSGVruiFlatButtonGeometry();
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
    vsg::ref_ptr<vsg::Node> createQuad(const vsg::vec3& origin, const vsg::vec3& horizontal, const vsg::vec3& vertical, vsg::ref_ptr<vsg::Data> image);

    vsg::ref_ptr<vsg::Node> createBox(const std::string& textureName);

    ///< creates the overlay check polygon
    vsg::ref_ptr<vsg::Node>  createCheck(const std::string &textureName);
    void createSharedLists(); ///< creates shared coordinate arrays

    vsg::ref_ptr<vsg::Node> normalNode; ///< normal geometry
    vsg::ref_ptr<vsg::Node> pressedNode; ///< pressed normal geometry
    vsg::ref_ptr<vsg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    vsg::ref_ptr<vsg::Node> pressedHighlightNode;
    vsg::ref_ptr<vsg::Node> disabledNode; ///< disabled geometry

    vsg::ref_ptr<vsg::Switch> switchNode;

    coFlatButtonGeometry *button;

    VSGVruiTransformNode *myDCS;

    vsg::ref_ptr<vsg::Data> defaultTexture;

private:
    //shared coord and color list
    static float A; ///< size parameters
    static float B; ///< size parameters
    static float D; ///< size parameters
    static vsg::ref_ptr<vsg::vec3Array> coord1; ///< coordinates
    static vsg::ref_ptr<vsg::vec3Array> coord2; ///< coordinates
    static vsg::ref_ptr<vsg::uintArray> coordIndices;
    static vsg::ref_ptr<vsg::vec3Array> normals; ///< normals
    static vsg::ref_ptr<vsg::vec4Array> colors;

    ///< texture coordinates
    static vsg::ref_ptr<vsg::vec2Array> texCoord;


};
}
#endif
