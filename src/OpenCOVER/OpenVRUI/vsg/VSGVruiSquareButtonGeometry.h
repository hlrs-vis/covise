/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef OSG_VRUI_SQUARE_BUTTON_GEOMETRY_H
#define OSG_VRUI_SQUARE_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/Switch.h>

#include <string>

namespace vrui
{

class coSquareButtonGeometry;
class VSGVruiTransformNode;

/**
    this class implements a square, textured button
*/
class VSGVRUIEXPORT VSGVruiSquareButtonGeometry : public vruiButtonProvider
{
public:
    VSGVruiSquareButtonGeometry(coSquareButtonGeometry *button);
    virtual ~VSGVruiSquareButtonGeometry();

    virtual float getWidth() const
    {
        return 2.0f * (A + B) + C;
    }
    virtual float getHeight() const
    {
        return 2.0f * (A + B) + C;
    }

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

protected:
    ///< creates the base button polygon
    vsg::ref_ptr<vsg::Node> createGeode(const std::string &textureName, int style);
    void createSharedLists(); ///< creates shared coordinate arrays

    vsg::ref_ptr<vsg::Node> normalNode; ///< normal geometry
    vsg::ref_ptr<vsg::Node> pressedNode; ///< pressed normal geometry
    vsg::ref_ptr<vsg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    vsg::ref_ptr<vsg::Node> pressedHighlightNode;
    vsg::ref_ptr<vsg::Node> disabledNode;

    vsg::ref_ptr<vsg::Switch> switchNode;

    coSquareButtonGeometry *button;

    VSGVruiTransformNode *myDCS;

private:
    //shared coord and color list
    static float A;
    static float B;
    static float C;

};
}
#endif
