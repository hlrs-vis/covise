/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <OpenVRUI/vsg/VSGVruiUIElement.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/text/Text.h>
#include <vsg/text/StandardLayout.h>

namespace vrui
{

class coSlider;
class VSGVruiTransformNode;

/** This class provides a basic 3D slider, which is based on a texture mapped
  tickmark field and a round slider position indicator.
*/
class VSGVRUIEXPORT VSGVruiSlider : public VSGVruiUIElement
{
public:
    VSGVruiSlider(coSlider *slider);
    virtual ~VSGVruiSlider();

    void createGeometry();
    void resizeGeometry();

    void createSharedLists();
    void update();

protected:
    ///< DCS of slider position indicator
    vsg::ref_ptr<vsg::MatrixTransform> sliderTransform;

    vsg::ref_ptr<vsg::vec3Array> coord1; ///< position indicator coordinates
    vsg::ref_ptr<vsg::vec3Array> coord2; ///< dial coordinates
    
    vsg::ref_ptr<vsg::vec2Array> texCoord1; ///< texture coordinates of slider position indicator
    vsg::ref_ptr<vsg::vec2Array> texCoord2; ///< texture coordinates of dial

    vsg::ref_ptr<vsg::Node> positionNode; ///< position indicator geode
    vsg::ref_ptr<vsg::Node> dialNode; ///< dial geode

    vsg::ref_ptr<vsg::Node> positionNodeDisabled; ///< disabled position indicator geode
    vsg::ref_ptr<vsg::Node> dialNodeDisabled; ///< disabled dial geode

    vsg::ref_ptr<vsg::VertexIndexDraw> positionNodeVid;
    vsg::ref_ptr<vsg::VertexIndexDraw> dialNodeVid;

    vsg::ref_ptr<vsg::Text> sliderText;
    vsg::ref_ptr<vsg::stringValue> sliderTextString;
    vsg::ref_ptr<vsg::StandardLayout> sliderTextLayout; 

    vsg::ref_ptr<vsg::Switch> switchPosition;
    vsg::ref_ptr<vsg::Switch> switchDial;
    vsg::ref_ptr<vsg::Text> numberText; ///< string for slider value
    
    static vsg::ref_ptr<vsg::uintArray> coordIndices; 
    static vsg::ref_ptr<vsg::vec3Array> normals; ///< slider textures normal
    static vsg::ref_ptr<vsg::vec4Array> colors; // per vertex color to share for dial and slider

    void updateSlider();
    void updateDial();

    // create texture node, bool isSlider decides whether for dial or slider
    vsg::ref_ptr<vsg::Node> createNode(const std::string& textureName, vsg::ref_ptr<vsg::vec3Array> coord
        , vsg::ref_ptr<vsg::vec2Array> texCoord, bool isSlider);

    coSlider *slider;

private:
    vsg::ref_ptr<vsg::Node> createSlider(const std::string &textureName);
    vsg::ref_ptr<vsg::Node> createDial(const std::string &textureName);
    vsg::ref_ptr<vsg::Node> createText(float xPos = 0.0f);

    float sliderDialSize;
    bool initiallyCompiled; 
};
}
