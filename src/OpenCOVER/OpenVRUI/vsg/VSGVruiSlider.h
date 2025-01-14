/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <OpenVRUI/vsg/VSGVruiUIElement.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/text/Text.h>

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

    void update();

protected:
    ///< DCS of slider position indicator
    vsg::ref_ptr<vsg::MatrixTransform> sliderTransform;

    vsg::ref_ptr<vsg::vec3Array> coord1; ///< position indicator coordinates
    vsg::ref_ptr<vsg::vec3Array> coord2; ///< dial coordinates
    vsg::ref_ptr<vsg::vec3Array> normal; ///< slider textures normal
    vsg::ref_ptr<vsg::vec2Array> texCoord1; ///< texture coordinates of slider position indicator
    vsg::ref_ptr<vsg::vec2Array> texCoord2; ///< texture coordinates of dial
    vsg::ref_ptr<vsg::Node> positionNode; ///< position indicator geode
    vsg::ref_ptr<vsg::Node> dialNode; ///< dial geode
    vsg::ref_ptr<vsg::Node> textNode;
    vsg::ref_ptr<vsg::Node> positionNodeDisabled; ///< disabled position indicator geode
    vsg::ref_ptr<vsg::Node> dialNodeDisabled; ///< disabled dial geode

    vsg::ref_ptr<vsg::Switch> switchPosition;
    vsg::ref_ptr<vsg::Switch> switchDial;
    vsg::ref_ptr<vsg::Text> numberText; ///< OSG string for slider value

    void updateSlider();
    void updateDial();

    coSlider *slider;

private:
    vsg::ref_ptr<vsg::Node> createSlider(const std::string &textureName);
    vsg::ref_ptr<vsg::Node> createDial(const std::string &textureName);
    vsg::ref_ptr<vsg::Node> createText(float xPos = 0.0f);

    float sliderDialSize;
};
}
