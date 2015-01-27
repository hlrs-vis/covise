/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_SLIDER_H
#define OSG_VRUI_SLIDER_H

#include <OpenVRUI/osg/OSGVruiUIElement.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osgText/Text>
#include <osg/Switch>

namespace vrui
{

class coSlider;
class OSGVruiTransformNode;

/** This class provides a basic 3D slider, which is based on a texture mapped
  tickmark field and a round slider position indicator.
*/
class OSGVRUIEXPORT OSGVruiSlider : public OSGVruiUIElement
{
public:
    OSGVruiSlider(coSlider *slider);
    virtual ~OSGVruiSlider();

    void createGeometry();
    void resizeGeometry();

    void update();

protected:
    ///< DCS of slider position indicator
    osg::ref_ptr<osg::MatrixTransform> sliderTransform;

    osg::ref_ptr<osg::Vec3Array> coord1; ///< position indicator coordinates
    osg::ref_ptr<osg::Vec3Array> coord2; ///< dial coordinates
    osg::ref_ptr<osg::Vec3Array> normal; ///< slider textures normal
    osg::ref_ptr<osg::Vec2Array> texCoord1; ///< texture coordinates of slider position indicator
    osg::ref_ptr<osg::Vec2Array> texCoord2; ///< texture coordinates of dial
    osg::ref_ptr<osg::Geode> positionNode; ///< position indicator geode
    osg::ref_ptr<osg::Geode> dialNode; ///< dial geode
    osg::ref_ptr<osg::Geode> textNode;
    osg::ref_ptr<osg::Geode> positionNodeDisabled; ///< disabled position indicator geode
    osg::ref_ptr<osg::Geode> dialNodeDisabled; ///< disabled dial geode

    osg::ref_ptr<osg::Switch> switchPosition;
    osg::ref_ptr<osg::Switch> switchDial;
    osg::ref_ptr<osgText::Text> numberText; ///< OSG string for slider value

    void updateSlider();
    void updateDial();

    coSlider *slider;

private:
    osg::ref_ptr<osg::Geode> createSlider(const std::string &textureName);
    osg::ref_ptr<osg::Geode> createDial(const std::string &textureName);
    osg::ref_ptr<osg::Geode> createText(float xPos = 0.0f);

    float sliderDialSize;
};
}
#endif
