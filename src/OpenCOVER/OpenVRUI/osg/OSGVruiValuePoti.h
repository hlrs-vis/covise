/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_VALUE_POTI_H
#define OSG_VRUI_VALUE_POTI_H

#include <OpenVRUI/osg/OSGVruiUIElement.h>

#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Switch>
#include <osgText/Text>

namespace vrui
{

class coValuePoti;
class OSGVruiTransformNode;

class OSGVRUIEXPORT OSGVruiValuePoti : public OSGVruiUIElement
{
public:
    OSGVruiValuePoti(coValuePoti *poti);
    virtual ~OSGVruiValuePoti();

    void createGeometry();
    void resizeGeometry();

    void update();

protected:
    coValuePoti *poti;

    ///< button orientation
    osg::ref_ptr<osg::MatrixTransform> potiTransform;
    ///< description text orientation
    osg::ref_ptr<osg::MatrixTransform> textTransform;

    osg::ref_ptr<osgText::Text> text;
    osg::ref_ptr<osg::Geode> textNode; ///< Representation of value display text

    osg::ref_ptr<osg::Switch> stateSwitch; ///< Switch node to switch between enabled and disabled geometry

    void initText();
    osg::ref_ptr<osg::Geode> createPanelNode(const std::string &);

private:
    float oldValue;
    std::string oldButtonText;
    bool oldEnabled;
};
}
#endif
