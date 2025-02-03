/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_VALUE_POTI_H
#define OSG_VRUI_VALUE_POTI_H

#include <OpenVRUI/vsg/VSGVruiUIElement.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/Geometry.h>
#include <vsg/text/Text.h>
#include <vsg/text/Font.h>

namespace vrui
{

class coValuePoti;
class VSGVruiTransformNode;

class VSGVRUIEXPORT VSGVruiValuePoti : public VSGVruiUIElement
{
public:
    VSGVruiValuePoti(coValuePoti *poti);
    virtual ~VSGVruiValuePoti();

    void createGeometry();
    void resizeGeometry();

    void update();

protected:
    coValuePoti *poti;

    ///< button orientation
    vsg::ref_ptr<vsg::MatrixTransform> potiTransform;
    ///< description text orientation
    vsg::ref_ptr<vsg::MatrixTransform> textTransform;

    vsg::ref_ptr<vsg::Text> text;

    vsg::ref_ptr<vsg::Switch> stateSwitch; ///< Switch node to switch between enabled and disabled geometry

    void initText();
    vsg::ref_ptr<vsg::Node> createPanelNode(const std::string &);

private:
    float oldValue;
    std::string oldButtonText;
    bool oldEnabled;
};
}
#endif
