/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _START_BUTTON_H
#define _START_BUTTON_H

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Material>
#include <osgText/Text>

#include <cover/coVRIntersectionInteractor.h>

#include <iostream>

enum ButtonState
{
    BUTTON_STATE_CHECK,
    BUTTON_STATE_OK,
    BUTTON_STATE_NOTOK
};
class CheckButton : public opencover::coVRIntersectionInteractor
{
public:
    CheckButton(osg::Vec3 pos, float size);
    virtual ~CheckButton();

    void startInteraction();

    bool wasClicked();
    void setText(std::string t);
    void setButtonState(ButtonState state);
    void setVisible(bool visible);

protected:
    virtual void createGeometry();

private:
    bool was_clicked;

    osg::ref_ptr<osg::Geode> cylinderGeode;
    osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
    osg::ref_ptr<osg::Cylinder> cylinderGeometry;

    osg::ref_ptr<osgText::Text> textDrawable;
    osg::ref_ptr<osg::Geode> textGeode;

    osg::ref_ptr<osg::Node> geometryNodeCheck_, geometryNodeOk_, geometryNodeNotOk_;
};

#endif
