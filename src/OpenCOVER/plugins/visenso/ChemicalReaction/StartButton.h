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

using namespace opencover;
using namespace covise;

#include <iostream>

enum ButtonState
{
    BUTTON_STATE_START,
    BUTTON_STATE_RESET
};

class StartButton : public coVRIntersectionInteractor
{
public:
    StartButton();
    virtual ~StartButton();

    void startInteraction();

    bool wasClicked();
    void setButtonState(ButtonState state);
    void setVisible(bool visible);

protected:
    virtual void createGeometry();

private:
    bool was_clicked;

    osg::ref_ptr<osg::Geode> cylinderGeode;
    osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
    osg::ref_ptr<osg::Cylinder> cylinderGeometry;

    osg::ref_ptr<osg::Geode> boxGeode;
    osg::ref_ptr<osg::ShapeDrawable> boxDrawable;
    osg::ref_ptr<osg::Box> boxGeometry;

    osg::ref_ptr<osgText::Text> textDrawable;
    osg::ref_ptr<osg::Geode> textGeode;
};

#endif
