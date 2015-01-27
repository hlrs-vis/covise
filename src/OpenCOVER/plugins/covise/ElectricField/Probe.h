/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PROBE_H
#define _PROBE_H

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <cover/coVRLabel.h>
#include <PluginUtil/coVR3DTransInteractor.h>
#include <PluginUtil/GenericGuiObject.h>

using namespace opencover;
using namespace vrui;

class Probe : public GenericGuiObject, public coMenuListener
{
public:
    Probe();
    virtual ~Probe();

    void preFrame();
    void update();

protected:
    void menuEvent(coMenuItem *menuItem);
    void guiParamChanged(GuiParam *guiParam);

private:
    void updateMenuItem();
    void updateProbe();
    void updateIsoSurface();

    coVR3DTransInteractor *interactor;
    osg::ref_ptr<osg::MatrixTransform> transform;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::ShapeDrawable> cylinderD;
    osg::ref_ptr<osg::Cylinder> cylinder;
    osg::ref_ptr<osg::ShapeDrawable> coneD;
    osg::ref_ptr<osg::Cone> cone;
    coVRLabel *label;

    GuiParamBool *p_visible;
    GuiParamVec3 *p_position;
    GuiParamBool *p_showField;
    GuiParamBool *p_showPotential;
    GuiParamBool *p_showIsoSurface;
    GuiParamBool *p_showArrow;

    coCheckboxMenuItem *menuItemVisible;
    coCheckboxMenuItem *menuItemAequipVisible;

    osg::ref_ptr<osg::Node> isoPlane_;
};

#endif
