/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2011 Visenso  **
 **                                                                        **
 ** Description: ParticlePathPlugin                                        **
 **              for CyberClassroom                                        **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _PARTICLE_PATH_PLUGIN_H
#define _PARTICLE_PATH_PLUGIN_H

#include "Path.h"
#include "Arrow.h"
#include "Target.h"
#include "BoundingBox.h"

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <config/CoviseConfig.h>
#include <vrbclient/VRBClient.h>
#include <PluginUtil/GenericGuiObject.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Material>

using namespace vrui;

class ParticlePathPlugin : public coVRPlugin, public coMenuListener, public GenericGuiObject
{
public:
    // constructor destructor
    ParticlePathPlugin();
    virtual ~ParticlePathPlugin();

    // variables of class
    static ParticlePathPlugin *plugin;

    // inherit from coVRPlugin or MenuListener
    virtual bool init();
    virtual void guiToRenderMsg(const char *msg);
    virtual void preFrame();

protected:
    void menuEvent(coMenuItem *menuItem);
    void menuReleaseEvent(coMenuItem *menuItem);
    void guiParamChanged(GuiParam *guiParam);

private:
    void createMenu();
    void rebuildMenu();
    void updatePreviousPath(bool visible);
    void updatePath();

    osg::ref_ptr<osg::Group> pluginBaseNode;

    BoundingBox *boundingBox;
    Arrow *electricFieldArrow;
    Arrow *magneticFieldArrow;
    Path *path;
    Path *previousPath;
    Target *target;

    int sliderMoving;

    coRowMenu *menu;
    coSliderMenuItem *m_mass;
    coSliderMenuItem *m_charge;
    coSliderMenuItem *m_velocity;
    coSliderMenuItem *m_voltage;
    coSliderMenuItem *m_angle;
    coSliderMenuItem *m_electricField;
    coSliderMenuItem *m_magneticField;

    GuiParamBool *p_visible;

    GuiParamFloat *p_mass;
    GuiParamFloat *p_charge;
    GuiParamFloat *p_velocity;
    GuiParamFloat *p_voltage;
    GuiParamFloat *p_angle;
    GuiParamFloat *p_electricField;
    GuiParamFloat *p_magneticField;

    GuiParamBool *p_mass_visible;
    GuiParamBool *p_charge_visible;
    GuiParamBool *p_velocity_visible;
    GuiParamBool *p_voltage_visible;
    GuiParamBool *p_angle_visible;
    GuiParamBool *p_electricField_visible;
    GuiParamBool *p_magneticField_visible;
};

#endif
