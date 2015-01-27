/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: ElectricFieldPlugin                                       **
 **              for VR4Schule                                             **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _ELECTRIC_FIELD_PLUGIN_H
#define _ELECTRIC_FIELD_PLUGIN_H

#include "ChargedObjectHandler.h"
#include "Probe.h"
#include "Tracer.h"

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

#include <config/CoviseConfig.h>
#include <PluginUtil/GenericGuiObject.h>

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Material>
using namespace opencover;
using namespace covise;

class ElectricFieldPlugin : public coVRPlugin, public coMenuListener, public GenericGuiObject
{
public:
    // constructor destructor
    ElectricFieldPlugin();
    virtual ~ElectricFieldPlugin();

    // variables of class
    static ElectricFieldPlugin *plugin;

    // inherit from coVRPlugin or MenuListener
    virtual bool init();
    virtual void guiToRenderMsg(const char *msg);
    virtual void preFrame();
    virtual void menuEvent(coMenuItem *menuItem);
    virtual void menuReleaseEvent(coMenuItem *menuItem);

    void fieldChanged();

    coRowMenu *getObjectsMenu()
    {
        return objectsMenu;
    };
    void setRadiusOfPlates(float radius);

    bool presentationOn()
    {
        return presentationMode_;
    };

    void setBoundingBoxVisible(bool visible);

protected:
    void createMenu();
    void drawBoundingBox();
    void guiParamChanged(GuiParam *guiParam);

private:
    bool presentationMode_;

    Probe *probe;
    Tracer *tracer;

    GuiParamBool *p_showMenu;

    coRowMenu *objectsMenu;
    coButtonMenuItem *objectsAddPointEntry;
    coButtonMenuItem *objectsAddPlateEntry;
    coButtonMenuItem *objectsClearEntry;

    coSliderMenuItem *menuItemRadius;

    osg::ref_ptr<osg::Geode> boxGeode;
};

#endif
