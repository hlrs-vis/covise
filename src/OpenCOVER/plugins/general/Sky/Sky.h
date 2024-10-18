/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Sky_NODE_PLUGIN_H
#define _Sky_NODE_PLUGIN_H

#include <util/common.h>

#include <OpenVRUI/sginterface/vruiActionUserData.h>

#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>
#include <cover/coTabletUI.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <osg/Group>
#include <osgEphemeris/EphemerisModel.h>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>

#include <util/coTypes.h>

#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coPotiMenuItem;
class coVRLabel;
class coTrackerButtonInteraction;
}

using namespace vrml;
using namespace vrui;
using namespace opencover;

class PLUGINEXPORT VrmlNodeSky : public VrmlNodeChild
{
public:
    static void initFields(VrmlNodeSky *node, vrml::VrmlNodeType *t);
    static const char *name();

    VrmlNodeSky(VrmlScene *scene = 0);
    VrmlNodeSky(const VrmlNodeSky &n);

    virtual VrmlNodeSky *toSky() const;

    void eventIn(double timeStamp, const char *eventName,
                 const VrmlField *fieldValue);

    virtual void render(Viewer *);

    bool isEnabled()
    {
        return d_enabled.get();
    }

private:
    // Fields

    VrmlSFBool d_enabled;
    VrmlSFBool d_timeLapse;
    VrmlSFBool d_currentTime;
    VrmlSFInt d_year;
    VrmlSFInt d_month;
    VrmlSFInt d_day;
    VrmlSFInt d_hour;
    VrmlSFInt d_minute;
    VrmlSFFloat d_radius;
    VrmlSFFloat d_latitude;
    VrmlSFFloat d_longitude;
    VrmlSFFloat d_altitude;
};

class SkyPlugin : public coTUIListener, public coVRPlugin
{
public:
    SkyPlugin();
    ~SkyPlugin();

    static SkyPlugin *plugin;

    // this will be called in PreFrame
    void preFrame();
    bool init();
    int cYear, cMonth, cDay, cHour, cMinute;

	coTUILabel *yearLabel;

    coTUITab *skyTab;
    coTUIToggleButton *showSky;
    coTUIEditIntField *yearField;
    coTUIEditIntField *monthField;
    coTUIEditIntField *dayField;
    coTUIEditIntField *hourField;
    coTUIEditIntField *minuteField;
    coTUIEditFloatField *radiusField;
    coTUIEditFloatField *latitudeField;
    coTUIEditFloatField *longitudeField;
    coTUIEditFloatField *altitudeField;
    coTUIToggleButton *currentTime;
    coTUIToggleButton *timeLapse;
    osg::ref_ptr<osgEphemeris::EphemerisModel> ephemerisModel;

    void tabletPressEvent(coTUIElement *);
    void menuEvent(coMenuItem *);
    void tabletEvent(coTUIElement *);
    void displaySky(bool visible);

    void setShowSky(bool v);
    void setRadius(float v);
    void setYear(int v);
    void setMonth(int v);
    void setDay(int v);
    void setHour(int v);
    void setMinute(int v);
    void setLongitude(float v);
    void setLatitude(float v);
    void setAltitude(float v);
    void setTimeLapse(bool v);
    void setCurrentTime(bool v);

    // void useSceneLightasSun(bool v);

private:
};
#endif
