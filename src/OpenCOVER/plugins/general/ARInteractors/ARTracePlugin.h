/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ARTRACE_PLUGIN_H
#define _ARTRACE_PLUGIN_H

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/MarkerTracking.h>
#include <cover/coTabletUI.h>
#include <cover/coInteractor.h>
#include <util/DLinkList.h>
#include <OpenVRUI/coMenu.h>

namespace vrui
{

class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coPotiMenuItem;
class coVRLabel;
}

using namespace vrui;
using namespace opencover;

class pfHighlight;
class ARTracePlugin;

class TraceModule : public coMenuListener, public coTUIListener
{
public:
    coTUILabel *TracerModuleLabel;
    coTUIToggleButton *updateOnVisibilityChange;
    coTUIEditFloatField *updateInterval;
    coTUIButton *updateNow;
    coTUIEditFloatField *p1X;
    coTUIEditFloatField *p1Y;
    coTUIEditFloatField *p1Z;
    coTUIEditFloatField *p2X;
    coTUIEditFloatField *p2Y;
    coTUIEditFloatField *p2Z;
    coInteractor *inter;

    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    coSubMenuItem *arMenuEntry;
    coRowMenu *moduleMenu;
    coCheckboxMenuItem *enabledToggle;
    bool enabled;
    void menuEvent(coMenuItem *);
    osg::Vec3 startpointOffset1;
    osg::Vec3 startpointOffset2;
    osg::Vec3 startnormal;
    osg::Vec3 startnormal2;
    osg::Vec3 lastPosition1;
    osg::Vec3 lastPosition2;
    osg::Vec3 currentPosition1;
    osg::Vec3 currentPosition2;
    osg::Vec3 currentNormal;
    osg::Vec3 currentNormal2;
    float positionThreshold;
    bool calcPositionChanged();
    void update();
    TraceModule(int ID, const char *n, int mInst, const char *fi, ARTracePlugin *p, coInteractor *inter);
    virtual ~TraceModule();
    MarkerTrackingMarker *marker;
    char *feedbackInfo;
    bool positionChanged;
    bool oldVisibility;
    bool firstUpdate;
    int id;
    int instance;
    char *moduleName;
    ARTracePlugin *plugin;
    double oldTime;
    bool doUpdate;
};

class ARTracePlugin : public coVRPlugin, public coMenuListener, public coMenuFocusListener
{
public:
    ARTracePlugin();
    virtual ~ARTracePlugin();
    virtual bool init();
    coSubMenuItem *pinboardEntry;
    coRowMenu *arMenu;
    coCheckboxMenuItem *enabledToggle;
    coTUITab *arTraceTab;
    coTUILabel *TracerModulesLabel;
    bool enabled;
    virtual void focusEvent(bool focus, coMenu *menu);
    // menu event for buttons and stuff
    void menuEvent(coMenuItem *);

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void newInteractor(const RenderObject *container, coInteractor *i);

    // this will be called if a COVISE object arrives
    void addObject(const RenderObject *container, osg::Group *root, const RenderObject *, const RenderObject *, const RenderObject *, const RenderObject *);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);
    bool idExists(int ID);

private:
    covise::DLinkList<TraceModule *> modules;
    int ID;
    MarkerTrackingMarker *timestepMarker;
};
#endif
