/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: AR ParticleTracer interface                                 **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** May-03  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#endif
#include "ARTracePlugin.h"
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <cover/RenderObject.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRConfig.h>
#include <cover/coInteractor.h>
#include <cover/VRViewer.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/MarkerTracking.h>
#include <cover/coVRTui.h>

#ifdef USE_COVISE
#include <appl/RenderInterface.h>
#endif

using namespace covise;

// C plugin interface, don't do any coding down here, do it in the C++ Class!

static ARTracePlugin *plugin = NULL;

TraceModule::TraceModule(int ID, const char *n, int mInst, const char *fi, ARTracePlugin *p, coInteractor *in)
{
    inter = in;
    startpointOffset1.set(0, 0, 0.1);
    startpointOffset2.set(0, 0, 100.1);
    startnormal.set(1, 0, 0);
    startnormal2.set(0, 1, 0);
    lastPosition1.set(0, 0, 0);
    lastPosition2.set(0, 0, 0);
    positionThreshold = 3000.0;
    currentPosition1.set(11110, 0, 0);
    currentPosition2.set(111110, 0, 0);
    plugin = p;
    if (fi)
    {
        feedbackInfo = new char[strlen(fi) + 1];
        strcpy(feedbackInfo, fi);
    }
    else
    {
        feedbackInfo = NULL;
    }
    moduleName = new char[strlen(n) + 1];
    strcpy(moduleName, n);
    instance = mInst;
    positionChanged = false;
    id = ID;
    oldVisibility = true;
    char markerName[100];
    sprintf(markerName, "%s%d", n, mInst);
    marker = MarkerTracking::instance()->getMarker(markerName);
    arMenuEntry = new coSubMenuItem(markerName);
    plugin->arMenu->add(arMenuEntry);
    moduleMenu = new coRowMenu(markerName);
    arMenuEntry->setMenu(moduleMenu);
    enabled = true;
    enabledToggle = new coCheckboxMenuItem("enabled", enabled);
    enabledToggle->setMenuListener(this);
    moduleMenu->add(enabledToggle);
    firstUpdate = true;

    doUpdate = false;

    char *label = new char[strlen(n) + 100];
    sprintf(label, "%s%d:", n, mInst);
    TracerModuleLabel = new coTUILabel(label, plugin->arTraceTab->getID());
    updateOnVisibilityChange = new coTUIToggleButton("updateOnVisibilityChange", plugin->arTraceTab->getID());
    updateNow = new coTUIButton("update", plugin->arTraceTab->getID());
    updateInterval = new coTUIEditFloatField("updateInterval", plugin->arTraceTab->getID());
    p1X = new coTUIEditFloatField("startPos1 X", plugin->arTraceTab->getID());
    p1Y = new coTUIEditFloatField("startPos1 Y", plugin->arTraceTab->getID());
    p1Z = new coTUIEditFloatField("startPos1 Z", plugin->arTraceTab->getID());
    p2X = new coTUIEditFloatField("startPos2 X", plugin->arTraceTab->getID());
    p2Y = new coTUIEditFloatField("startPos2 Y", plugin->arTraceTab->getID());
    p2Z = new coTUIEditFloatField("startPos2 Z", plugin->arTraceTab->getID());
    delete[] label;

    TracerModuleLabel->setEventListener(this);
    updateOnVisibilityChange->setEventListener(this);
    updateInterval->setEventListener(this);
    updateNow->setEventListener(this);
    p1X->setEventListener(this);
    p1Y->setEventListener(this);
    p1Z->setEventListener(this);
    p2X->setEventListener(this);
    p2Y->setEventListener(this);
    p2Z->setEventListener(this);

    updateOnVisibilityChange->setState(false);
    updateInterval->setValue(0.4);
    p1X->setValue(0.0f);
    p1Y->setValue(0.0f);
    p1Z->setValue(0.1f);
    p2X->setValue(0.0f);
    p2Y->setValue(0.0f);
    p2Z->setValue(1.1f);

    TracerModuleLabel->setPos(0, 1 + ID * 5);
    updateOnVisibilityChange->setPos(0, 1 + ID * 5 + 1);
    updateInterval->setPos(1, 1 + ID * 5 + 1);
    updateNow->setPos(0, 1 + ID * 5 + 2);
    p1X->setPos(0, 1 + ID * 5 + 3);
    p1Y->setPos(1, 1 + ID * 5 + 3);
    p1Z->setPos(2, 1 + ID * 5 + 3);
    p2X->setPos(0, 1 + ID * 5 + 4);
    p2Y->setPos(1, 1 + ID * 5 + 4);
    p2Z->setPos(2, 1 + ID * 5 + 4);
    oldTime = 0.0;
}

TraceModule::~TraceModule()
{
    delete marker;
    delete[] feedbackInfo;
    delete[] moduleName;
    delete enabledToggle;
    delete moduleMenu;
    delete arMenuEntry;

    delete TracerModuleLabel;
    delete updateOnVisibilityChange;
    delete updateInterval;
    delete updateNow;
    delete p1X;
    delete p1Y;
    delete p1Z;
    delete p2X;
    delete p2Y;
    delete p2Z;
}

void TraceModule::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == updateNow)
    {
        doUpdate = true;
    }
}

void TraceModule::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == p1X)
    {
        startpointOffset1[0] = p1X->getValue();
    }
    if (tUIItem == p1Y)
    {
        startpointOffset1[1] = p1Y->getValue();
    }
    if (tUIItem == p1Z)
    {
        startpointOffset1[2] = p1Z->getValue();
    }
    if (tUIItem == p2X)
    {
        startpointOffset2[0] = p2X->getValue();
    }
    if (tUIItem == p2Y)
    {
        startpointOffset2[1] = p2Y->getValue();
    }
    if (tUIItem == p2Z)
    {
        startpointOffset2[2] = p2Z->getValue();
    }
    if (tUIItem == p1X)
    {
        startpointOffset1[0] = p1X->getValue();
    }
}
void TraceModule::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == enabledToggle)
    {
        enabled = enabledToggle->getState();
    }
}

bool TraceModule::calcPositionChanged()
{
    osg::Vec3 d1 = lastPosition1 - currentPosition1;
    osg::Vec3 d2 = lastPosition2 - currentPosition2;

    if ((d1.length() > positionThreshold) || (d2.length() > positionThreshold))
    {
        lastPosition2 = currentPosition2;
        lastPosition1 = currentPosition1;
        return true;
    }
    return false;
}

void TraceModule::update()
{
    if (marker == NULL)
        return;
    if (firstUpdate)
    {
        firstUpdate = false;
        return;
    }
    positionChanged = calcPositionChanged();
    bool visibilityChanged = oldVisibility != marker->isVisible();
    oldVisibility = marker->isVisible();
    osg::Matrix MarkerPos; // marker position in camera coordinate system
    MarkerPos = marker->getMarkerTrans();
    osg::Matrix MarkerInLocalCoords;
    osg::Matrix MarkerInWorld, leftCameraTrans;
    leftCameraTrans = VRViewer::instance()->getViewerMat();
    if (coVRConfig::instance()->stereoState())
    {
        leftCameraTrans = osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0) * VRViewer::instance()->getViewerMat();
    }
    MarkerInWorld = MarkerPos * leftCameraTrans;
    MarkerInLocalCoords = MarkerInWorld * cover->getInvBaseMat();

    /*coCoord coord = MarkerInLocalCoords;
   // heading only
   //coord.hpr[0]=0;
   coord.hpr[1]=0;
   coord.hpr[2]=0;
   //coord.xyz[2]=0;
   coord.makeMat(MarkerInLocalCoords);
   */

    //currentPosition1.fullXformPt(startpointOffset1,MarkerInLocalCoords);
    //currentPosition2.fullXformPt(startpointOffset2,MarkerInLocalCoords);
    currentPosition1 = MarkerInLocalCoords.preMult(startpointOffset1);
    currentPosition2 = MarkerInLocalCoords.preMult(startpointOffset2);
    currentNormal = osg::Matrix::transform3x3(MarkerInLocalCoords, startnormal);
    currentNormal2 = osg::Matrix::transform3x3(MarkerInLocalCoords, startnormal2);
    if (positionChanged)
    {
        lastPosition1 = currentPosition1;
        lastPosition2 = currentPosition2;
    }
    else
    {
        if (((updateOnVisibilityChange->getState() && visibilityChanged) || doUpdate || ((!updateOnVisibilityChange->getState()) && (updateInterval->getValue() > 0) && ((cover->frameTime() - oldTime) > updateInterval->getValue()))) && marker->isVisible())
        {
            // send

            doUpdate = false;
            oldTime = cover->frameTime();

            float c;
            currentNormal.normalize();
            c = currentPosition1 * currentNormal;
            c /= currentNormal.length();
            currentNormal.normalize();
            currentNormal2.normalize();
            char ch;
            char buf[10000];
            fprintf(stderr, "%s %f %f %f    %f %f %f\n", moduleName, currentPosition1[0], currentPosition1[1], currentPosition1[2], currentNormal[0], currentNormal[1], currentNormal[2]);

            if (inter)
            {
                if (strncmp(inter->getPluginName(), "Tracer", 6) == 0)
                {
                    inter->setVectorParam("startpoint1", currentPosition1[0], currentPosition1[1], currentPosition1[2]);
                    inter->setVectorParam("startpoint2", currentPosition2[0], currentPosition2[1], currentPosition2[2]);
                    inter->executeModule();
                }
            }
            else if (feedbackInfo)
            {
#ifdef USE_COVISE
                CoviseRender::set_feedback_info(feedbackInfo);
                //cerr << "visibility changed " << CoviseRender::get_feedback_type() << endl;

                switch (ch = CoviseRender::get_feedback_type())
                {
                case 'C':
                    /* button just pressed */
                    {
                        fprintf(stdout, "\a");
                        fflush(stdout);
                        sprintf(buf, "vertex\nFloatVector\n%f %f %f\n", currentNormal[0], currentNormal[1], currentNormal[2]);
                        CoviseRender::send_feedback_message("PARAM", buf);
                        CoviseRender::send_feedback_message("PARAM", buf);
                        sprintf(buf, "scalar\nFloatScalar\n%f\n", c);
                        CoviseRender::send_feedback_message("PARAM", buf);
                        CoviseRender::send_feedback_message("PARAM", buf);
                        buf[0] = '\0';
                        CoviseRender::send_feedback_message("EXEC", buf);
                    }
                    break;
                case 'G': // CutGeometry with new parameter names
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "normal\nFloatVector\n%f %f %f\n", currentNormal[0], currentNormal[1], currentNormal[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "distance\nFloatScalar\n%f\n", c);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
                break;
                case 'Z':
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "vertex\nFloatVector\n%f %f %f\n", currentPosition1[0], currentPosition1[1], currentPosition1[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
                break;

                case 'A':
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "position\nFloatVector\n%f %f %f\n", currentPosition1[0], currentPosition1[1], currentPosition1[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "normal\nFloatVector\n%f %f %f\n", currentNormal[0], currentNormal[1], currentNormal[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "normal2\nFloatVector\n%f %f %f\n", currentNormal2[0], currentNormal2[1], currentNormal2[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
                break;

                case 'T':
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "startpoint1\nFloatVector\n%f %f %f\n", currentPosition1[0], currentPosition1[1], currentPosition1[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "startpoint2\nFloatVector\n%f %f %f\n", currentPosition2[0], currentPosition2[1], currentPosition2[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
                break;
                case 'P':
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "startpoint1\nFloatVector\n%f %f %f\n", currentPosition1[0], currentPosition1[1], currentPosition1[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "startpoint2\nFloatVector\n%f %f %f\n", currentPosition2[0], currentPosition2[1], currentPosition2[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    // TracerUsg has no normal parameter (nor Tracer)...
                    /* && strcmp(currentFeedbackInfo+1,"Tracer")!= 0 */
                    if (strncmp(feedbackInfo + 1, "Tracer", strlen("Tracer")) != 0)
                    {
                        sprintf(buf, "normal\nFloatVector\n%f %f %f\n", currentNormal[0], currentNormal[1], currentNormal[2]);
                        CoviseRender::send_feedback_message("PARAM", buf);
                    }
                    // ... but has direction
                    sprintf(buf, "direction\nFloatVector\n%f %f %f\n", currentNormal2[0], currentNormal2[1], currentNormal2[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
                break;

                case 'I':
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "isopoint\nFloatVector\n%f %f %f\n", currentPosition1[0], currentPosition1[1], currentPosition1[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
                break;

                default:
                    printf("unknown feedback type %c\n", ch);
                }
#else
                printf("Old style COVISE feedback not support, recompile with -DUSE_COVISE\n");
#endif
            }
        }
    }
}

bool ARTracePlugin::idExists(int ID)
{
    modules.reset();
    while (modules.current())
    {
        if (modules.current()->id == ID)
            return true;
        modules.next();
    }
    return false;
}

ARTracePlugin::ARTracePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "ARTracePlugin::ARTracePlugin\n");
    plugin = this;
    timestepMarker = NULL;
    pinboardEntry = NULL;
    arMenu = NULL;
    enabledToggle = NULL;
    TracerModulesLabel = NULL;
    arTraceTab = NULL;
}

bool ARTracePlugin::init()
{
    arTraceTab = new coTUITab("ARTrace", coVRTui::instance()->mainFolder->getID());
    arTraceTab->setPos(0, 0);
    TracerModulesLabel = new coTUILabel("TracerModules", arTraceTab->getID());
    TracerModulesLabel->setPos(0, 0);

    timestepMarker = MarkerTracking::instance()->getMarker("TimestepMarker");
    ID = 0;
    pinboardEntry = new coSubMenuItem("AR Interactors");
    cover->getMenu()->add(pinboardEntry);
    arMenu = new coRowMenu("AR Interactors");
    arMenu->setMenuListener(this);
    pinboardEntry->setMenu(arMenu);
    enabled = true;
    enabledToggle = new coCheckboxMenuItem("enabled", enabled);
    enabledToggle->setMenuListener(this);
    arMenu->add(enabledToggle);
    return true;
}

// this is called if the plugin is removed at runtime
ARTracePlugin::~ARTracePlugin()
{
    fprintf(stderr, "ARTracePlugin::~ARTracePlugin\n");
    delete arMenu;
    delete pinboardEntry;
    delete TracerModulesLabel;
    delete arTraceTab;
}

void ARTracePlugin::newInteractor(const RenderObject *, coInteractor *inter)
{
    fprintf(stderr, "ARTracePlugin::feedback\n");

    modules.reset();
    while (modules.current())
    {
        if ((modules.current()->instance == inter->getModuleInstance()) && (strcmp(modules.current()->moduleName, inter->getModuleName()) == 0))
            return; // we already have this module
        modules.next();
    }
    fprintf(stderr, "ARTracePlugin::new Trace Module %s %d\n", inter->getModuleName(), inter->getModuleInstance());
    // module not found, create it;
    ID++;
    modules.append(new TraceModule(ID, inter->getModuleName(), inter->getModuleInstance(), NULL, this, inter));
    inter->incRefCount();
}

void ARTracePlugin::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == enabledToggle)
    {
        enabled = enabledToggle->getState();
    }
}

void ARTracePlugin::focusEvent(bool, coMenu *)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "VRMenu::focusEvent\n");
}

void ARTracePlugin::addObject(const RenderObject *container, osg::Group *, const RenderObject *obj, const RenderObject *, const RenderObject *, const RenderObject *)
{
    unsigned int i;
    const char *feedbackInfo;
	if (obj == nullptr) // todo: obj is null in vistle due to delayload
		return;
	
    feedbackInfo = obj->getAttribute("FEEDBACK");

    char moduleName[200];
    int moduleInstance;
    if (feedbackInfo)
    {
        char tmp[200];

        strcpy(tmp, feedbackInfo + 1);
        for (i = 0; i < strlen(tmp); i++)
        {
            if (tmp[i] == '\n')
            {
                tmp[i] = ' ';
                break;
            }
        }
        for (; i < strlen(tmp); i++)
        {
            if (tmp[i] == '\n')
            {
                tmp[i] = '\0';
                break;
            }
        }

        //CoviseRender::set_feedback_info(feedbackInfo);

        if (sscanf(tmp, "%s %d", moduleName, &moduleInstance) != 2)
            fprintf(stderr, "ARTracePlugin: sscanf for feedback info failed\n");

        // the module in the module listmodules.reset();

        modules.reset();
        while (modules.current())
        {
            if ((modules.current()->instance == moduleInstance) && (strcmp(modules.current()->moduleName, moduleName) == 0))
                return; // we already have this module
            modules.next();
        }
        fprintf(stderr, "ARTracePlugin::new Trace Module %s %d\n", moduleName, moduleInstance);
        // module not found, create it;
        ID++;
        modules.append(new TraceModule(ID, moduleName, moduleInstance, feedbackInfo, this, NULL));
    }
}

void
ARTracePlugin::removeObject(const char *, bool)
{
    fprintf(stderr, "ARTracePlugin::removeObject\n");
}

void
ARTracePlugin::preFrame()
{

    if (timestepMarker && timestepMarker->isVisible())
    {
        osg::Matrix MarkerPos; // marker position in camera coordinate system
        MarkerPos = timestepMarker->getMarkerTrans();
        osg::Matrix MarkerInLocalCoords;
        osg::Matrix MarkerInWorld, leftCameraTrans;
        leftCameraTrans = VRViewer::instance()->getViewerMat();
        if (coVRConfig::instance()->stereoState())
        {
            leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
        }
        else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_LEFT)
        {
            leftCameraTrans.preMult(osg::Matrix::translate(-(VRViewer::instance()->getSeparation() / 2.0), 0, 0));
        }
        else if (coVRConfig::instance()->monoView() == coVRConfig::MONO_RIGHT)
        {
            leftCameraTrans.preMult(osg::Matrix::translate((VRViewer::instance()->getSeparation() / 2.0), 0, 0));
        }
        MarkerInWorld = MarkerPos * leftCameraTrans;
        MarkerInLocalCoords = MarkerInWorld * cover->getInvBaseMat();
        coCoord coord = MarkerInLocalCoords;
        if (coord.hpr[0] < 0)
            coord.hpr[0] = 360.0 + coord.hpr[0];
        if (coord.hpr[0] > 0)
            coord.hpr[0] = coord.hpr[0] - 360.0;
        cerr << "h: " << coord.hpr[0] << "p: " << coord.hpr[1] << "r: " << coord.hpr[2] << endl;
        int numTS = coVRAnimationManager::instance()->getNumTimesteps();

        int newTS = 0;
        if(numTS > 0)
            newTS = ((int)(coord.hpr[0] / 360.0 * numTS)) % numTS;
        if (newTS < 0)
            newTS = numTS + newTS;
        if (newTS > numTS)
            newTS = newTS - numTS;
        cerr << "newTS: " << newTS << endl;

        coVRAnimationManager::instance()->requestAnimationFrame(newTS);
    }
    if (enabled)
    {
        modules.reset();
        while (modules.current())
        {
            modules.current()->update();
            modules.next();
        }
    }
}

COVERPLUGIN(ARTracePlugin)
