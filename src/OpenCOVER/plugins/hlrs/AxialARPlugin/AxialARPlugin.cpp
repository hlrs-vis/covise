/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AxialARPlugin.h"

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

#include <cover/RenderObject.h>
#include <cover/VRViewer.h>
#include <cover/ARToolKit.h>
#include <cover/coVRTui.h>
#include <cover/coInteractor.h>
#include <cover/coVRConfig.h>

#include <OpenVRUI/osg/mathUtils.h>

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#endif

AxialARPlugin::AxialARPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "AxialARPlugin::AxialARPlugin\n");
    pinboardEntry = new coSubMenuItem("AxialRunner");
    execButton = new coButtonMenuItem("RestartSimulation");
    execButton->setMenuListener(this);
    angleMarker = new ARToolKitMarker("AngleMarker");
    simulationMenu = new coRowMenu("AxialRunner");
    simulationMenu->add(execButton);
    cover->getMenu()->add(pinboardEntry);
    pinboardEntry->setMenu(simulationMenu);

    tab = new coTUITab("AxialRunner", coVRTui::tui->mainFolder->getID());
    restartSimulation = new coTUIButton("RestartSimulation", tab->getID());
    tab->setPos(0, 0);
    restartSimulation->setPos(0, 0);
    restartSimulation->setEventListener(this);

    interactor = 0;
}

AxialARPlugin::~AxialARPlugin()
{

    delete execButton;
    delete simulationMenu;
    delete pinboardEntry;

    if (interactor)
        interactor->decRefCount();
}

void AxialARPlugin::tabletPressEvent(coTUIElement *item)
{
    if (item == restartSimulation)
        updateAndExec();
}

void AxialARPlugin::updateAndExec()
{
    if (interactor)
    {
        interactor->setSliderParam("blade_angle", -10.0, 5.0, getAngle());
        (interactor)->executeModule();
    }
}

void AxialARPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == restartSimulation)
        updateAndExec();
}

void AxialARPlugin::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == execButton)
        updateAndExec();
}

void AxialARPlugin::focusEvent(bool, coMenu *)
{
    //fprintf(stderr,"VRMenu::focusEvent\n");
}

void AxialARPlugin::addObject(RenderObject * container, osg::Group *parent, const RenderObject *obj, const RenderObject *, const RenderObject *, const RenderObject *)
{
    if (!obj)
        return;

    unsigned int i;
    //char *feedbackInfo = obj->getAttribute("FEEDBACK");
    char *feedbackInfo = obj->getAttribute("INTERACTOR");

    //printf("feedback [%s]\n", feedbackInfo);

    char moduleName[200];
    int moduleInstance;
    if (feedbackInfo)
    {
        char tmp[200];
        strncpy(tmp, feedbackInfo + 1, 200);
        int len = strlen(tmp);
        for (i = 0; i < len; i++)
            if (tmp[i] == '\n')
            {
                tmp[i] = ' ';
                break;
            }

        for (; i < len; i++)
            if (tmp[i] == '\n')
            {
                tmp[i] = '\0';
                break;
            }

        if (sscanf(tmp, "%s %d", moduleName, &moduleInstance) != 2)
            fprintf(stderr, "AxialARPlugin: sscanf for feedback info failed\n");
    }
}

void AxialARPlugin::removeObject(const char *, int)
{
    if (interactor)
    {
        interactor->decRefCount();
        interactor = NULL;
    }
}

void AxialARPlugin::preFrame()
{
    if (angleMarker && angleMarker->isVisible())
    {
        osg::Matrix MarkerPos; // marker position in camera coordinate system
        MarkerPos = angleMarker->getMarkerTrans();
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
        while (coord.hpr[0] < 0)
            coord.hpr[0] = 360.0 + coord.hpr[0];
        while (coord.hpr[0] > 360)
            coord.hpr[0] = coord.hpr[0] - 360.0;

        angle.push_back(coord.hpr[0]);

        if (angle.size() > 10)
            angle.pop_front();

        /*
      int numTS = coVRAnimationManager::instance()->getNumTimesteps();
      int newTS = ((int)(coord.hpr[0] / 360.0 * numTS)) % numTS;
      if(newTS < 0)
         newTS = numTS + newTS;
      if(newTS > numTS)
         newTS = newTS - numTS;
      cerr << "newTS: " << newTS << endl;
      
      coVRAnimationManager::instance()->setAnimationFrame(newTS);
*/
    }
}

void AxialARPlugin::feedback(coInteractor *i)
{
    if (interactor)
    {
        interactor->decRefCount();
        interactor = NULL;
    }

    interactor = i;
    interactor->incRefCount();
    //i->setSliderParam("blade_angle", -10, 5, 2);
}

float AxialARPlugin::getAngle()
{

    int num = 0;
    float val = 0;
    std::deque<float>::const_iterator i;

    for (i = angle.begin(); i != angle.end(); i++)
    {
        val += (*i);
        num++;
    }

    if (num > 0)
        return (val / (float)num);

    return -1;
}

COVERPLUGIN(AxialARPlugin)
