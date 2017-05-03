/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <PluginUtil/coSphere.h>
#include <cover/coInteractor.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>

#include <osg/MatrixTransform>
#include <osg/Geode>

#include <PluginUtil/BoxSelection.h>

#include "PickSpherePlugin.h"
#include <cover/RenderObject.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

#include <do/coDoSpheres.h>

//#define DEBUG

using namespace osg;
using covise::coRestraint;
using covise::coDistributedObject;
using covise::coDoSpheres;

BoxSelection *PickSpherePlugin::boxSelection = NULL;
PickSphereInteractor *PickSpherePlugin::s_pickSphereInteractor = NULL;
float PickSpherePlugin::s_scale = 1.f;

void
PickSpherePlugin::addObject(const RenderObject *container, osg::Group *, const RenderObject *geometry, const RenderObject *, const RenderObject *, const RenderObject *)
{
    if (geometry)
    {
        if (geometry->getAttribute("PICKSPHERE") && spheres.empty())
        {
            traceName = geometry->getName();
#ifdef DEBUG
            printf("geometry with PickSphereAttribute\n");
#endif
            char *sphereNames = strcpy(new char[strlen(geometry->getAttribute("PICKSPHERE")) + 1], geometry->getAttribute("PICKSPHERE"));
            char *name = strtok(sphereNames, "\n");
            while (name != NULL)
            {
                const coDistributedObject *tmp_obj = coDistributedObject::createFromShm(name);
                const coDoSpheres *obj = dynamic_cast<const coDoSpheres *>(tmp_obj);
                if (obj)
                {
                    getSphereData(obj);
                }
                name = strtok(NULL, "\n");
            }
        }
    }
}

void
PickSpherePlugin::getSphereData(const coDoSpheres *spheresObj)
{
    int size;
    spheresObj->getAddresses(&xpos, &ypos, &zpos, &radii);
    size = spheresObj->getNumElements();
    spheres.push_back(new SphereData(size, xpos, ypos, zpos, radii));
    addedSphereNames[spheresObj->getName()] = addedSphereNames.size();
    s_pickSphereInteractor->updateSpheres(&spheres);
}

//-----------------------------------------------------------------------------

PickSpherePlugin::PickSpherePlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n    new PickSpherePlugin\n");

    startParamName = "start";
    stopParamName = "stop";
    x_dimParamName = "x_dim";                           
    y_dimParamName = "y_dim";
    z_dimParamName = "z_dim";
    particlesParamName = "selection";
    UnsortedParticlesParamName = "UnsortedSelection";
    showTraceParamName = "traceParticle";
    regardInterruptParamName = "LeavingBoundingBox";

    animateViewerParamName = "animateViewer";
    animateLookAtParamName = "animLookAt";

    if (cover->debugLevel(4))
    {
        fprintf(stderr, "startParamName=[%s]\n", startParamName);
        fprintf(stderr, "stopParamName=[%s]\n", stopParamName);
        fprintf(stderr, "x_dimParamName=[%s]\n", x_dimParamName);                   
        fprintf(stderr, "y_dimParamName=[%s]\n", y_dimParamName);
        fprintf(stderr, "z_dimParamName=[%s]\n", z_dimParamName);
        fprintf(stderr, "particlesParamName=[%s]\n", particlesParamName);
        fprintf(stderr, "UnsortedParticlesParamName=[%s]\n", UnsortedParticlesParamName);
        fprintf(stderr, "showTraceParamName=[%s]\n", showTraceParamName);
        fprintf(stderr, "regardInterruptParamName=[%s]\n", regardInterruptParamName);
    }

    firsttime = true;
    maxTimestep = 0;

    sphereNames = NULL;
    pickSphereSubmenu = NULL;
    inter = NULL;
    pinboardButton = NULL;
    startSlider = NULL;
    stopSlider = NULL;
    x_dimSlider = NULL;                             
    y_dimSlider = NULL;                             
    z_dimSlider = NULL;                             
    opacityPoti = NULL;
    scalePoti = NULL;
    singlePickCheckbox = NULL;
    multiplePickCheckbox = NULL;
    showTraceCheckbox = NULL;
    regardInterruptCheckbox = NULL;
    attachViewerCheckbox = NULL;
    particleString = NULL;
    clearSelectionButton = NULL;
    clearPointButton = NULL;
    executeButton = NULL;
    PickSpherePlugin::boxSelection = NULL;
    animateViewer = 0;

    // create the cube interactor
    PickSpherePlugin::s_pickSphereInteractor = new PickSphereInteractor(coInteraction::ButtonA, "PickSphere", coInteraction::High);
}

PickSpherePlugin::~PickSpherePlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n    delete PickSpherePlugin\n");

    delete pinboardButton;
    pinboardButton = NULL;
    deleteSubmenu();
    delete sphereNames;
    delete PickSpherePlugin::s_pickSphereInteractor;
}

void
PickSpherePlugin::selectWithBox()
{
    float xmin, ymin, zmin;
    float xmax, ymax, zmax;
    PickSpherePlugin::boxSelection->getBox(xmin, ymin, zmin, xmax, ymax, zmax);
#ifdef DEBUG
    printf("%f, %f\n", xmin, xmax);
    printf("%f, %f\n", ymin, ymax);
    printf("%f, %f\n", zmin, zmax);
#endif
    osg::Vec3 *min = new Vec3(xmin, ymin, zmin);
    osg::Vec3 *max = new Vec3(xmax, ymax, zmax);

    PickSpherePlugin::s_pickSphereInteractor->boxSelect(*min, *max);
}

void
PickSpherePlugin::createSubmenu()
{
    // create the submenu and attach it to the pinboard button
    pickSphereSubmenu = new coRowMenu("PickSphere");
    pinboardButton->setMenu(pickSphereSubmenu);

    //label containing the selected particles information
    particleString = new coLabelMenuItem("");

    clearSelectionButton = new coButtonMenuItem("clear selection");
    clearSelectionButton->setMenuListener(this);

    clearPointButton = new coButtonMenuItem("clear last point");
    clearPointButton->setMenuListener(this);

    startSlider = new coSliderMenuItem("start", 0.0, 0.0, 0.0);
    startSlider->setInteger(true);
    startSlider->setMenuListener(this);

    stopSlider = new coSliderMenuItem("stop", 0.0, 0.0, 0.0);
    stopSlider->setInteger(true);
    stopSlider->setMenuListener(this);

    x_dimSlider = new coSliderMenuItem("x_dimGrid", 0.0, 0.0, 0.0);                     
    x_dimSlider->setInteger(true);
    x_dimSlider->setMenuListener(this);

    y_dimSlider = new coSliderMenuItem("y_dimGrid", 0.0, 0.0, 0.0);
    y_dimSlider->setInteger(true);
    y_dimSlider->setMenuListener(this);

    z_dimSlider = new coSliderMenuItem("z_dimGrid", 0.0, 0.0, 0.0);
    z_dimSlider->setInteger(true);
    z_dimSlider->setMenuListener(this);                                 

    opacityPoti = new coPotiMenuItem("sphere opacity", 0.0, 1.0, 1.0);
    opacityPoti->setMenuListener(this);

    scalePoti = new coPotiMenuItem("sphere scale", -2.0, 2.0, 0.0);
    scalePoti->setMenuListener(this);

    // groupPointerArray[0] returns the pointer to checkboxes
    // concerning the navigation --> any navigation gets deactivated
    // if any selection mode gets activated
    singlePickCheckbox = new coCheckboxMenuItem("single select", false, groupPointerArray[0]);
    singlePickCheckbox->setMenuListener(this);

    multiplePickCheckbox = new coCheckboxMenuItem("multiple select", false, groupPointerArray[0]);
    multiplePickCheckbox->setMenuListener(this);

    showTraceCheckbox = new coCheckboxMenuItem("show trace", true, NULL);
    showTraceCheckbox->setMenuListener(this);

    regardInterruptCheckbox = new coCheckboxMenuItem("regard interrupt", true, NULL);
    regardInterruptCheckbox->setMenuListener(this);

    attachViewerCheckbox = new coCheckboxMenuItem("attach viewer", false, NULL);
    attachViewerCheckbox->setMenuListener(this);

    executeButton = new coButtonMenuItem("execute");
    executeButton->setMenuListener(this);

    PickSpherePlugin::boxSelection = new BoxSelection(pickSphereSubmenu, "box selection", "box selection");
    PickSpherePlugin::boxSelection->setMenuListener(this);
    PickSpherePlugin::boxSelection->registerInteractionFinishedCallback(PickSpherePlugin::selectWithBox);

    // add all elements to menu
    pickSphereSubmenu->add(showTraceCheckbox);
    //pickSphereSubmenu->add(regardInterruptCheckbox);
    pickSphereSubmenu->add(opacityPoti);
    pickSphereSubmenu->add(scalePoti);
    pickSphereSubmenu->add(particleString);
    pickSphereSubmenu->add(clearSelectionButton);
    pickSphereSubmenu->add(clearPointButton);
    pickSphereSubmenu->add(startSlider);
    pickSphereSubmenu->add(stopSlider);
    pickSphereSubmenu->add(x_dimSlider);                            
    pickSphereSubmenu->add(y_dimSlider);
    pickSphereSubmenu->add(z_dimSlider);
    pickSphereSubmenu->add(singlePickCheckbox);
    pickSphereSubmenu->add(multiplePickCheckbox);
    pickSphereSubmenu->add(PickSpherePlugin::boxSelection->getCheckbox());
    //pickSphereSubmenu->add(PickSpherePlugin::boxSelection->getSubMenu());
    pickSphereSubmenu->add(executeButton);
    pickSphereSubmenu->add(attachViewerCheckbox);
}

void
PickSpherePlugin::deleteSubmenu()
{
    delete startSlider;
    startSlider = NULL;

    delete stopSlider;
    stopSlider = NULL;

    delete x_dimSlider;                                     
    x_dimSlider = NULL;

    delete y_dimSlider;
    y_dimSlider = NULL;

    delete z_dimSlider;
    z_dimSlider = NULL;                                     

    singlePickCheckbox->setState(false, true);
    delete singlePickCheckbox;
    singlePickCheckbox = NULL;

    multiplePickCheckbox->setState(false, true);
    delete multiplePickCheckbox;
    multiplePickCheckbox = NULL;

    boxSelection->getCheckbox()->setState(false, true);
    delete boxSelection;
    boxSelection = NULL;

    attachViewerCheckbox->setState(false, true);
    delete attachViewerCheckbox;
    attachViewerCheckbox = NULL;

    delete pickSphereSubmenu;
    pickSphereSubmenu = NULL;
}

void
PickSpherePlugin::newInteractor(const RenderObject *, coInteractor *in)
{
    if (strcmp(in->getPluginName(), "PickSphere") != 0)
        return;

#ifdef DEBUG
    printf("PickSpherePlugin::VRNewInteractor(%s)\n", in->getObject()->getName());
#endif

    if (in->getObject()->getAttribute("MODULE") && strcmp(in->getObject()->getAttribute("MODULE"), "PickSphere") == 0)
    {
        if (inter)
        {
            inter->decRefCount();
            inter = NULL;
        }
        // save the interactor for feedback
        inter = in;
        inter->incRefCount();
    }
    const char *particles;
    int showTrace, regardInterrupt;
    int animLookAtComponents;
    float *animateLookAt;

    if (cover->debugLevel(3))
        fprintf(stderr, "\n    PickSpherePlugin::add\n");

    if (firsttime)
    {
        firsttime = false;

        if (cover->debugLevel(4))
            fprintf(stderr, "firsttime\n");

        // create the button for the pinboard
        pinboardButton = new coSubMenuItem("PickSphere...");

        // create submenu
        createSubmenu();

        // add the button to the pinboard
        cover->getMenu()->add(pinboardButton);
    }

    // get the start, stop and particle parameters of the  module
    in->getIntSliderParam(startParamName, min, max, start);
    in->getIntSliderParam(stopParamName, min, max, stop);
    in->getIntSliderParam(x_dimParamName, Min, Max, x_dim);                 
    in->getIntSliderParam(y_dimParamName, Min, Max, y_dim);
    in->getIntSliderParam(z_dimParamName, Min, Max, z_dim);
    in->getStringParam(particlesParamName, particles);
    in->getStringParam(UnsortedParticlesParamName, particles);
    in->getBooleanParam(showTraceParamName, showTrace);
    in->getBooleanParam(regardInterruptParamName, regardInterrupt);

    in->getChoiceParam(animateViewerParamName, animateViewerNumValues, animateViewerValueNames, animateViewer);
    in->getFloatVectorParam(animateLookAtParamName, animLookAtComponents, animateLookAt);
    animLookAt = osg::Vec3(animateLookAt[0], animateLookAt[1], animateLookAt[2]);

    if (cover->debugLevel(4))
    {
        fprintf(stderr, "start=%d\n", start);
        fprintf(stderr, "stop=%d\n", stop);
        fprintf(stderr, "x_dim=%d\n", x_dim);                           
        fprintf(stderr, "y_dim=%d\n", y_dim);
        fprintf(stderr, "z_dim=%d\n", z_dim);
        fprintf(stderr, "particles=%s\n", particles);
        fprintf(stderr, "showTrace=%d\n", showTrace);
        fprintf(stderr, "regardInterrupt=%d\n", regardInterrupt);
        fprintf(stderr, "animateViewer=%d/%d (%s)\n", animateViewer, animateViewerNumValues, animateViewerValueNames[animateViewer]);
        fprintf(stderr, "animateLookAt=(%f %f %f)\n", animateLookAt[0], animateLookAt[1], animateLookAt[2]);
    }

    //irgendwas um die ausgewhlten Kugeln zu markieren
    s_pickSphereInteractor->updateSelection(particles);

    // update start and stop values
    startSlider->setMin(min);
    startSlider->setMax(max);
    startSlider->setValue(start);

    stopSlider->setMin(min);
    stopSlider->setMax(max);
    stopSlider->setValue(stop);

    //Dimension of Grid (Output)
    x_dimSlider->setMin(Min);                                   
    x_dimSlider->setMax(Max);
    x_dimSlider->setValue(x_dim);

    y_dimSlider->setMin(Min);
    y_dimSlider->setMax(Max);
    y_dimSlider->setValue(y_dim);

    z_dimSlider->setMin(Min);
    z_dimSlider->setMax(Max);
    z_dimSlider->setValue(z_dim);                               

    setParticleStringLabel();

    showTraceCheckbox->setState(showTrace);
    regardInterruptCheckbox->setState(regardInterrupt);

    attachViewerCheckbox->setState(animateViewer != 0);
}

void
PickSpherePlugin::removeObject(const char *objName, bool replace)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n    PickSpherePlugin::remove\n");

    if (traceName == objName || addedSphereNames.find(objName) != addedSphereNames.end())
    {
        clearTempSpheres();
        if (addedSphereNames.find(objName) != addedSphereNames.end())
        {
            addedSphereNames.erase(objName);
        }

        if (!replace)
        {
            cover->getMenu()->remove(pinboardButton); // braucht man das ?
            delete pinboardButton;
            pinboardButton = NULL;
            deleteSubmenu();
            firsttime = true;
            animateViewerNumValues = 0;
            animateViewerValueNames = NULL;
            if (inter)
            {
                inter->decRefCount();
                inter = NULL;
            }
        }
    }
}

/*function removes all elements of the temporary vector,
  i.e. the created elements and the pointers to these elements*/
void
PickSpherePlugin::clearTempSpheres()
{
    while (!spheres.empty())
    {
        delete (spheres.back());
        spheres.pop_back();
    }
}

void
PickSpherePlugin::setParticleStringLabel()
{
    int count = 0;
    std::string selectedParticlesString = "";
    std::string UnsortedSelectedParticlesString = "";
    if (coVRMSController::instance()->isMaster())
    {
        count = s_pickSphereInteractor->getSelectedParticleCount();
        coVRMSController::instance()->sendSlaves((char *)&count, sizeof(int));
       
        selectedParticlesString = s_pickSphereInteractor->getSelectedParticleString();
        UnsortedSelectedParticlesString = s_pickSphereInteractor->getUnsortedSelectedParticleString();
        int length = selectedParticlesString.length();
        coVRMSController::instance()->sendSlaves((char *)&length, sizeof(length));
        coVRMSController::instance()->sendSlaves(selectedParticlesString.c_str(), length + 1);
        coVRMSController::instance()->sendSlaves(UnsortedSelectedParticlesString.c_str(), length + 1);
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&count, sizeof(int));
        int length;
        coVRMSController::instance()->readMaster((char *)&length, sizeof(length));
        char *charString = new char[length + 1];
        coVRMSController::instance()->readMaster(charString, length + 1);
        selectedParticlesString = string(charString);
        UnsortedSelectedParticlesString = string(charString);
        delete[] charString;
    }
    const char *selectedParticles = selectedParticlesString.c_str();
    const char *UnsortedSelectedParticles = UnsortedSelectedParticlesString.c_str();
    string ss = "selection:";

    if (singlePickCheckbox->getState())
    {
        ss = "selection: " + selectedParticlesString;
    }
    else
    {
        std::ostringstream countStream;
        countStream << count;
        ss = "selection count: " + countStream.str();
    }
    particleString->setLabel(ss.c_str());
    if (inter)
        inter->setStringParam(particlesParamName, selectedParticles);
        inter->setStringParam(UnsortedParticlesParamName, UnsortedSelectedParticles);
}

void
PickSpherePlugin::preFrame()
{
    bool selected = s_pickSphereInteractor->selectedWithBox();
    if ((s_pickSphereInteractor->wasStopped() || selected) && s_pickSphereInteractor->selectionChanged())
    {
        if (selected)
            s_pickSphereInteractor->setSelectedWithBox(false);
        setParticleStringLabel();
        if (!selected)
            getInteractor()->executeModule();
    }

    if (!animateViewer)
        return;

    int currentFrame = coVRAnimationManager::instance()->getAnimationFrame();
    static int lastFrame = -1, refFrame = -1;
    //bool frameChanged = false;
    if (refFrame < 0)
        refFrame = 0;
    if (lastFrame != currentFrame)
    {
        //frameChanged = true;
        refFrame = lastFrame;
        lastFrame = currentFrame;
    }
    if (currentFrame < 0 || currentFrame >= int(spheres.size()))
        return;
    const SphereData *curSpheres = spheres[currentFrame];

    const coRestraint &selection = s_pickSphereInteractor->getSelection();
    int firstIndex = selection.lower();                                 
    if (firstIndex < 0 || firstIndex >= curSpheres->n)
        return;
    float x[3] = { curSpheres->x[firstIndex], curSpheres->y[firstIndex], curSpheres->z[firstIndex] };

    osg::Matrix mat = cover->getXformMat();
    osg::Vec3 pos(x[0], x[1], x[2]);
    osg::Vec3 lastPos(0., 0., 0.);

    if (refFrame >= 0 && refFrame < int(spheres.size()))
    {
        const SphereData *s = spheres[refFrame];
        lastPos = osg::Vec3(s->x[firstIndex], s->y[firstIndex], s->z[firstIndex]);
    }

    osg::Vec3 focus(0., 0., 0.);
    pos = osg::Vec3(x[0], x[1], x[2]);   
    switch (animateViewer)
    {
        case 0:
            // off - we don't get here
            break;
        case 1:
            // don't change viewing direction
            mat = cover->getXformMat();
            mat.setTrans(osg::Vec3(0., 0., 0.));
            break;
        case 2:
            // focus animLookAt
            focus = animLookAt;
            break;
        case 3:
            // look into direction of movement
            focus = pos + pos - lastPos;
            break;
        case 4:
            // look back
            focus = lastPos;
            break;
    }

    if (animateViewer >= 2)
    {
        osg::Vec3 dir = pos - focus;
        dir.normalize();
#ifdef DEBUG
        if (frameChanged)
        {
            fprintf(stderr, "%d last=(%f %f %f)\n", currentFrame, lastPos[0], lastPos[1], lastPos[2]);
            fprintf(stderr, "%d pos= (%f %f %f)\n", currentFrame, pos[0], pos[1], pos[2]);
            fprintf(stderr, "%d dir= (%f %f %f)\n", currentFrame, dir[0], dir[1], dir[2]);
        }
#endif
        osg::Vec3 viewerDir = osg::Vec3(0., -1., 0.);
        osg::Quat quat;
        quat.makeRotate(dir, viewerDir);
        osg::Matrix rot;
        quat.get(rot);

        mat = rot;
    }
    pos = mat.preMult(osg::Vec3(x[0], x[1], x[2]));
    pos *= cover->getScale();
    pos *= -1.;

#ifdef DEBUG
    if (frameChanged)
        fprintf(stderr, "%d trans pos=(%f %f %f)\n", currentFrame, pos[0], pos[1], pos[2]);
#endif
    mat.setTrans(pos);
    cover->setXformMat(mat);
}

void
PickSpherePlugin::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "PickSpherePlugin::menuEvent for %s\n", menuItem->getName());

    if (menuItem == startSlider)
    {
        if (inter)
            inter->setSliderParam(startParamName, (int)startSlider->getMin(), (int)stopSlider->getMax(), (int)startSlider->getValue());
        if (startSlider->getValue() > stopSlider->getValue())
            stopSlider->setValue(startSlider->getValue());
    }
    else if (menuItem == stopSlider)
    {
        if (inter)
            inter->setSliderParam(stopParamName, (int)stopSlider->getMin(), (int)stopSlider->getMax(), (int)stopSlider->getValue());
        if (stopSlider->getValue() < startSlider->getValue())
            startSlider->setValue(stopSlider->getValue());
    }
    else if (menuItem == x_dimSlider)                               
    {
        if (inter)
            inter->setSliderParam(x_dimParamName, (int)x_dimSlider->getMin(), (int)x_dimSlider->getMax(), (int)x_dimSlider->getValue());
    }
    else if (menuItem == y_dimSlider)
    {
        if (inter)
            inter->setSliderParam(y_dimParamName, (int)y_dimSlider->getMin(), (int)y_dimSlider->getMax(), (int)y_dimSlider->getValue());
    }
    else if (menuItem == z_dimSlider)
    {
        if (inter)
            inter->setSliderParam(z_dimParamName, (int)z_dimSlider->getMin(), (int)z_dimSlider->getMax(), (int)z_dimSlider->getValue());
    }                                               
    else if (menuItem == opacityPoti)
        coSphere::setTransparency(opacityPoti->getValue());
    else if (menuItem == scalePoti)
    {
        coSphere::setScale(powf(10., scalePoti->getValue()));
        s_scale = powf(10., scalePoti->getValue());
    }
    else if (menuItem == singlePickCheckbox)
    {
        if (singlePickCheckbox->getState())
        {
            //enable interaction
            coInteractionManager::the()->registerInteraction(s_pickSphereInteractor);
            //pickSphereSubmenu->remove(clearSelectionButton);
            //pickSphereSubmenu->remove(executeButton);
            s_pickSphereInteractor->enableMultipleSelect(false);
        }
        else
        {
            coInteractionManager::the()->unregisterInteraction(s_pickSphereInteractor);
        }
    }
    else if (menuItem == multiplePickCheckbox)
    {
        if (multiplePickCheckbox->getState())
        {
            coInteractionManager::the()->registerInteraction(s_pickSphereInteractor);
            s_pickSphereInteractor->enableMultipleSelect(true);
            //pickSphereSubmenu->insert(clearSelectionButton, 3);
            //pickSphereSubmenu->add(executeButton);
        }
        else
        {
            coInteractionManager::the()->unregisterInteraction(s_pickSphereInteractor);
            s_pickSphereInteractor->enableMultipleSelect(false);
            //pickSphereSubmenu->remove(clearSelectionButton);
            //pickSphereSubmenu->remove(executeButton);
        }
    }
    else if (menuItem == showTraceCheckbox)
    {
        if (inter)
        {
            inter->setBooleanParam(showTraceParamName, (bool)showTraceCheckbox->getState());
            inter->executeModule();
        }
    }
    else if (menuItem == regardInterruptCheckbox)
    {
        if (inter)
        {
            inter->setBooleanParam(regardInterruptParamName, (bool)regardInterruptCheckbox->getState());
            inter->executeModule();
        }
    }
    else if (menuItem == executeButton)
    {
        if (inter)
            inter->executeModule();
    }
    else if (menuItem == clearSelectionButton)
    {
        particleString->setLabel("selection count: 0");
        if (inter)
            inter->setStringParam(particlesParamName, "");
            inter->setStringParam(UnsortedParticlesParamName, "");
        s_pickSphereInteractor->updateSelection("");
        if (inter)
            inter->executeModule();
    }

    else if (menuItem == clearPointButton)
    {
        int count = s_pickSphereInteractor->getSelectedParticleCount();
        if (count != 0)
        {
            count = 0;
            std::string selectedParticlesString = "";
            std::string UnsortedSelectedParticlesString = "";
            if (coVRMSController::instance()->isMaster())
            {
                count = s_pickSphereInteractor->getReducedSelectedParticleCount();
                coVRMSController::instance()->sendSlaves((char *)&count, sizeof(int));
                
                selectedParticlesString = s_pickSphereInteractor->getSelectedParticleString();
                UnsortedSelectedParticlesString = s_pickSphereInteractor->getUnsortedSelectedParticleString();
                int length = selectedParticlesString.length();
                coVRMSController::instance()->sendSlaves((char *)&length, sizeof(length));
                coVRMSController::instance()->sendSlaves(selectedParticlesString.c_str(), length + 1);
                coVRMSController::instance()->sendSlaves(UnsortedSelectedParticlesString.c_str(), length + 1);
            }
            else
            {
                coVRMSController::instance()->readMaster((char *)&count, sizeof(int));
                int length;
                coVRMSController::instance()->readMaster((char *)&length, sizeof(length));
                char *charString = new char[length + 1];
                coVRMSController::instance()->readMaster(charString, length + 1);
                selectedParticlesString = string(charString);
                UnsortedSelectedParticlesString = string(charString);
                delete[] charString;
            }
            const char *selectedParticles = selectedParticlesString.c_str();
            const char *UnsortedSelectedParticles = UnsortedSelectedParticlesString.c_str();
            string ss = "selection:";

            if (singlePickCheckbox->getState())
            {
                ss = "selection: " + selectedParticlesString;
            }
            else
            {
                std::ostringstream countStream;
                countStream << count;
                ss = "selection count: " + countStream.str();
            }
            particleString->setLabel(ss.c_str());
            if (inter)
                inter->setStringParam(particlesParamName, selectedParticles);
                inter->setStringParam(UnsortedParticlesParamName, UnsortedSelectedParticles);
        }
        else
        {
            particleString->setLabel("selection count: 0");
            if (inter)
                inter->setStringParam(particlesParamName, "");
                inter->setStringParam(UnsortedParticlesParamName, "");
            s_pickSphereInteractor->updateSelection("");
            if (inter)
                inter->executeModule();
        }
    }

    else if (menuItem == PickSpherePlugin::boxSelection)
    {
        if (PickSpherePlugin::boxSelection->getCheckboxState())
        {
            //pickSphereSubmenu->insert(clearSelectionButton, 3);
            //pickSphereSubmenu->add(executeButton);
        }
        else
        {
            //pickSphereSubmenu->remove(clearSelectionButton);
            //pickSphereSubmenu->remove(executeButton);
        }
    }
    else if (menuItem == attachViewerCheckbox)
    {
        if (inter)
        {
            int newstate = attachViewerCheckbox->getState() ? 1 : 0;
            if (newstate < animateViewerNumValues)
            {
                inter->setChoiceParam(animateViewerParamName, animateViewerNumValues, animateViewerValueNames, newstate);
                inter->executeModule();
            }
        }
    }
    else
    {
        fprintf(stderr, "no suitable menuItem\n");
    }
}

void
PickSpherePlugin::menuReleaseEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "PickSpherePlugin::menuReleaseEvent for %s\n", menuItem->getName());
    if (menuItem == startSlider || menuItem == stopSlider)
    {
        if (((int)stopSlider->getValue()) != stop || ((int)startSlider->getValue()) != start)
            if (inter)
                inter->executeModule();
    }
}

coMenuItem *
PickSpherePlugin::getMenuButton(const std::string &name)
{
    if (name == "SingleSelect")
        return singlePickCheckbox;
    else if (name == "MultiSelect")
        return multiplePickCheckbox;
    else if (name == "BoxSelect")
    {
        if (boxSelection)
            return boxSelection->getCheckbox();
    }
    else if (name == "ClearSelection")
    {
        return clearSelectionButton;
    }
    else if (name == "ClearPoint")
    {
        return clearPointButton;
    }
    else if (name == "AttachViewer")
    {
        return attachViewerCheckbox;
    }

    return NULL;
}

COVERPLUGIN(PickSpherePlugin)
