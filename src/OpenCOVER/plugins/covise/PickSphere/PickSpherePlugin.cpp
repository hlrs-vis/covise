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
#include <cover/coVRNavigationManager.h>
#include <OpenVRUI/coInteractionManager.h>

#include <osg/MatrixTransform>
#include <osg/Geode>

#include <PluginUtil/BoxSelection.h>

#include "PickSpherePlugin.h"
#include <cover/RenderObject.h>

#ifdef VRUI
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#else
#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#endif

#include <do/coDoSpheres.h>

//#define DEBUG

using namespace osg;
using covise::coRestraint;
using covise::coDistributedObject;
using covise::coDoSpheres;
using vrui::coInteraction;

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
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("PickSpherePlugin", cover->ui)
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
    pickSphereSubmenu = new ui::Menu("PickSphere", this);

    //label containing the selected particles information
    particleString = new ui::Label(pickSphereSubmenu, "ParticleSelection");

    clearSelectionButton = new ui::Action(pickSphereSubmenu, "ClearSelection");
    clearSelectionButton->setText("Clear selection");
    clearSelectionButton->setCallback([this](){
        particleString->setText("selection count: 0");
        if (inter)
            inter->setStringParam(particlesParamName, "");
            inter->setStringParam(UnsortedParticlesParamName, "");
        s_pickSphereInteractor->updateSelection("");
        if (inter)
            inter->executeModule();
    });

    clearPointButton = new ui::Action(pickSphereSubmenu, "ClearLastPoint");
    clearPointButton->setText("Clear last point");
    clearPointButton->setCallback([this](){
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

            if (singlePickCheckbox->state())
            {
                ss = "selection: " + selectedParticlesString;
            }
            else
            {
                std::ostringstream countStream;
                countStream << count;
                ss = "selection count: " + countStream.str();
            }
            particleString->setText(ss.c_str());
            if (inter)
                inter->setStringParam(particlesParamName, selectedParticles);
                inter->setStringParam(UnsortedParticlesParamName, UnsortedSelectedParticles);
        }
        else
        {
            particleString->setText("selection count: 0");
            if (inter)
                inter->setStringParam(particlesParamName, "");
                inter->setStringParam(UnsortedParticlesParamName, "");
            s_pickSphereInteractor->updateSelection("");
            if (inter)
                inter->executeModule();
        }
    });

    startSlider = new ui::Slider(pickSphereSubmenu, "Start");
    startSlider->setBounds(0.0, 0.0);
    startSlider->setValue(0.0);
    startSlider->setIntegral(true);
    startSlider->setCallback([this](double value, bool released){
        if (inter)
            inter->setSliderParam(startParamName, (int)startSlider->min(), (int)stopSlider->max(), (int)startSlider->value());
        if (startSlider->value() > stopSlider->value())
            stopSlider->setValue(startSlider->value());
    });

    stopSlider = new ui::Slider(pickSphereSubmenu, "Stop");
    stopSlider->setBounds(0.0, 0.0);
    stopSlider->setValue(0.0);
    stopSlider->setIntegral(true);
    stopSlider->setCallback([this](double value, bool released){
        if (inter)
            inter->setSliderParam(stopParamName, (int)stopSlider->min(), (int)stopSlider->max(), (int)stopSlider->value());
        if (stopSlider->value() < startSlider->value())
            startSlider->setValue(stopSlider->value());
    });

    x_dimSlider = new ui::Slider(pickSphereSubmenu, "x_dimGrid");
    x_dimSlider->setBounds(0.0, 0.0);
    x_dimSlider->setValue(0.0);
    x_dimSlider->setIntegral(true);
    x_dimSlider->setCallback([this](double value, bool released){
        if (inter)
            inter->setSliderParam(x_dimParamName, (int)x_dimSlider->min(), (int)x_dimSlider->max(), (int)x_dimSlider->value());
    });

    y_dimSlider = new ui::Slider(pickSphereSubmenu, "y_dimGrid");
    y_dimSlider->setBounds(0.0, 0.0);
    y_dimSlider->setValue(0.0);
    y_dimSlider->setIntegral(true);
    y_dimSlider->setCallback([this](double value, bool released){
        if (inter)
            inter->setSliderParam(y_dimParamName, (int)y_dimSlider->min(), (int)y_dimSlider->max(), (int)y_dimSlider->value());
    });

    y_dimSlider = new ui::Slider(pickSphereSubmenu, "z_dimGrid");
    y_dimSlider->setBounds(0.0, 0.0);
    y_dimSlider->setValue(0.0);
    y_dimSlider->setIntegral(true);
    z_dimSlider->setCallback([this](double value, bool released){
        if (inter)
            inter->setSliderParam(z_dimParamName, (int)z_dimSlider->min(), (int)z_dimSlider->max(), (int)z_dimSlider->value());
    });

    opacityPoti = new ui::Slider(pickSphereSubmenu, "SphereOpacity");
    opacityPoti->setPresentation(ui::Slider::AsDial);
    opacityPoti->setText("Sphere opacity");
    opacityPoti->setBounds(0.0, 1.0);
    opacityPoti->setValue(1.0);
    opacityPoti->setCallback([this](double value, bool released){
        coSphere::setTransparency(opacityPoti->value());
    });

    scalePoti = new ui::Slider(pickSphereSubmenu, "SphereScale");
    scalePoti->setPresentation(ui::Slider::AsDial);
    scalePoti->setText("Sphere scale");
    scalePoti->setBounds(-2.0, 2.0);
    scalePoti->setValue(0.0);
    scalePoti->setCallback([this](double value, bool released){
        coSphere::setScale(powf(10., scalePoti->value()));
        s_scale = powf(10., scalePoti->value());
    });

    // groupPointerArray[0] returns the pointer to checkboxes
    // concerning the navigation --> any navigation gets deactivated
    // if any selection mode gets activated
    singlePickCheckbox = new ui::Button(pickSphereSubmenu, "SingleSelect");
    singlePickCheckbox->setText("Single select");
    singlePickCheckbox->setState(false);
    singlePickCheckbox->setGroup(cover->navGroup(), coVRNavigationManager::NavOther);
    singlePickCheckbox->setCallback([this](bool state){
        if (state)
        {
            //enable interaction
            vrui::coInteractionManager::the()->registerInteraction(s_pickSphereInteractor);
            //pickSphereSubmenu->remove(clearSelectionButton);
            //pickSphereSubmenu->remove(executeButton);
            s_pickSphereInteractor->enableMultipleSelect(false);
        }
        else
        {
            vrui::coInteractionManager::the()->unregisterInteraction(s_pickSphereInteractor);
        }
    });

    multiplePickCheckbox = new ui::Button(pickSphereSubmenu, "MultiSelect");
    multiplePickCheckbox->setText("Multiple select");
    multiplePickCheckbox->setState(false);
    multiplePickCheckbox->setGroup(cover->navGroup(), coVRNavigationManager::NavOther);
    multiplePickCheckbox->setCallback([this](bool state){
        if (state)
        {
            vrui::coInteractionManager::the()->registerInteraction(s_pickSphereInteractor);
            s_pickSphereInteractor->enableMultipleSelect(true);
            //pickSphereSubmenu->insert(clearSelectionButton, 3);
            //pickSphereSubmenu->add(executeButton);
        }
        else
        {
            vrui::coInteractionManager::the()->unregisterInteraction(s_pickSphereInteractor);
            s_pickSphereInteractor->enableMultipleSelect(false);
            //pickSphereSubmenu->remove(clearSelectionButton);
            //pickSphereSubmenu->remove(executeButton);
        }
    });

    showTraceCheckbox = new ui::Button(pickSphereSubmenu, "ShowTrace");
    showTraceCheckbox->setText("Show trace");
    showTraceCheckbox->setState(true);
    showTraceCheckbox->setCallback([this](bool state){
        if (inter)
        {
            inter->setBooleanParam(showTraceParamName, state);
            inter->executeModule();
        }
    });

    regardInterruptCheckbox = new ui::Button(pickSphereSubmenu, "RegardInterrupt");
    regardInterruptCheckbox->setText("Regard interrupt");
    regardInterruptCheckbox->setState(true);
    regardInterruptCheckbox->setCallback([this](bool state){
        if (inter)
        {
            inter->setBooleanParam(regardInterruptParamName, state);
            inter->executeModule();
        }
    });

    attachViewerCheckbox = new ui::Button(pickSphereSubmenu, "AttachViewer");
    attachViewerCheckbox->setText("Attach viewer");
    attachViewerCheckbox->setState(false);
    attachViewerCheckbox->setCallback([this](bool state){
        if (inter)
        {
            int newstate = state ? 1 : 0;
            if (newstate < animateViewerNumValues)
            {
                inter->setChoiceParam(animateViewerParamName, animateViewerNumValues, animateViewerValueNames, newstate);
                inter->executeModule();
            }
        }
    });

    executeButton = new ui::Action(pickSphereSubmenu, "Execute");
    executeButton->setCallback([this](){
        if (inter)
            inter->executeModule();
    });

    PickSpherePlugin::boxSelection = new BoxSelection(pickSphereSubmenu, "BoxSelection", "Box selection");
    PickSpherePlugin::boxSelection->registerInteractionFinishedCallback(PickSpherePlugin::selectWithBox);
#if 0
    PickSpherePlugin::boxSelection->gutButton()->setCallback([this](bool state){
        if (state)
        {
            //pickSphereSubmenu->insert(clearSelectionButton, 3);
            //pickSphereSubmenu->add(executeButton);
        }
        else
        {
            //pickSphereSubmenu->remove(clearSelectionButton);
            //pickSphereSubmenu->remove(executeButton);
        }
    });
#endif

#ifdef VRUI
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
    pickSphereSubmenu->add(PickSpherePlugin::boxSelection->getButton());
    //pickSphereSubmenu->add(PickSpherePlugin::boxSelection->getSubMenu());
    pickSphereSubmenu->add(executeButton);
    pickSphereSubmenu->add(attachViewerCheckbox);
#endif
}

void
PickSpherePlugin::deleteSubmenu()
{
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

        // create submenu
        createSubmenu();
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
    startSlider->setBounds(min, max);
    startSlider->setValue(start);

    stopSlider->setBounds(min, max);
    stopSlider->setValue(stop);

    //Dimension of Grid (Output)
    x_dimSlider->setBounds(Min, Max);
    x_dimSlider->setValue(x_dim);

    y_dimSlider->setBounds(Min, Max);
    y_dimSlider->setValue(y_dim);

    z_dimSlider->setBounds(Min, Max);
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

    if (singlePickCheckbox->state())
    {
        ss = "selection: " + selectedParticlesString;
    }
    else
    {
        std::ostringstream countStream;
        countStream << count;
        ss = "selection count: " + countStream.str();
    }
    particleString->setText(ss.c_str());
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

#ifdef VRUI
void
PickSpherePlugin::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "PickSpherePlugin::menuEvent for %s\n", menuItem->getName());

    if (menuItem == startSlider)
    {
        if (inter)
            inter->setSliderParam(startParamName, (int)startSlider->min(), (int)stopSlider->max(), (int)startSlider->value());
        if (startSlider->value() > stopSlider->value())
            stopSlider->setValue(startSlider->value());
    }
    else if (menuItem == stopSlider)
    {
        if (inter)
            inter->setSliderParam(stopParamName, (int)stopSlider->min(), (int)stopSlider->max(), (int)stopSlider->value());
        if (stopSlider->value() < startSlider->value())
            startSlider->setValue(stopSlider->value());
    }
    else if (menuItem == x_dimSlider)                               
    {
        if (inter)
            inter->setSliderParam(x_dimParamName, (int)x_dimSlider->min(), (int)x_dimSlider->max(), (int)x_dimSlider->value());
    }
    else if (menuItem == y_dimSlider)
    {
        if (inter)
            inter->setSliderParam(y_dimParamName, (int)y_dimSlider->min(), (int)y_dimSlider->max(), (int)y_dimSlider->value());
    }
    else if (menuItem == z_dimSlider)
    {
        if (inter)
            inter->setSliderParam(z_dimParamName, (int)z_dimSlider->min(), (int)z_dimSlider->max(), (int)z_dimSlider->value());
    }                                               
    else if (menuItem == opacityPoti)
        coSphere::setTransparency(opacityPoti->value());
    else if (menuItem == scalePoti)
    {
        coSphere::setScale(powf(10., scalePoti->value()));
        s_scale = powf(10., scalePoti->value());
    }
    else if (menuItem == singlePickCheckbox)
    {
        if (singlePickCheckbox->state())
        {
            //enable interaction
            vrui::coInteractionManager::the()->registerInteraction(s_pickSphereInteractor);
            //pickSphereSubmenu->remove(clearSelectionButton);
            //pickSphereSubmenu->remove(executeButton);
            s_pickSphereInteractor->enableMultipleSelect(false);
        }
        else
        {
            vrui::coInteractionManager::the()->unregisterInteraction(s_pickSphereInteractor);
        }
    }
    else if (menuItem == multiplePickCheckbox)
    {
        if (multiplePickCheckbox->state())
        {
            vrui::coInteractionManager::the()->registerInteraction(s_pickSphereInteractor);
            s_pickSphereInteractor->enableMultipleSelect(true);
            //pickSphereSubmenu->insert(clearSelectionButton, 3);
            //pickSphereSubmenu->add(executeButton);
        }
        else
        {
            vrui::coInteractionManager::the()->unregisterInteraction(s_pickSphereInteractor);
            s_pickSphereInteractor->enableMultipleSelect(false);
            //pickSphereSubmenu->remove(clearSelectionButton);
            //pickSphereSubmenu->remove(executeButton);
        }
    }
    else if (menuItem == showTraceCheckbox)
    {
        if (inter)
        {
            inter->setBooleanParam(showTraceParamName, (bool)showTraceCheckbox->state());
            inter->executeModule();
        }
    }
    else if (menuItem == regardInterruptCheckbox)
    {
        if (inter)
        {
            inter->setBooleanParam(regardInterruptParamName, (bool)regardInterruptCheckbox->state());
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
        particleString->setText("selection count: 0");
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

            if (singlePickCheckbox->state())
            {
                ss = "selection: " + selectedParticlesString;
            }
            else
            {
                std::ostringstream countStream;
                countStream << count;
                ss = "selection count: " + countStream.str();
            }
            particleString->setText(ss.c_str());
            if (inter)
                inter->setStringParam(particlesParamName, selectedParticles);
                inter->setStringParam(UnsortedParticlesParamName, UnsortedSelectedParticles);
        }
        else
        {
            particleString->setText("selection count: 0");
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
            int newstate = attachViewerCheckbox->state() ? 1 : 0;
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
        if (((int)stopSlider->value()) != stop || ((int)startSlider->value()) != start)
            if (inter)
                inter->executeModule();
    }
}
#endif

#ifdef VRUI
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
#endif

COVERPLUGIN(PickSpherePlugin)
