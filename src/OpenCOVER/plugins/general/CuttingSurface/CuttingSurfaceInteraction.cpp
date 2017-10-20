/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CuttingSurfaceInteraction.h"
#include "CuttingSurfacePlane.h"
#include "CuttingSurfaceCylinder.h"
#include "CuttingSurfaceSphere.h"
#include "CuttingSurfacePlugin.h"

#ifdef VRUI
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#else
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Slider.h>
#include <cover/ui/SelectionList.h>
#endif

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coInteractor.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <net/message.h>
#include <PluginUtil/PluginMessageTypes.h>

#include <config/CoviseConfig.h>

#include <string>

#include <grmsg/coGRObjVisMsg.h>
#include <grmsg/coGRObjRestrictAxisMsg.h>
#include <grmsg/coGRObjAttachedClipPlaneMsg.h>
#include <util/common.h>

using namespace osg;
using namespace vrui;
using namespace opencover;
using namespace grmsg;
using covise::coCoviseConfig;
using covise::Message;

const char *CuttingSurfaceInteraction::OPTION = "option";
const char *CuttingSurfaceInteraction::POINT = "point";
const char *CuttingSurfaceInteraction::VERTEX = "vertex";
const char *CuttingSurfaceInteraction::SCALAR = "scalar";

CuttingSurfaceInteraction::CuttingSurfaceInteraction(const RenderObject *container, coInteractor *inter, const char *pluginName, CuttingSurfacePlugin *p)
    : ModuleInteraction(container, inter, pluginName)
    , wait_(false)
    , newObject_(false)
    , planeOptionsInMenu_(false)
    , option_(0)
    , oldOption_(0)
    , csPlane_(NULL)
    , csCylX_(NULL)
    , csCylY_(NULL)
    , csCylZ_(NULL)
    , csSphere_(NULL)
    , activeClipPlane_(0)
    , clipPlaneMenu_(NULL)
    , clipPlaneIndexGroup_(NULL)
    , clipPlaneNoneCheckbox_(NULL)
    , clipPlaneOffsetSlider_(NULL)
    , clipPlaneFlipCheckbox_(NULL)
    , plugin(NULL)


{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::CuttingSurfaceInteraction\n");

    for (int i=0; i<6; ++i)
        clipPlaneIndexCheckbox_[i] = NULL;

    newObject_ = false;
    plugin = p;

    option_ = OPTION_PLANE;
    getParameters();
    oldOption_ = option_;

    // create interactors
    csPlane_ = new CuttingSurfacePlane(inter, plugin);
    csCylX_ = new CuttingSurfaceCylinder(inter);
    csCylY_ = new CuttingSurfaceCylinder(inter);
    csCylZ_ = new CuttingSurfaceCylinder(inter);
    csSphere_ = new CuttingSurfaceSphere(inter);

    // deafult is no restriction
    restrictToAxis_ = RESTRICT_NONE;

    // default is no clipplane attached
    activeClipPlane_ = -1;

    planeOptionsInMenu_ = false;

    // create menu
    createMenu();
    updateMenu();
}

CuttingSurfaceInteraction::~CuttingSurfaceInteraction()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::~CuttingSurfaceInteraction\n");

    deleteMenu();
    delete csPlane_;
    delete csCylX_;
    delete csCylY_;
    delete csCylZ_;
    delete csSphere_;

    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::~CuttingSurfaceInteraction done\n");
}

void
CuttingSurfaceInteraction::update(const RenderObject *container, coInteractor *inter)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::update\n");

    ModuleInteraction::update(container, inter);

    oldOption_ = option_;
    getParameters();

    updateMenu();

    csPlane_->update(inter);
    csCylX_->update(inter);
    csCylY_->update(inter);
    csCylZ_->update(inter);
    csSphere_->update(inter);

    newObject_ = true;

    if (option_ == OPTION_PLANE)
    {
        //       osg::BoundingSphere bsphere = cover->getObjectsScale()->getBound();
        //       clipPlaneOffsetSlider_->setMax(*2*bsphere.radius()*0.01);
        // update clip plane
        sendClipPlanePositionMsg(activeClipPlane_);
    }
}

void
CuttingSurfaceInteraction::preFrame()
{

    if (cover->debugLevel(5))
        fprintf(stderr, "CuttingSurfaceInteraction::preFrame\n");

    if (newObject_ && hideCheckbox_ != NULL)
    {
        hideCheckbox_->trigger();
        newObject_ = false;
    }

    csPlane_->preFrame(restrictToAxis_);
    csCylX_->preFrame();
    csCylY_->preFrame();
    csCylZ_->preFrame();
    csSphere_->preFrame();

    if (option_ == OPTION_PLANE && csPlane_->sendClipPlane())
    {
        // update clip plane
        sendClipPlanePositionMsg(activeClipPlane_);
    }
}

#ifdef VRUI
void
CuttingSurfaceInteraction::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == orientFree_)
    {
        if (orientFree_->state())
        {
            restrictNone();
            sendRestrictNoneMsg();
        }
    }
    else if (menuItem == orientX_)
    {
        if (orientX_->state())
        {
            restrictX();
            csPlane_->restrict(RESTRICT_X);
            wait_ = true;
            sendRestrictXMsg();
        }
    }
    else if (menuItem == orientY_)
    {
        if (orientY_->state())
        {
            restrictY();
            csPlane_->restrict(RESTRICT_Y);
            wait_ = true;
            sendRestrictYMsg();
        }
    }
    else if (menuItem == orientZ_)
    {
        if (orientZ_->state())
        {
            restrictZ();
            csPlane_->restrict(RESTRICT_Z);
            wait_ = true;
            sendRestrictZMsg();
        }
    }
    else if ((menuItem == clipPlaneIndexCheckbox_[0])
             || (menuItem == clipPlaneIndexCheckbox_[1])
             || (menuItem == clipPlaneIndexCheckbox_[2])
             || (menuItem == clipPlaneIndexCheckbox_[3])
             || (menuItem == clipPlaneIndexCheckbox_[4])
             || (menuItem == clipPlaneIndexCheckbox_[5]))
    {
        int index=0;
        sscanf(menuItem->getName(), "ClipPlane %d", &index);
        switchClipPlane(index);
        sendClipPlaneToGui();
    }
    else if (menuItem == clipPlaneNoneCheckbox_)
    {
        switchClipPlane(-1);
        sendClipPlaneToGui();
    }
    else if ((menuItem == clipPlaneOffsetSlider_) || (menuItem == clipPlaneFlipCheckbox_))
    {
        sendClipPlanePositionMsg(activeClipPlane_);
        sendClipPlaneToGui();
    }
    else if (menuItem == optionPlane_ || menuItem == optionCylX_ || menuItem == optionCylY_ || menuItem == optionCylZ_ || menuItem == optionSphere_)
    {
        int numSufaces;
        char **labels;
        int active;

        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (inter_->getChoiceParam("option", numSufaces, labels, active) == 0)
        {
            plugin->getSyncInteractors(inter_);
            if (menuItem == optionPlane_ && optionPlane_->state() && option_ != OPTION_PLANE)
            {
                plugin->setChoiceParam("option", numSufaces, labels, 0);
                switchClipPlane(activeClipPlane_);
                option_ = OPTION_PLANE;
                csPlane_->restrict(restrictToAxis_);
            }
            else if (menuItem == optionCylX_ && optionCylX_->state() && option_ != OPTION_CYLX)
            {
                plugin->setChoiceParam("option", numSufaces, labels, 2);
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_CYLX;
            }
            else if (menuItem == optionCylY_ && optionCylY_->state() && option_ != OPTION_CYLY)
            {
                plugin->setChoiceParam("option", numSufaces, labels, 3);
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_CYLY;
            }
            else if (menuItem == optionCylZ_ && optionCylZ_->state() && option_ != OPTION_CYLZ)
            {
                plugin->setChoiceParam("option", numSufaces, labels, 4);
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_CYLZ;
            }
            else if (menuItem == optionSphere_ && optionSphere_->state() && option_ != OPTION_SPHERE)
            {
                plugin->setChoiceParam("option", numSufaces, labels, 1);
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_SPHERE;
            }
            plugin->executeModule();
        }
    }
    else // other menu actions are handeled by the base class
    {
        ModuleInteraction::menuEvent(menuItem);
    }
}
#endif

void
CuttingSurfaceInteraction::restrictX()
{
    restrictToAxis_ = RESTRICT_X;
}
void
CuttingSurfaceInteraction::restrictY()
{
    restrictToAxis_ = RESTRICT_Y;
}
void
CuttingSurfaceInteraction::restrictZ()
{
    restrictToAxis_ = RESTRICT_Z;
}
void
CuttingSurfaceInteraction::restrictNone()
{
    restrictToAxis_ = RESTRICT_NONE;
}

void
CuttingSurfaceInteraction::getParameters()
{

    int numoptions;
    char **labels;
    inter_->getChoiceParam(OPTION, numoptions, labels, option_);
}

void
CuttingSurfaceInteraction::createMenu()
{
    // pick interactor checkbox

    optionChoice_ = new ui::SelectionList(menu_, "SurfaceStyle");
    optionChoice_->setText("Surface style");
    optionChoice_->append("Plane");
    optionChoice_->append("Sphere");
    optionChoice_->append("Cylinder: X");
    optionChoice_->append("Cylinder: Y");
    optionChoice_->append("Cylinder: Z");
    optionChoice_->select(0);

#if 0
    optionMenu_ = new ui::Menu(menu_, "SurfaceStyle");
    optionMenu_->setText("Surface style: ---");
    optionGroup_ = new ui::ButtonGroup("OptionGroup", this);
    optionPlane_ = new ui::Button(optionGroup_, "Plane", 0);
    optionPlane_->setState(true);
    optionCylX_ = new ui::Button(optionGroup_, "CylinderX", 2);
    optionCylX_->setState(false);
    optionCylY_ = new ui::Button(optionGroup_, "CylinderY", 3);
    optionCylY_->setState(false);
    optionCylZ_ = new ui::Button(optionGroup_, "CylinderZ", 4);
    optionCylZ_->setState(false);
    optionSphere_ = new ui::Button(optionGroup_, "Sphere", 1);
    optionSphere_->setState(false);
#endif
    optionChoice_->setCallback([this](int idx){
        int numSufaces;
        char **labels;
        int active;

        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (inter_->getChoiceParam("option", numSufaces, labels, active) == 0)
        {
            plugin->getSyncInteractors(inter_);
            plugin->setChoiceParam("option", numSufaces, labels, idx);

            if (idx==0 && option_ != OPTION_PLANE)
            {
                switchClipPlane(activeClipPlane_);
                option_ = OPTION_PLANE;
                csPlane_->restrict(restrictToAxis_);
            }
            else if (idx==2 && option_ != OPTION_CYLX)
            {
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_CYLX;
            }
            else if (idx==3 && option_ != OPTION_CYLY)
            {
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_CYLY;
            }
            else if (idx==4 && option_ != OPTION_CYLZ)
            {
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_CYLZ;
            }
            else if (idx==1 && option_ != OPTION_SPHERE)
            {
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_SPHERE;
            }
            plugin->executeModule();
        }
    });

    orient_ = new ui::SelectionList("Orientation", this);
    orient_->append("Free");
    orient_->append("X Axis");
    orient_->append("Y Axis");
    orient_->append("Z Axis");
    orient_->select(0);
    orient_->setCallback([this](int idx){
        restrictToAxis_ = idx;
        if (idx > 0)
        {
            csPlane_->restrict(idx);
            wait_ = true;
        }

        switch (idx)
        {
        case 0:
            sendRestrictNoneMsg();
            break;
        case 1:
            sendRestrictXMsg();
            break;
        case 2:
            sendRestrictYMsg();
            break;
        case 3:
            sendRestrictZMsg();
            break;
        }
    });
#if 0
    orientGroup_ = new ui::ButtonGroup("OrientationGroup", this);

    orientFree_ = new ui::Button(orientGroup_, "Free");
    orientFree_->setState(true);
    orientFree_->setCallback([this](bool state){
        if (state)
        {
            restrictNone();
            sendRestrictNoneMsg();
        }
    });
    orientX_ = new ui::Button(orientGroup_, "X Axis");
    orientX_->setState(false);
    orientX_->setCallback([this](bool state){
        if (state)
        {
            restrictX();
            csPlane_->restrict(RESTRICT_X);
            wait_ = true;
            sendRestrictXMsg();
        }
    });
    orientY_ = new ui::Button(orientGroup_, "Y Axis");
    orientY_->setState(false);
    orientY_->setCallback([this](bool state){
        if (state)
        {
            restrictY();
            csPlane_->restrict(RESTRICT_Y);
            wait_ = true;
            sendRestrictYMsg();
        }
    });
    orientZ_ = new ui::Button(orientGroup_, "Z Axis");
    orientZ_->setState(false);
    orientZ_->setCallback([this](bool state){
        if (state)
        {
            restrictZ();
            csPlane_->restrict(RESTRICT_Z);
            wait_ = true;
            sendRestrictZMsg();
        }
    });
#endif

    char title[100];
    sprintf(title, "%s_ClipPlane", menu_->name().c_str());
    clipPlaneMenu_ = new ui::Menu(title, this);
    clipPlaneMenu_->setText("Attached clip plane");

    clipPlaneIndexGroup_ = new ui::ButtonGroup("ClipPlaneIndex", this);

    clipPlaneNoneCheckbox_ = new ui::Button(clipPlaneMenu_, "NoClipPlane", clipPlaneIndexGroup_, -1);
    clipPlaneNoneCheckbox_->setText("No clip plane");
    clipPlaneNoneCheckbox_->setState(true);
    for (int i = 0; i < 3; i++)
    {
        char name[100];
        sprintf(name, "ClipPlane%d", i);
        clipPlaneIndexCheckbox_[i] = new ui::Button(clipPlaneMenu_, name, clipPlaneIndexGroup_, i);
        sprintf(name, "Clip plane %d", i);
        clipPlaneIndexCheckbox_[i]->setText(name);
        clipPlaneIndexCheckbox_[i]->setState(false);
    }
    clipPlaneIndexGroup_->setCallback([this](int idx){
        switchClipPlane(idx);
        sendClipPlaneToGui();
    });

    clipPlaneOffsetSlider_ = new ui::Slider(clipPlaneMenu_, "Offset");
    clipPlaneOffsetSlider_->setBounds(0., 1.);
    clipPlaneOffsetSlider_->setValue(0.005);
    clipPlaneOffsetSlider_->setCallback([this](double value, bool released){
        sendClipPlanePositionMsg(activeClipPlane_);
        sendClipPlaneToGui();
    });

    clipPlaneFlipCheckbox_ = new ui::Button(clipPlaneMenu_, "Flip");
    clipPlaneFlipCheckbox_->setState(false);
    clipPlaneFlipCheckbox_->setCallback([this](bool state){
        sendClipPlanePositionMsg(activeClipPlane_);
        sendClipPlaneToGui();
    });

    if (option_ == OPTION_PLANE)
    {
        planeOptionsInMenu_ = true;
        menu_->add(orient_);
        menu_->add(clipPlaneMenu_);
    }
}

void
CuttingSurfaceInteraction::updateMenu()
{
    optionChoice_->select(option_);

    bool addOpt=false, removeOpt=false;
    switch (option_)
    {
    case OPTION_PLANE:
        //optionButton_->setName("SurfaceStyle: Plane");
        if (!planeOptionsInMenu_)
            addOpt=true;
        break;
    case OPTION_CYLX:
        //optionButton_->setName("SurfaceStyle: CylinderX");
        if (planeOptionsInMenu_)
            removeOpt=true;
        break;
    case OPTION_CYLY:
        //optionButton_->setName("SurfaceStyle: CylinderY");
        if (planeOptionsInMenu_)
            removeOpt=true;
        break;
    case OPTION_CYLZ:
        //optionButton_->setName("SurfaceStyle: CylinderZ");
        if (planeOptionsInMenu_)
            removeOpt=true;
        break;
    case OPTION_SPHERE:
        //optionButton_->setName("SurfaceStyle: Sphere");
        if (planeOptionsInMenu_)
            removeOpt=true;
        break;
    }

    updatePickInteractors(showPickInteractor_);
    updateDirectInteractors(showDirectInteractor_);

    if (addOpt)
    {
        planeOptionsInMenu_ = true;
        menu_->add(orient_);
        menu_->add(clipPlaneMenu_);
    }
    else if (removeOpt)
    {
        planeOptionsInMenu_ = false;
        menu_->remove(orient_);
        menu_->remove(clipPlaneMenu_);
    }
}

void
CuttingSurfaceInteraction::deleteMenu()
{
#ifdef VRUI
    delete optionMenu_;
    delete optionPlane_;
    delete optionCylX_;
    delete optionCylY_;
    delete optionCylZ_;
    delete optionSphere_;
    delete optionGroup_;

    switchClipPlane(-1);
    delete clipPlaneNoneCheckbox_;
    for (int i = 0; i < 3; i++)
        delete clipPlaneIndexCheckbox_[i];
    delete clipPlaneOffsetSlider_;
    delete clipPlaneFlipCheckbox_;
    delete clipPlaneMenu_;
#endif
}

void CuttingSurfaceInteraction::updatePickInteractorVisibility()
{
    //fprintf(stderr,"updatePickInteractorVisibility\n");
    // if geometry is hidden, hide also interactor
    updatePickInteractors(!hideCheckbox_->state() && showPickInteractorCheckbox_->state());
}
/*
void 
CuttingSurfaceInteraction::setShowInteractorFromGui(bool state)
{  
   //fprintf(stderr,"\n\nCuttingSurfaceInteraction::setShowInteractorCheckbox %d\n", state);

   if (state)
   {
      if( !showPickInteractorCheckbox_->state() )
      {
         showPickInteractorCheckbox_->setState(true);
         showPickInteractor_=true;
         switch (option_)
         {
            case OPTION_PLANE:
               if (showPickInteractorCheckbox_->state())
                  csPlane_->showPickInteractor();
               break;
            case OPTION_CYLX:
               if (showPickInteractorCheckbox_->state())
                  csCylX_->showPickInteractor();
               break;
            case OPTION_CYLY:
               if (showPickInteractorCheckbox_->state())
                  csCylZ_->showPickInteractor();
               break;
            case OPTION_CYLZ:
               if (showPickInteractorCheckbox_->state())
                  csCylY_->showPickInteractor();
               break;
            case OPTION_SPHERE:
               if (showPickInteractorCheckbox_->state())
                  csSphere_->showPickInteractor();
               break;
         }
         //enable intersection??
      }
   }
   else
   {
      if( showPickInteractorCheckbox_->state() )
      {
         showPickInteractorCheckbox_->setState(false);
         showPickInteractor_=false;
         switch (option_)
         {
            case OPTION_PLANE:
               csPlane_->hidePickInteractor();
               break;
            case OPTION_CYLX:
               csCylX_->hidePickInteractor();
               break;
            case OPTION_CYLY:
               csCylY_->hidePickInteractor();
               break;
            case OPTION_CYLZ:
               csCylZ_->hidePickInteractor();
               break;
            case OPTION_SPHERE:
               csSphere_->hidePickInteractor();
               break;
         }
         //disable intersection??
      }
   }
}*/

void
CuttingSurfaceInteraction::interactorSetCaseFromGui(const char *caseName)
{
    //fprintf(stderr,"CuttingSurfaceInteraction::interactorSetCaseFromGui case=%s \n", caseName);

    string interactorCaseName(caseName);
    interactorCaseName += "_INTERACTOR";

    osg::MatrixTransform *interDCS = VRSceneGraph::instance()->findFirstNode<osg::MatrixTransform>(interactorCaseName.c_str(), false, cover->getObjectsScale());
    if (!interDCS)

    {
        // firsttime we create also a case DCS
        //fprintf(stderr,"ModuleFeedbackManager::setCaseFromGui create interactor case DCS named %s\n", interactorCaseName.c_str());
        interDCS = new osg::MatrixTransform();
        interDCS->setName(interactorCaseName.c_str());
        cover->getObjectsScale()->addChild(interDCS);
    }

    if (interDCS)
    {
        csPlane_->setCaseTransform(interDCS);
    }
    else
        fprintf(stderr, "CuttingsurfaceInteraction::interactorSetCaseFromGui didn't find case dcs\n");
}

void
CuttingSurfaceInteraction::setInteractorPointFromGui(float x, float y, float z)
{
    //fprintf(stderr,"\n\nCuttingSurfaceInteraction::setInteractorPoint %f %f %f\n", x, y, z);
    Vec3 point(x, y, z);
    // gui only supports plane
    switch (option_)
    {
    case OPTION_PLANE:
        csPlane_->setInteractorPoint(point);
        break;
    }
}

void
CuttingSurfaceInteraction::setInteractorNormalFromGui(float x, float y, float z)
{
    //fprintf(stderr,"\n\nCuttingSurfaceInteraction::setInteractorNormal %f %f %f\n", x, y, z);

    Vec3 normal(x, y, z);
    normal.normalize();
    // gui only supports plane
    switch (option_)
    {
    case OPTION_PLANE:
        csPlane_->setInteractorNormal(normal);
        break;
    }
}

void
CuttingSurfaceInteraction::setRestrictXFromGui()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::setRestrictXFromGui\n");
    orient_->select(1);
    restrictX();
}

void
CuttingSurfaceInteraction::setRestrictYFromGui()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::setRestrictYFromGui\n");
    orient_->select(2);
    restrictY();
}
void
CuttingSurfaceInteraction::setRestrictZFromGui()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::setRestrictZFromGui\n");
    orient_->select(3);
    restrictZ();
}

void
CuttingSurfaceInteraction::setRestrictNoneFromGui()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::setRestrictNoneFromGui\n");
    orient_->select(0);
    restrictNone();
}

void
CuttingSurfaceInteraction::setClipPlaneFromGui(int index, float offset, bool flip)
{
    // disable index-checkBoxes
    for (int i = 0; i < 3; i++)
    {
        clipPlaneIndexCheckbox_[i]->setState(false);
    }
    // set according to message
    if ((index < 0) || (index > 2))
    {
        clipPlaneNoneCheckbox_->setState(true);
    }
    else
    {
        clipPlaneNoneCheckbox_->setState(false);
        clipPlaneIndexCheckbox_[index]->setState(true);
        clipPlaneOffsetSlider_->setValue(offset);
        clipPlaneFlipCheckbox_->setState(flip);
    }
    switchClipPlane(index);
}

// Switch to a specific ClipPlane. This includes deactivating a selected previous one.
void
CuttingSurfaceInteraction::switchClipPlane(int index)
{
    if (activeClipPlane_ > -1)
    {
        sendClipPlaneVisibilityMsg(activeClipPlane_, false);
    }
    activeClipPlane_ = index;
    if (activeClipPlane_ > -1)
    {
        sendClipPlanePositionMsg(activeClipPlane_);
        sendClipPlaneVisibilityMsg(activeClipPlane_, !(hideCheckbox_->state()));
    }
}

// Change visibility of ClipPlane "index".
void
CuttingSurfaceInteraction::sendClipPlaneVisibilityMsg(int index, bool enabled)
{
    //fprintf(stderr, "CuttingSurfaceInteraction::sendClipPlaneVisibilityMsg %d %d\n", index, enabled);
    if (index >= 0)
    {
        char msg[100];
        sprintf(msg, "%s %d", (enabled ? "enable" : "disable"), index);
        cover->sendMessage(NULL, "ClipPlane", PluginMessageTypes::ClipPlaneMessage, strlen(msg), msg);
    }
}

// Send position to ClipPlane "index". Uses offset and flip to calculate position.
void
CuttingSurfaceInteraction::sendClipPlanePositionMsg(int index)
{
    //fprintf(stderr,"CuttingSurfaceInteraction::sendClipPlanePositionMsg %d\n", index);
    if (index >= 0)
    {

        Vec3 point = csPlane_->getInteractorPoint();
        Vec3 normal = csPlane_->getInteractorNormal();
        if (clipPlaneFlipCheckbox_->state())
        {
            normal = normal * -1.0;
        }

        float distance = -point * normal;
        distance += clipPlaneOffsetSlider_->value();

        char msg[255];
        sprintf(msg, "set %d %f %f %f %f", index, normal[0], normal[1], normal[2], distance);
        cover->sendMessage(NULL, "ClipPlane", PluginMessageTypes::ClipPlaneMessage, strlen(msg), msg);
    }
}

void
CuttingSurfaceInteraction::sendClipPlaneToGui()
{
    if (coVRMSController::instance()->isMaster())
    {
        coGRObjAttachedClipPlaneMsg msg(coGRMsg::ATTACHED_CLIPPLANE, initialObjectName_.c_str(), activeClipPlane_, clipPlaneOffsetSlider_->value(), clipPlaneFlipCheckbox_->state());
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(msg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

void
CuttingSurfaceInteraction::sendShowPickInteractorMsg()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCuttingSurfaceInteraction::sendShowPickInteractorMsg\n");

    if (coVRMSController::instance()->isMaster())
    {
        coGRObjVisMsg visMsg(coGRMsg::INTERACTOR_VISIBLE, initialObjectName_.c_str(), 1);
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(visMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

void
CuttingSurfaceInteraction::sendHidePickInteractorMsg()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCuttingSurfaceInteraction::sendHidePickInteractorMsg\n");

    if (coVRMSController::instance()->isMaster())
    {
        coGRObjVisMsg visMsg(coGRMsg::INTERACTOR_VISIBLE, initialObjectName_.c_str(), 0);
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(visMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

void
CuttingSurfaceInteraction::sendRestrictXMsg()
{
    if (coVRMSController::instance()->isMaster())
    {
        coGRObjRestrictAxisMsg restrictXMsg(coGRMsg::RESTRICT_AXIS, initialObjectName_.c_str(), "xAxis");
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(restrictXMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

void
CuttingSurfaceInteraction::sendRestrictYMsg()
{
    if (coVRMSController::instance()->isMaster())
    {
        coGRObjRestrictAxisMsg restrictYMsg(coGRMsg::RESTRICT_AXIS, initialObjectName_.c_str(), "yAxis");
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(restrictYMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

void
CuttingSurfaceInteraction::sendRestrictZMsg()
{
    if (coVRMSController::instance()->isMaster())
    {
        coGRObjRestrictAxisMsg restrictZMsg(coGRMsg::RESTRICT_AXIS, initialObjectName_.c_str(), "zAxis");
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(restrictZMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

void
CuttingSurfaceInteraction::sendRestrictNoneMsg()
{
    if (coVRMSController::instance()->isMaster())
    {
        coGRObjRestrictAxisMsg restrictNoneMsg(coGRMsg::RESTRICT_AXIS, initialObjectName_.c_str(), "freeAxis");
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(restrictNoneMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

void CuttingSurfaceInteraction::updatePickInteractors(bool show)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::updatePickInteractors %d\n", show);
    if (show && !hideCheckbox_->state())
    {

        if (option_ == OPTION_PLANE)
        {
            csPlane_->showPickInteractor();
            csCylX_->hidePickInteractor();
            csCylY_->hidePickInteractor();
            csCylZ_->hidePickInteractor();
            csSphere_->hidePickInteractor();
        }
        else if (option_ == OPTION_CYLX)
        {
            csPlane_->hidePickInteractor();
            csCylX_->showPickInteractor();
            csCylY_->hidePickInteractor();
            csCylZ_->hidePickInteractor();
            csSphere_->hidePickInteractor();
        }
        else if (option_ == OPTION_CYLY)
        {
            csPlane_->hidePickInteractor();
            csCylX_->hidePickInteractor();
            csCylY_->showPickInteractor();
            csCylZ_->hidePickInteractor();
            csSphere_->hidePickInteractor();
        }
        else if (option_ == OPTION_CYLZ)
        {
            csPlane_->hidePickInteractor();
            csCylX_->hidePickInteractor();
            csCylY_->hidePickInteractor();
            csCylZ_->showPickInteractor();
            csSphere_->hidePickInteractor();
        }

        else if (option_ == OPTION_SPHERE)
        {
            csPlane_->hidePickInteractor();
            csCylX_->hidePickInteractor();
            csCylY_->hidePickInteractor();
            csCylZ_->hidePickInteractor();
            csSphere_->showPickInteractor();
        }
        //sendShowPickInteractorMsg();
    }
    else
    {
        csPlane_->hidePickInteractor();
        csCylX_->hidePickInteractor();
        csCylY_->hidePickInteractor();
        csCylZ_->hidePickInteractor();
        csSphere_->hidePickInteractor();
        //sendHidePickInteractorMsg();
    }
}

void CuttingSurfaceInteraction::updateDirectInteractors(bool show)
{
    if (show)
    {
        if (option_ == OPTION_PLANE)
        {
            csPlane_->showDirectInteractor();
            csCylX_->hideDirectInteractor();
            csCylY_->hideDirectInteractor();
            csCylZ_->hideDirectInteractor();
            csSphere_->hideDirectInteractor();
        }
        else if (option_ == OPTION_CYLX)
        {
            csPlane_->hideDirectInteractor();
            csCylX_->showDirectInteractor();
            csCylY_->hideDirectInteractor();
            csCylZ_->hideDirectInteractor();
            csSphere_->hideDirectInteractor();
        }
        else if (option_ == OPTION_CYLY)
        {
            csPlane_->hideDirectInteractor();
            csCylX_->hideDirectInteractor();
            csCylY_->showDirectInteractor();
            csCylZ_->hideDirectInteractor();
            csSphere_->hideDirectInteractor();
        }
        else if (option_ == OPTION_CYLZ)
        {
            csPlane_->hideDirectInteractor();
            csCylX_->hideDirectInteractor();
            csCylY_->hideDirectInteractor();
            csCylZ_->showDirectInteractor();
            csSphere_->hideDirectInteractor();
        }
        else if (option_ == OPTION_SPHERE)
        {
            csPlane_->hideDirectInteractor();
            csCylX_->hideDirectInteractor();
            csCylY_->hideDirectInteractor();
            csCylZ_->hideDirectInteractor();
            csSphere_->showDirectInteractor();
        }
    }
    else
    {
        csPlane_->hideDirectInteractor();
        csCylX_->hideDirectInteractor();
        csCylY_->hideDirectInteractor();
        csCylZ_->hideDirectInteractor();
        csSphere_->hideDirectInteractor();
    }
}
