/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CuttingSurfaceInteraction.h"
#include "CuttingSurfacePlane.h"
#include "CuttingSurfaceCylinder.h"
#include "CuttingSurfaceSphere.h"
#include "CuttingSurfacePlugin.h"

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

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

CuttingSurfaceInteraction::CuttingSurfaceInteraction(RenderObject *container, coInteractor *inter, const char *pluginName, CuttingSurfacePlugin *p)
    : ModuleInteraction(container, inter, pluginName)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::CuttingSurfaceInteraction\n");

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
CuttingSurfaceInteraction::update(RenderObject *container, coInteractor *inter)
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
        menuEvent(hideCheckbox_);
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

void
CuttingSurfaceInteraction::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == orientFree_)
    {
        if (orientFree_->getState())
        {
            restrictNone();
            sendRestrictNoneMsg();
        }
    }
    else if (menuItem == orientX_)
    {
        if (orientX_->getState())
        {
            restrictX();
            csPlane_->restrict(RESTRICT_X);
            wait_ = true;
            sendRestrictXMsg();
        }
    }
    else if (menuItem == orientY_)
    {
        if (orientY_->getState())
        {
            restrictY();
            csPlane_->restrict(RESTRICT_Y);
            wait_ = true;
            sendRestrictYMsg();
        }
    }
    else if (menuItem == orientZ_)
    {
        if (orientZ_->getState())
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
            if (menuItem == optionPlane_ && optionPlane_->getState() && option_ != OPTION_PLANE)
            {
                plugin->setChoiceParam("option", numSufaces, labels, 0);
                switchClipPlane(activeClipPlane_);
                option_ = OPTION_PLANE;
                csPlane_->restrict(restrictToAxis_);
            }
            else if (menuItem == optionCylX_ && optionCylX_->getState() && option_ != OPTION_CYLX)
            {
                plugin->setChoiceParam("option", numSufaces, labels, 2);
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_CYLX;
            }
            else if (menuItem == optionCylY_ && optionCylY_->getState() && option_ != OPTION_CYLY)
            {
                plugin->setChoiceParam("option", numSufaces, labels, 3);
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_CYLY;
            }
            else if (menuItem == optionCylZ_ && optionCylZ_->getState() && option_ != OPTION_CYLZ)
            {
                plugin->setChoiceParam("option", numSufaces, labels, 4);
                sendClipPlaneVisibilityMsg(activeClipPlane_, false);
                option_ = OPTION_CYLZ;
            }
            else if (menuItem == optionSphere_ && optionSphere_->getState() && option_ != OPTION_SPHERE)
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

    optionButton_ = new coSubMenuItem("SurfaceStyle:---");
    optionButton_->setMenuListener(this);
    menu_->add(optionButton_);

    optionMenu_ = new coRowMenu("SurfaceStyle", menu_);
    optionGroup_ = new coCheckboxGroup();

    optionPlane_ = new coCheckboxMenuItem("Plane", true, optionGroup_);
    optionPlane_->setMenuListener(this);
    optionMenu_->add(optionPlane_);

    optionCylX_ = new coCheckboxMenuItem("CylinderX", false, optionGroup_);
    optionCylX_->setMenuListener(this);
    optionMenu_->add(optionCylX_);

    optionCylY_ = new coCheckboxMenuItem("CylinderY", false, optionGroup_);
    optionCylY_->setMenuListener(this);
    optionMenu_->add(optionCylY_);

    optionCylZ_ = new coCheckboxMenuItem("CylinderZ", false, optionGroup_);
    optionCylZ_->setMenuListener(this);
    optionMenu_->add(optionCylZ_);

    optionSphere_ = new coCheckboxMenuItem("Sphere", false, optionGroup_);
    optionSphere_->setMenuListener(this);
    optionMenu_->add(optionSphere_);

    optionButton_->setMenu(optionMenu_);

    orientGroup_ = new coCheckboxGroup();

    orientFree_ = new coCheckboxMenuItem("Free", true, orientGroup_);
    orientFree_->setMenuListener(this);

    orientX_ = new coCheckboxMenuItem("X Axis", false, orientGroup_);
    orientX_->setMenuListener(this);

    orientY_ = new coCheckboxMenuItem("Y Axis", false, orientGroup_);
    orientY_->setMenuListener(this);

    orientZ_ = new coCheckboxMenuItem("Z Axis", false, orientGroup_);
    orientZ_->setMenuListener(this);

    char title[100];
    sprintf(title, "%s:ClipPlane", menu_->getName());
    clipPlaneMenuItem_ = new coSubMenuItem("Attached ClipPlane...");
    clipPlaneMenu_ = new coRowMenu(title, menu_);
    clipPlaneMenuItem_->setMenu(clipPlaneMenu_);

    clipPlaneIndexGroup_ = new coCheckboxGroup();

    clipPlaneNoneCheckbox_ = new coCheckboxMenuItem("No ClipPlane", true, clipPlaneIndexGroup_);
    clipPlaneNoneCheckbox_->setMenuListener(this);
    clipPlaneMenu_->add(clipPlaneNoneCheckbox_);
    for (int i = 0; i < 3; i++)
    {
        char name[100];
        sprintf(name, "ClipPlane %d", i);
        clipPlaneIndexCheckbox_[i] = new coCheckboxMenuItem(name, false, clipPlaneIndexGroup_);
        clipPlaneIndexCheckbox_[i]->setMenuListener(this);
        clipPlaneMenu_->add(clipPlaneIndexCheckbox_[i]);
    }

    clipPlaneOffsetSlider_ = new coSliderMenuItem("Offset", 0.0, 1.0, 0.005);
    clipPlaneOffsetSlider_->setMenuListener(this);
    clipPlaneMenu_->add(clipPlaneOffsetSlider_);

    clipPlaneFlipCheckbox_ = new coCheckboxMenuItem("Flip", false);
    clipPlaneFlipCheckbox_->setMenuListener(this);
    clipPlaneMenu_->add(clipPlaneFlipCheckbox_);

    if (option_ == OPTION_PLANE)
    {
        planeOptionsInMenu_ = true;
        menu_->add(orientFree_);
        menu_->add(orientX_);
        menu_->add(orientY_);
        menu_->add(orientZ_);
        menu_->add(clipPlaneMenuItem_);
    }
}

void
CuttingSurfaceInteraction::updateMenu()
{

    switch (option_)
    {
    case OPTION_PLANE:
        optionPlane_->setState(true);
        optionCylX_->setState(false);
        optionCylY_->setState(false);
        optionCylZ_->setState(false);
        optionSphere_->setState(false);
        optionButton_->setName("SurfaceStyle: Plane");

        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (!planeOptionsInMenu_)
        {
            planeOptionsInMenu_ = true;
            menu_->add(orientFree_);
            menu_->add(orientX_);
            menu_->add(orientY_);
            menu_->add(orientZ_);
            menu_->add(clipPlaneMenuItem_);
        }
        break;
    case OPTION_CYLX:
        optionPlane_->setState(false);
        optionCylX_->setState(true);
        optionCylY_->setState(false);
        optionCylZ_->setState(false);
        optionSphere_->setState(false);
        optionButton_->setName("SurfaceStyle: CylinderX");

        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (planeOptionsInMenu_)
        {
            planeOptionsInMenu_ = false;
            menu_->remove(orientFree_);
            menu_->remove(orientX_);
            menu_->remove(orientY_);
            menu_->remove(orientZ_);
            menu_->remove(clipPlaneMenuItem_);
        }
        break;
    case OPTION_CYLY:
        optionPlane_->setState(false);
        optionCylX_->setState(false);
        optionCylY_->setState(true);
        optionCylZ_->setState(false);
        optionSphere_->setState(false);
        optionButton_->setName("SurfaceStyle: CylinderY");

        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (planeOptionsInMenu_)
        {
            planeOptionsInMenu_ = false;
            menu_->remove(orientFree_);
            menu_->remove(orientX_);
            menu_->remove(orientY_);
            menu_->remove(orientZ_);
            menu_->remove(clipPlaneMenuItem_);
        }
        break;
    case OPTION_CYLZ:
        optionPlane_->setState(false);
        optionCylX_->setState(false);
        optionCylY_->setState(false);
        optionCylZ_->setState(true);
        optionSphere_->setState(false);
        optionButton_->setName("SurfaceStyle: CylinderZ");

        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (planeOptionsInMenu_)
        {
            planeOptionsInMenu_ = false;
            menu_->remove(orientFree_);
            menu_->remove(orientX_);
            menu_->remove(orientY_);
            menu_->remove(orientZ_);
            menu_->remove(clipPlaneMenuItem_);
        }
        break;
    case OPTION_SPHERE:
        optionPlane_->setState(false);
        optionCylX_->setState(false);
        optionCylY_->setState(false);
        optionCylZ_->setState(false);
        optionSphere_->setState(true);
        optionButton_->setName("SurfaceStyle: Sphere");

        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (planeOptionsInMenu_)
        {
            planeOptionsInMenu_ = false;
            menu_->remove(orientFree_);
            menu_->remove(orientX_);
            menu_->remove(orientY_);
            menu_->remove(orientZ_);
            menu_->remove(clipPlaneMenuItem_);
        }
        break;
    }
}

void
CuttingSurfaceInteraction::deleteMenu()
{

    delete optionButton_;
    delete optionMenu_;
    delete optionPlane_;
    delete optionCylX_;
    delete optionCylY_;
    delete optionCylZ_;
    delete optionSphere_;
    delete optionGroup_;

    delete orientFree_;
    delete orientX_;
    delete orientY_;
    delete orientZ_;
    delete orientGroup_;

    switchClipPlane(-1);
    delete clipPlaneNoneCheckbox_;
    for (int i = 0; i < 3; i++)
        delete clipPlaneIndexCheckbox_[i];
    delete clipPlaneOffsetSlider_;
    delete clipPlaneFlipCheckbox_;
    delete clipPlaneMenuItem_;
    delete clipPlaneMenu_;
}

void CuttingSurfaceInteraction::updatePickInteractorVisibility()
{
    //fprintf(stderr,"updatePickInteractorVisibility\n");
    // if geometry is hidden, hide also interactor
    updatePickInteractors(!hideCheckbox_->getState() && showPickInteractorCheckbox_->getState());
}
/*
void 
CuttingSurfaceInteraction::setShowInteractorFromGui(bool state)
{  
   //fprintf(stderr,"\n\nCuttingSurfaceInteraction::setShowInteractorCheckbox %d\n", state);

   if (state)
   {
      if( !showPickInteractorCheckbox_->getState() )
      {
         showPickInteractorCheckbox_->setState(true);
         showPickInteractor_=true;
         switch (option_)
         {
            case OPTION_PLANE:
               if (showPickInteractorCheckbox_->getState())
                  csPlane_->showPickInteractor();
               break;
            case OPTION_CYLX:
               if (showPickInteractorCheckbox_->getState())
                  csCylX_->showPickInteractor();
               break;
            case OPTION_CYLY:
               if (showPickInteractorCheckbox_->getState())
                  csCylZ_->showPickInteractor();
               break;
            case OPTION_CYLZ:
               if (showPickInteractorCheckbox_->getState())
                  csCylY_->showPickInteractor();
               break;
            case OPTION_SPHERE:
               if (showPickInteractorCheckbox_->getState())
                  csSphere_->showPickInteractor();
               break;
         }
         //enable intersection??
      }
   }
   else
   {
      if( showPickInteractorCheckbox_->getState() )
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
    orientX_->setState(true);
    orientY_->setState(false);
    orientZ_->setState(false);
    orientFree_->setState(false);
    restrictX();
}

void
CuttingSurfaceInteraction::setRestrictYFromGui()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::setRestrictYFromGui\n");
    orientX_->setState(false);
    orientY_->setState(true);
    orientZ_->setState(false);
    orientFree_->setState(false);
    restrictY();
}
void
CuttingSurfaceInteraction::setRestrictZFromGui()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::setRestrictZFromGui\n");
    orientX_->setState(false);
    orientY_->setState(false);
    orientZ_->setState(true);
    orientFree_->setState(false);
    restrictZ();
}

void
CuttingSurfaceInteraction::setRestrictNoneFromGui()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceInteraction::setRestrictNoneFromGui\n");
    orientX_->setState(false);
    orientY_->setState(false);
    orientZ_->setState(false);
    orientFree_->setState(true);
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
        sendClipPlaneVisibilityMsg(activeClipPlane_, !(hideCheckbox_->getState()));
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
        if (clipPlaneFlipCheckbox_->getState())
        {
            normal = normal * -1.0;
        }

        float distance = -point * normal;
        distance += clipPlaneOffsetSlider_->getValue();

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
        coGRObjAttachedClipPlaneMsg msg(coGRMsg::ATTACHED_CLIPPLANE, initialObjectName_.c_str(), activeClipPlane_, clipPlaneOffsetSlider_->getValue(), clipPlaneFlipCheckbox_->getState());
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
    if (show && !hideCheckbox_->getState())
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
