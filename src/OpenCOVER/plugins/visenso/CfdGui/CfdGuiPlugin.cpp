/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "CfdGuiPlugin.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRNavigationManager.h>
#include <cover/VRSceneGraph.h>
#include <config/CoviseConfig.h>
#include <cover/OpenCOVER.h>
#include <cover/ui/Button.h>
#include <cover/ui/Manager.h>
#include <net/message_types.h>
#include <net/message.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/sginterface/vruiButtons.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <grmsg/coGRObjTransformCaseMsg.h>
#include <grmsg/coGRGraphicRessourceMsg.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <cover/coVRPluginList.h>
#include <osgDB/WriteFile>
#include <OpenVRUI/sginterface/vruiButtons.h>

using namespace vrui;
using namespace opencover;
using namespace grmsg;
using namespace covise;

CfdGuiPlugin::CfdGuiPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

CfdGuiPlugin::~CfdGuiPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCfdGuiPlugin::~CfdGuiPlugin\n");
}

bool CfdGuiPlugin::init()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCfdGuiPlugin::CfdGuiPlugin\n");

    hideMenu = coCoviseConfig::isOn("COVER.Plugin.CfdGui.HideMenuOnNewPresStep", true);

    coverMenuButton_ = new coSubMenuItem("Presentation...");
    cover->getMenu()->add(coverMenuButton_);
    presentationMenu_ = new coRowMenu("Presentation", cover->getMenu());
    coverMenuButton_->setMenu(presentationMenu_);

    toStartButton_ = new coButtonMenuItem("go to start");
    toStartButton_->setMenuListener(this);
    presentationMenu_->add(toStartButton_);

    backwardButton_ = new coButtonMenuItem("step backward");
    backwardButton_->setMenuListener(this);
    presentationMenu_->add(backwardButton_);

    stopButton_ = new coButtonMenuItem("stop");
    stopButton_->setMenuListener(this);
    presentationMenu_->add(stopButton_);

    playButton_ = new coButtonMenuItem("play");
    playButton_->setMenuListener(this);
    presentationMenu_->add(playButton_);

    forwardButton_ = new coButtonMenuItem("step forward");
    forwardButton_->setMenuListener(this);
    presentationMenu_->add(forwardButton_);

    reloadButton_ = new coButtonMenuItem("reload step");
    reloadButton_->setMenuListener(this);
    presentationMenu_->add(reloadButton_);

    toEndButton_ = new coButtonMenuItem("go to end");
    toEndButton_->setMenuListener(this);
    presentationMenu_->add(toEndButton_);

    return true;
}

void CfdGuiPlugin::preFrame()
{
    const unsigned released = cover->getPointerButton()->wasReleased();
    if (released & vruiButtons::FORWARD_BUTTON)
        sendPresentationForwardMsgToGui();
    else if (released & vruiButtons::BACKWARD_BUTTON)
        sendPresentationBackwardMsgToGui();
}

coMenuItem *CfdGuiPlugin::getMenuButton(const std::string &buttonName)
{
    if (buttonName == "presentationForward")
        return forwardButton_;
    else if (buttonName == "presentationBackward")
        return backwardButton_;
    else if (buttonName == "presentationReload")
        return reloadButton_;
    else if (buttonName == "presentationToStart")
        return toStartButton_;
    else if (buttonName == "presentationToEnd")
        return toEndButton_;
    else if (buttonName == "presentationStart")
        return playButton_;
    else //if (buttonName == "presentationStop")
        return stopButton_;
}

void CfdGuiPlugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin CfdGui coVRGuiToRenderMsg\n");

    if (msg.isValid())
    {
        if (msg.getType() == coGRMsg::TRANSFORM_CASE)
        {

            auto &transformCaseMsg = msg.as<coGRObjTransformCaseMsg>();
            const char *caseName = transformCaseMsg.getObjName();
            if (cover->debugLevel(5))
                fprintf(stderr, "\tcoGRMsg::TRANSFORM_CASE case=%s\n", caseName);

            float row0[4];
            float row1[4];
            float row2[4];
            float row3[4];
            for (int i = 0; i < 4; i++)
            {
                row0[i] = transformCaseMsg.getMatrix(0, i);
                row1[i] = transformCaseMsg.getMatrix(1, i);
                row2[i] = transformCaseMsg.getMatrix(2, i);
                row3[i] = transformCaseMsg.getMatrix(3, i);
            }

            handleTransformCaseMsg(caseName, row0, row1, row2, row3);
        }
        else if (msg.getType() == coGRMsg::KEYWORD)
        {

            auto &keyWordMsg = msg.as<coGRKeyWordMsg>();
            const char *keyword = keyWordMsg.getKeyWord();
            if (cover->debugLevel(3))
                fprintf(stderr, "\tcoGRMsg::KEYWORD keyword=%s\n", keyword);
            if (strcmp(keyword, "viewAll") == 0)
                VRSceneGraph::instance()->viewAll();
            else if (strcmp(keyword, "orthographicProjection") == 0)
            {
                auto elem = cover->ui->getByPath("Manager.ViewOptions.Orthographic");
                if (elem)
                {
                    if (auto button = dynamic_cast<ui::Button *>(elem))
                    {
                        button->setState(!button->state());
                        button->trigger();
                    }
                    else
                    {
                        std::cerr << "CfdGuiPlugin:  GRMsg " << keyword << ": Manager.ViewOptions.Orthographic not a Button" << std::endl;
                    }
                }
                else
                {
                    std::cerr << "CfdGuiPlugin:  GRMsg " << keyword << ": did not find Manager.ViewOptions.Orthographic" << std::endl;
                }
            }
        }
    }
}

void CfdGuiPlugin::menuEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "CfdGuiPlugin::menuItem::menuEvent for %s\n", menuItem->getName());

    if (menuItem == forwardButton_)
        sendPresentationForwardMsgToGui();
    else if (menuItem == reloadButton_)
        sendPresentationReloadMsgToGui();
    else if (menuItem == backwardButton_)
        sendPresentationBackwardMsgToGui();
    else if (menuItem == playButton_)
        sendPresentationPlayMsgToGui();
    else if (menuItem == stopButton_)
        sendPresentationStopMsgToGui();
    else if (menuItem == toStartButton_)
        sendPresentationToStartMsgToGui();
    else if (menuItem == toEndButton_)
        sendPresentationToEndMsgToGui();
}

void
CfdGuiPlugin::sendPresentationForwardMsgToGui()
{
    if (hideMenu)
        coVRNavigationManager::instance()->setMenuMode(false);

    if (coVRMSController::instance()->isMaster())
    {
        //fprintf(stderr,"CfdGuiPlugin::sendPresentationForwardMsgToGui\n");
        coGRKeyWordMsg keyWordMsg("PRESENTATION_FORWARD", false);
        cover->sendGrMessage(keyWordMsg);
    }
}

void
CfdGuiPlugin::sendPresentationReloadMsgToGui()
{
    if (hideMenu)
        coVRNavigationManager::instance()->setMenuMode(false);

    if (coVRMSController::instance()->isMaster())
    {
        //fprintf(stderr,"CfdGuiPlugin::sendPresentationReloadMsgToGui\n");
        coGRKeyWordMsg keyWordMsg("PRESENTATION_RELOAD", false);
        cover->sendGrMessage(keyWordMsg);
    }
}

void
CfdGuiPlugin::sendPresentationBackwardMsgToGui()
{
    if (hideMenu)
        coVRNavigationManager::instance()->setMenuMode(false);

    if (coVRMSController::instance()->isMaster())
    {

        //fprintf(stderr,"CfdGuiPlugin::sendPresentationBackwardMsgToGui\n");
        coGRKeyWordMsg keyWordMsg("PRESENTATION_BACKWARD", false);
        cover->sendGrMessage(keyWordMsg);
    }
}
void
CfdGuiPlugin::sendPresentationPlayMsgToGui()
{
    if (coVRMSController::instance()->isMaster())
    {

        coGRKeyWordMsg keyWordMsg("PRESENTATION_PLAY", false);
        cover->sendGrMessage(keyWordMsg);
    }
}
void
CfdGuiPlugin::sendPresentationStopMsgToGui()
{
    if (hideMenu)
        coVRNavigationManager::instance()->setMenuMode(false);

    if (coVRMSController::instance()->isMaster())
    {

        coGRKeyWordMsg keyWordMsg("PRESENTATION_STOP", false);
        cover->sendGrMessage(keyWordMsg);
    }
}

void
CfdGuiPlugin::sendPresentationToStartMsgToGui()
{
    if (hideMenu)
        coVRNavigationManager::instance()->setMenuMode(false);

    if (coVRMSController::instance()->isMaster())
    {
        //fprintf(stderr,"CfdGuiPlugin::sendPresentationToStartMsgToGui\n");
        coGRKeyWordMsg keyWordMsg("PRESENTATION_GO_TO_START", false);
        cover->sendGrMessage(keyWordMsg);
    }
}

void
CfdGuiPlugin::sendPresentationToEndMsgToGui()
{
    if (hideMenu)
        coVRNavigationManager::instance()->setMenuMode(false);

    if (coVRMSController::instance()->isMaster())
    {
        //fprintf(stderr,"CfdGuiPlugin::sendPresentationToEndMsgToGui\n");

        coGRKeyWordMsg keyWordMsg("PRESENTATION_GO_TO_END", false);
        cover->sendGrMessage(keyWordMsg);
    }
}

void
CfdGuiPlugin::handleTransformCaseMsg(const char *caseName, float *row0, float *row1, float *row2, float *row3)
{
    //fprintf(stderr,"CfdGuiPlugin::handleTransformCaseMsg (case=%s)\n", caseName);
    osg::MatrixTransform *dcs, *interDCS;

    if (caseName)
    {

        //set the matrix of the dcs
        osg::Matrix m;

        m.set(row0[0], row1[0], row2[0], row3[0],
              row0[1], row1[1], row2[1], row3[1],
              row0[2], row1[2], row2[2], row3[2],
              row0[3], row1[3], row2[3], row3[3]);

        dcs = VRSceneGraph::instance()->findFirstNode<osg::MatrixTransform>(caseName);

        if (dcs)
        {
            //fprintf(stderr,"...found case DCS\n");
            dcs->setMatrix(m);
        }
        //else
        //  fprintf(stderr,"...! found case DCS named %s\n", caseName);

        string interactorCaseName(caseName);
        interactorCaseName += "_INTERACTOR";
        //fprintf(stderr,"---looking for interactor case dcs named %s\n", interactorCaseName.c_str());
        interDCS = VRSceneGraph::instance()->findFirstNode<osg::MatrixTransform>(interactorCaseName.c_str(), false, cover->getObjectsScale());
        if (interDCS)
        {
            //fprintf(stderr,"...found inter case dcs numChildren=%d\n", interDCS->getNumChildren());
            interDCS->setMatrix(m);
        }
        else
        {
            //fprintf(stderr,"... !could not find inter case dcs %s in sg\n", interactorCaseName.c_str());
            ///osgDB::writeNodeFile((*cover->getObjectsScale()), "searchInterCase.osg");
        }
    }
}

void CfdGuiPlugin::key(int type, int keySym, int mod)
{
    switch (type)
    {
    case (osgGA::GUIEventAdapter::KEYDOWN):
        if (mod & osgGA::GUIEventAdapter::MODKEY_ALT)
        {
            if (keySym == 'o' || keySym == 'O')
            {
                //fprintf(stderr,"CfdGuiPlugin::sendPresentationBackwardMsgToGui\n");
                coGRKeyWordMsg keyWordMsg("PRESENTATION_BACKWARD", false);
                cover->sendGrMessage(keyWordMsg);
            }

            if (keySym == 'p' || keySym == 'P')
            {
                //fprintf(stderr,"CfdGuiPlugin::sendPresentationForwardMsgToGui\n");
                coGRKeyWordMsg keyWordMsg("PRESENTATION_FORWARD", false);
                cover->sendGrMessage(keyWordMsg);
            }
        }
    }
}

COVERPLUGIN(CfdGuiPlugin)
