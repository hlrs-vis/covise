/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coInteractor.h>
#include "VRCoviseObjectManager.h"
#include "VRCoviseConnection.h"
#include "coVRMenuList.h"
#include "CovisePlugin.h"
#include <net/message.h>
#include <util/coTimer.h>
#include <covise/covise_appproc.h>
#include <appl/RenderInterface.h>
#include <cover/VRSceneGraph.h>
#include <OpenVRUI/sginterface/vruiButtons.h>

#include "VRSlider.h"
#include "VRRotator.h"
#include "VRVectorInteractor.h"
#include "coVRTUIParam.h"
#include "coVRDistributionManager.h"

#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRCommunication.h>
#include <PluginUtil/FeedbackManager.h>
#include <PluginUtil/ModuleInteraction.h>

#include <cover/ui/Action.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Manager.h>

#include <net/message.h>

using namespace covise;
using namespace opencover;
using namespace std;
using vrui::vruiButtons;

CovisePlugin::CovisePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("CovisePlugin", cover->ui)
{
    std::cerr << "Starting COVISE connection..." << std::endl;
    new ObjectManager(this);
    new VRCoviseConnection();
    CoviseRender::set_custom_callback([this](const covise::Message &msg)
                                      { handleVrbMessage(msg); });
}

static void messageCallback(const DataHandle &dh)
{
    coVRPluginList::instance()->forwardMessage(dh);
}

bool CovisePlugin::init()
{
    coVRDistributionManager::instance().init();

    if (!cover->visMenu)
    {
        cover->visMenu = new ui::Menu("COVISE", this);
        auto e = new ui::Action("Execute", cover->visMenu);
        cover->visMenu->add(e, ui::Container::KeepFirst);
        e->setShortcut("e");
        e->setCallback([this](){
            executeAll();
        });
        e->setIcon("view-refresh");
        e->setPriority(ui::Element::Toolbar);

        auto selectInteract = cover->ui->getByPath("NavigationManager.Navigation.Modes.SelectInteract");
        if (selectInteract)
        {
            selectInteract->setEnabled(true);
            selectInteract->setVisible(true);
            //cover->visMenu->add(selectInteract);
        }
    }

    //CoviseRender::set_custom_callback(CovisePlugin::OpenCOVERCallback, this); //get covisemessages from 
    CoviseRender::set_render_module_callback(messageCallback);
    return VRCoviseConnection::covconn;
}

CovisePlugin::~CovisePlugin()
{
    if (cover->visMenu)
    {
        cover->visMenu = NULL;
    }
    delete VRCoviseConnection::covconn;
    VRCoviseConnection::covconn = NULL;
}

void CovisePlugin::notify(NotificationLevel level, const char *text)
{
    if (!text || !text[0])
        return;

    std::cerr << text << std::endl;
    switch(level)
    {
        case coVRPlugin::Info:
            CoviseBase::sendInfo("%s", text);
            break;
        case coVRPlugin::Warning:
            CoviseBase::sendWarning("%s", text);
            break;
        case coVRPlugin::Error:
        case coVRPlugin::Fatal:
            CoviseBase::sendError("%s", text);
            break;
    }
}

void CovisePlugin::param(const char *paramName, bool)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "ViewPoints::param paramName=[%s]", paramName);

    if (!strcmp(paramName, "Viewpoints"))
    {
        std::string vwpPath;
        if (coVRMSController::instance()->isMaster())
        {
            const char *tmp = NULL;
            CoviseBase::get_reply_browser(&tmp);
            int length = strlen(tmp);
            coVRMSController::instance()->sendSlaves(&length, sizeof(int));
            coVRMSController::instance()->sendSlaves(tmp, length);
            vwpPath = tmp;
        }
        else
        {
            int length;
            coVRMSController::instance()->readMaster(&length, sizeof(int));
            char *tmp = new char[length + 1];
            coVRMSController::instance()->readMaster(tmp, length);
            tmp[length] = 0;
            vwpPath = tmp;
            delete[] tmp;
        }
        coVRFileManager::instance()->setViewPointFile(vwpPath);
        cover->sendMessage(this, coVRPluginSupport::TO_ALL, 0, strlen("readViewpointsFile"), "readViewpointsFile");
    }
}


static void updateScenegraph()
{
    SliderList::instance()->update();
    RotatorList::instance()->update();
    coVRMenuList::instance()->update();

    if (VRSceneGraph::instance()->m_vectorInteractor && !cover->isPointerLocked())
    {
        float position[3];
        float direction[3];
        float direction2[3];
        // retrieve pointer coordinates
        VRSceneGraph::instance()->getHandWorldPosition(position, direction, direction2);

        coPointerButton *button = cover->getPointerButton();
        if (button->wasPressed(vruiButtons::ACTION_BUTTON))
        {

            VectorInteractor::vector = vectorList.find(position[0], position[1], position[2]);
            if (VectorInteractor::vector)
            {
                VectorInteractor::vector->updateValue(position[0], position[1], position[2]);
            }
        }
        else if (button->getState() & vruiButtons::ACTION_BUTTON)
        {
            if (VectorInteractor::vector)
            {
                VectorInteractor::vector->updateValue(position[0], position[1], position[2]);
            }
        }
        else if (button->wasReleased(vruiButtons::ACTION_BUTTON))
        {
            if (VectorInteractor::vector)
            {
                VectorInteractor::vector->updateValue(position[0], position[1], position[2]);
                VectorInteractor::vector->updateParameter();
                VectorInteractor::vector = NULL;
            }
        }
    }
}

bool CovisePlugin::update()
{
    return VRCoviseConnection::covconn->update();
}

void CovisePlugin::preFrame()
{
    updateScenegraph();
}

void CovisePlugin::requestQuit(bool killSession)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (killSession)
            CoviseRender::send_quit_message();

        // send DEL to controller to do a clean exit
        if (VRCoviseConnection::covconn)
        {
            VRCoviseConnection::covconn->sendQuit();
        }
    }
}

void CovisePlugin::removeNode(osg::Node *group, bool isGroup, osg::Node *node)
{
    (void)group;

    // DON'T FORGETT TO UPDATE ROTATOR-STUFF !!!
    if (RotatorList::instance()->find(node))
        RotatorList::instance()->remove();

    // remove all sliders attached to this geometry
    SliderList::instance()->removeAll(node);
    // remove all vectors attached to this geometry
    vectorList.removeAll(node);

    // remove all vectors attached to this geometry
    tuiParamList.removeAll(node);
}

bool CovisePlugin::sendVisMessage(const Message *msg)
{
    if (CoviseRender::appmod)
    {
        if (coVRMSController::instance()->isMaster())
        {
            CoviseRender::appmod->send_ctl_msg(msg);
        }
        return true;
    }
    return false;
}

bool CovisePlugin::becomeCollaborativeMaster()
{
    if ((VRCoviseConnection::covconn))
    {
        if (coVRMSController::instance()->isMaster())
            CoviseRender::send_ui_message("FORCE_MASTER", CoviseRender::get_host());
        return true;
    }
    return false;
}

bool CovisePlugin::executeAll()
{
    if ((VRCoviseConnection::covconn))
    {
        if (coVRMSController::instance()->isMaster())
            VRCoviseConnection::covconn->executeCallback(NULL, NULL);
        return true;
    }
    return false;
}

covise::Message *CovisePlugin::waitForVisMessage(int type)
{
    return CoviseRender::appmod->wait_for_msg(type, CoviseRender::appmod->getControllerConnection());
}

void CovisePlugin::expandBoundingSphere(osg::BoundingSphere &bsphere)
{
    if (coVRMSController::instance()->isCluster() && coVRDistributionManager::instance().isActive())
    {
        struct BSphere
        {
            double x, y, z, radius;
        };
        BSphere b_sphere;
        b_sphere.x = bsphere.center()[0];
        b_sphere.y = bsphere.center()[1];
        b_sphere.z = bsphere.center()[2];
        b_sphere.radius = bsphere.radius();

        if (coVRMSController::instance()->isMaster())
        {
            coVRMSController::SlaveData result(sizeof(b_sphere));

            if (coVRMSController::instance()->readSlaves(&result) < 0)
            {
                std::cerr << "VRSceneGraph::getBoundingSphere err: sync error";
                return;
            }

            BSphere bs;
            for (std::vector<void *>::iterator i = result.data.begin();
                 i != result.data.end(); ++i)
            {
                memcpy(&bs, *i, sizeof(bs));
                osg::BoundingSphere otherBs = osg::BoundingSphere(osg::Vec3(bs.x, bs.y, bs.z), bs.radius);
                bsphere.expandBy(otherBs);
            }

            b_sphere.x = bsphere.center()[0];
            b_sphere.y = bsphere.center()[1];
            b_sphere.z = bsphere.center()[2];
            b_sphere.radius = bsphere.radius();

            coVRMSController::instance()->sendSlaves(&b_sphere, sizeof(b_sphere));
        }
        else
        {
            coVRMSController::instance()->sendMaster(&b_sphere, sizeof(b_sphere));
            coVRMSController::instance()->readMaster(&b_sphere, sizeof(b_sphere));
            bsphere = osg::BoundingSphere(osg::Vec3(b_sphere.x, b_sphere.y, b_sphere.z), b_sphere.radius);
        }
    }
}

bool CovisePlugin::requestInteraction(coInteractor *inter, osg::Node *triggerNode, bool isMouse)
{
    (void)triggerNode;

    auto interaction = FeedbackManager::instance()->findFeedback(inter);
    if (!interaction)
        return false;
    if (isMouse)
        interaction->setShowInteractorFromGui(true);
    else
        interaction->enableDirectInteractorFromGui(true);
    return true;
}

void CovisePlugin::handleVrbMessage(const covise::Message &msg)
{
    coVRCommunication::instance()->handleVRB(msg);
}

COVERPLUGIN(CovisePlugin)
