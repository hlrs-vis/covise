/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Vrml97 Plugin (does nothing)                                **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner                                                       **
 **                                                                          **
 ** History:                                                                 **
 ** Nov-01  v1                                                               **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <config/CoviseConfig.h>
#include <net/message.h>
#include <net/message_types.h>
#include <util/byteswap.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <cover/RenderObject.h>
#include <cover/OpenCOVER.h>
#include <cover/coHud.h>
#include <cover/coVRPluginList.h>
#include <cover/VRRegisterSceneGraph.h>

#include <vrml97/vrml/Player.h>
#include <vrml97/vrml/VrmlNodeCOVER.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlMFString.h>
#include <vrml97/vrml/VrmlSFString.h>
#include <vrml97/vrml/Doc.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/VrmlNamespace.h>

#include "Vrml97Plugin.h"
#include "ListenerCover.h"
#include "SystemCover.h"
#include "ViewerOsg.h"

#include "VrmlNodeTUI.h"
#include "VrmlNodeTimesteps.h"
#include "VrmlNodeCOVERPerson.h"
#include "VrmlNodeCOVERBody.h"
#include "VrmlNodeARSensor.h"
#include "VrmlNodeMirrorCamera.h"
#include "VrmlNodeMultiTouchSensor.h"
#include "VrmlNodeCOVISEObject.h"
#include "VrmlNodeClippingPlane.h"
#include "VrmlNodeShadowedScene.h"
#include "VrmlNodePrecipitation.h"
#include "VrmlNodeMatrixLight.h"
#include "VrmlNodePhotometricLight.h"

#include <osgGA/GUIEventAdapter>
#include <osgDB/Registry>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osgDB/WriteFile>
#include <osgUtil/Optimizer>

#include <grmsg/coGRObjShaderObjMsg.h>
#include <grmsg/coGRObjVisMsg.h>
#include <grmsg/coGRObjSetMoveMsg.h>
#include <grmsg/coGRObjSensorMsg.h>
#include <grmsg/coGRObjSensorEventMsg.h>
#include <grmsg/coGRKeyWordMsg.h>

#include <cover/ui/Action.h>

using namespace covise;
using namespace grmsg;

Vrml97Plugin *Vrml97Plugin::plugin = NULL;

static FileHandler handlers[] = {
    { Vrml97Plugin::loadUrl,
      Vrml97Plugin::loadVrml,
      Vrml97Plugin::unloadVrml,
      "wrl" },
    { Vrml97Plugin::loadUrl,
      Vrml97Plugin::loadVrml,
      Vrml97Plugin::unloadVrml,
      "wrl.gz" },
    { Vrml97Plugin::loadUrl,
      Vrml97Plugin::loadVrml,
      Vrml97Plugin::unloadVrml,
      "wrz" },
      { Vrml97Plugin::loadUrl,
	  Vrml97Plugin::loadVrml,
	  Vrml97Plugin::unloadVrml,
	  "x3d" },
      { Vrml97Plugin::loadUrl,
	  Vrml97Plugin::loadVrml,
	  Vrml97Plugin::unloadVrml,
	  "x3dv" }
};

// descend two levels into vrml scene graph
osg::Node *Vrml97Plugin::getRegistrationRoot()
{
    const int descend = 2;

    if (!plugin)
        return NULL;
    if (!plugin->viewer)
        return NULL;

    osg::Group *g = plugin->viewer->VRMLRoot;
    int level = 0;
    while (g)
    {
        ++level;
        if (g->getNumChildren() < 1)
            return NULL;
        if (level == descend)
        {
            return g->getChild(0);
        }
        g = dynamic_cast<osg::Group *>(g->getChild(0));
    }
    return plugin->viewer->VRMLRoot;
}

int Vrml97Plugin::loadUrl(const Url &url, osg::Group *group, const char *ck)
{
    return loadVrml(url.str().c_str(), group, ck);
}

int Vrml97Plugin::loadVrml(const char *filename, osg::Group *group, const char *)
{
    //fprintf(stderr, "----Vrml97Plugin::loadVrml %s\n", filename);
    fprintf(stderr, "Loading VRML %s\n", filename);
    if (plugin->vrmlScene)
    {
        VrmlMFString url(filename);
        if (group)
            plugin->viewer->setRootNode(group);
        else
            plugin->viewer->setRootNode(cover->getObjectsRoot());
        plugin->vrmlScene->clearRelativeURL();
        plugin->vrmlScene->loadUrl(&url, NULL, false);


        //allow plugin to unregister
        VRRegisterSceneGraph::instance()->unregisterNode(getRegistrationRoot(), "root");

        plugin->isNewVRML = true;
    }
    else
    {
        const char *local = NULL;
        Doc url(filename);
        std::string proto = url.urlProtocol();
        if (proto.empty() || proto=="file")
            local = filename;
        static bool creatingVrmlScene = false;
        if (!creatingVrmlScene)
        {
            creatingVrmlScene = true;
            plugin->vrmlScene = new VrmlScene(filename, local);
            creatingVrmlScene = false;
            if (plugin->vrmlScene->loadSucceeded())
            {
                if (group)
                    plugin->viewer = new ViewerOsg(plugin->vrmlScene, group);
                else
                    plugin->viewer = new ViewerOsg(plugin->vrmlScene, cover->getObjectsRoot());
                plugin->viewer->setPlayer(plugin->player);
                plugin->vrmlScene->addWorldChangedCallback(worldChangedCB);
                // worldChangedCB(VrmlScene::REPLACE_WORLD);
                plugin->isNewVRML = true;
            }
            else
            {
                delete plugin->vrmlScene;
                plugin->vrmlScene = NULL;
            }
        }
    }
	if (plugin->vrmlScene)
	{
		if (plugin->vrmlScene->wasEncrypted())
		{
			cover->protectScenegraph();
		}
		plugin->vrmlFilename = filename;
		OpenCOVER::instance()->hud->setText2("done parsing");
		if (plugin->viewer)
			plugin->viewer->update();
		if (plugin->player)
			plugin->player->update();
		if (System::the)
			System::the->update();
		osg::Node* root = getRegistrationRoot();

		VRRegisterSceneGraph::instance()->registerNode(root, "root");
		// set OPAQUE_BIN for VRML root
		if (coCoviseConfig::isOn("COVER.Plugin.Vrml97.ForceOpaqueBin", false))
		{
			osg::StateSet* sset = root->getOrCreateStateSet();
			sset->setRenderBinMode(osg::StateSet::OVERRIDE_RENDERBIN_DETAILS);
			sset->setRenderingHint(osg::StateSet::OPAQUE_BIN);
		}

		for (int i = 0; i < plugin->viewer->sensors.size(); i++)
		{

			// send sensors to GUI (a message for each sensor)
			coGRObjSensorMsg sensorMsg(coGRMsg::SENSOR, plugin->vrmlFilename.c_str(), i);
            cover->sendGrMessage(sensorMsg);
		}
		return 0;
	}
	return -1;
}

int Vrml97Plugin::replaceVrml(const char *filename, osg::Group *group, const char *)
{
    if (plugin->viewer)
        delete plugin->viewer;
    plugin->viewer = NULL;
    if (plugin->vrmlScene)
        delete plugin->vrmlScene;
    plugin->vrmlScene = NULL;

    plugin->vrmlScene = new VrmlScene(filename, filename);
    if (group)
        plugin->viewer = new ViewerOsg(plugin->vrmlScene, group);
    else
        plugin->viewer = new ViewerOsg(plugin->vrmlScene, cover->getObjectsRoot());
    plugin->viewer->setPlayer(plugin->player);
    plugin->vrmlScene->addWorldChangedCallback(worldChangedCB);
    if (plugin->vrmlScene->wasEncrypted())
    {
        cover->protectScenegraph();
    }
    plugin->vrmlFilename = filename;

    if (plugin->viewer)
        plugin->viewer->update();
    if (plugin->player)
        plugin->player->update();
    if (System::the)
        System::the->update();

    // send unregister
    VRRegisterSceneGraph::instance()->unregisterNode(getRegistrationRoot(), "root");
    plugin->isNewVRML = true;

    return 0;
}

int Vrml97Plugin::unloadVrml(const char *filename, const char *)
{
    (void)filename;
    //fprintf(stderr, "----Vrml97Plugin::unloadVrml %s\n", filename);

    // send unregister
    VRRegisterSceneGraph::instance()->unregisterNode(getRegistrationRoot(), "root");

    if (plugin->viewer)
        delete plugin->viewer;
    plugin->viewer = NULL;
    if (plugin->vrmlScene)
        delete plugin->vrmlScene;
    plugin->vrmlScene = NULL;
    //plugin->vrmlFilename = "";

    plugin->isNewVRML = true;

    return 0;
}

void Vrml97Plugin::worldChangedCB(int reason)
{
    switch (reason)
    {
    case VrmlScene::DESTROY_WORLD:
        delete plugin->viewer;
        delete plugin->vrmlScene;
        plugin->viewer = NULL;
        plugin->vrmlScene = NULL;
        break;

    case VrmlScene::REPLACE_WORLD:
        plugin->viewer->startLoadTime = 0;
        break;
    }
}

Vrml97Plugin::Vrml97Plugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("Vrml97Plugin", cover->ui)
, listener(NULL)
, viewer(NULL)
, vrmlScene(NULL)
, player(NULL)
, sensorList(NULL)
{
    //fprintf(stderr,"Vrml97Plugin::Vrml97Plugin\n");
    if (plugin)
    {
        if (cover->debugLevel(1))
            fprintf(stderr, "have already an instance of Vrml97Plugin !!!\n");
        return;
    }

    plugin = this;
}

bool Vrml97Plugin::init()
{
    if (System::the == NULL)
    {
        system = new SystemCover();
        System::the = system;
    }
    else
    {
        if (cover->debugLevel(1))
            fprintf(stderr, "Vrml97Plugin::Vrml97Plugin(): System::the was non-NULL !!!\n");
        return false;
    }

    if (!coVRMSController::instance()->isSlave())
    {
        listener = new ListenerCover(cover);
        this->player = listener->createPlayer();
        cover->registerPlayer(player);
    }

    sensorList = new coSensorList();

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);
	coVRFileManager::instance()->registerFileHandler(&handlers[2]);
	coVRFileManager::instance()->registerFileHandler(&handlers[3]);
	coVRFileManager::instance()->registerFileHandler(&handlers[4]);

    VrmlNamespace::addBuiltIn(VrmlNodeTUIProgressBar::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUITab::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUITabFolder::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUIButton::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUIToggleButton::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUIFrame::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUISplitter::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUIListBox::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUIMap::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUIComboBox::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUISlider::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUIFloatSlider::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTUILabel::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeTimesteps::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeCOVERPerson::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeCOVERBody::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeCOVISEObject::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodePrecipitation::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeMatrixLight::defineType());
#ifdef HAVE_VRMLNODEPHOTOMETRICLIGHT
    VrmlNamespace::addBuiltIn(VrmlNodePhotometricLight::defineType());
#endif

    VrmlNamespace::addBuiltIn(VrmlNodeARSensor::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeMirrorCamera::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeMultiTouchSensor::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeClippingPlane::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeShadowedScene::defineType());

    coEventQueue::registerEventType(&VrmlNodeARSensor::AREventType);
    coEventQueue::registerEventType(&VrmlNodeMultiTouchSensor::MultiTouchEventType);

    vrmlFilename = "";

    return true;
}

// this is called if the plugin is removed at runtime
Vrml97Plugin::~Vrml97Plugin()
{
    unloadVrml("");

    if (!coVRMSController::instance()->isSlave())
    {
        if (listener)
        {

            cover->unregisterPlayer(player);
            listener->destroyPlayer(this->player);
            this->player = NULL;
            delete listener;
            listener = NULL;
        }
    }

	coVRFileManager::instance()->unregisterFileHandler(&handlers[4]);
	coVRFileManager::instance()->unregisterFileHandler(&handlers[3]);
	coVRFileManager::instance()->unregisterFileHandler(&handlers[2]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);

    coEventQueue::unregisterEventType(&VrmlNodeARSensor::AREventType);

    delete sensorList;
    sensorList = NULL;

    delete System::the;
}

bool
Vrml97Plugin::update()
{
    bool render = false;

    if (this->sensorList)
        sensorList->update();
    if (this->viewer)
    {
        render = this->viewer->update();
    }
    if (this->player)
        this->player->update();
    if (System::the)
        System::the->update();

    return render;
}

void
Vrml97Plugin::preFrame()
{
    VrmlNodeMatrixLight::updateAll();
#ifdef HAVE_VRMLNODEPHOTOMETRICLIGHT
    VrmlNodePhotometricLight::updateAll();
#endif
    if (plugin->viewer)
	{
		if (plugin->viewer->VRMLRoot && (plugin->isNewVRML || coSensiveSensor::modified))
		{
			plugin->isNewVRML = false;
			coSensiveSensor::modified = false;

			for (int i = 0; i < plugin->viewer->sensors.size(); i++)
			{
				coSensiveSensor *s = plugin->viewer->sensors[i];
				osg::Node *n = s->getNode();
				cover->setNodesIsectable(n, true);
			}
		}

		viewer->preFrame();
    }
}

void
Vrml97Plugin::key(int type, int keySym, int mod)
{
    VrmlNodeCOVER::KeyEventType vrmlType = VrmlNodeCOVER::Unknown;
    switch (type)
    {
    case (osgGA::GUIEventAdapter::KEYDOWN):
        vrmlType = VrmlNodeCOVER::Press;
        break;
    case (osgGA::GUIEventAdapter::KEYUP):
        vrmlType = VrmlNodeCOVER::Release;
        break;
    default:
        cerr << "Vrml97Plugin::keyEvent: unknown key event type " << type << endl;
        return;
    }

    char keyString[1024] = "";
    if (mod & osgGA::GUIEventAdapter::MODKEY_CTRL)
    {
        strcat(keyString, "Ctrl-");
    }
    if (mod & osgGA::GUIEventAdapter::MODKEY_ALT)
    {
        strcat(keyString, "Alt-");
    }
    if (mod & osgGA::GUIEventAdapter::MODKEY_META)
    {
        strcat(keyString, "Meta-");
    }

    if (!(keySym & 0xff00))
    {
        char buf[2] = { static_cast<char>(keySym), '\0' };
        strcat(keyString, buf);
    }
    else if (keySym >= osgGA::GUIEventAdapter::KEY_F1 && keySym <= osgGA::GUIEventAdapter::KEY_F35)
    {
        char buf[10];
        sprintf(buf, "F%d", keySym - (osgGA::GUIEventAdapter::KEY_F1 - 1));
        strcat(keyString, buf);
    }
    else if (keySym >= osgGA::GUIEventAdapter::KEY_KP_0 && keySym <= osgGA::GUIEventAdapter::KEY_KP_9)
    {
        char buf[] = { static_cast<char>('0' + keySym - osgGA::GUIEventAdapter::KEY_KP_0), '\0' };
        strcat(keyString, buf);
    }
    else
    {
        switch (keySym)
        {
        case osgGA::GUIEventAdapter::KEY_Home:
            strcat(keyString, "Home");
            break;
        case osgGA::GUIEventAdapter::KEY_Left:
            strcat(keyString, "Left");
            break;
        case osgGA::GUIEventAdapter::KEY_Up:
            strcat(keyString, "Up");
            break;
        case osgGA::GUIEventAdapter::KEY_Right:
            strcat(keyString, "Right");
            break;
        case osgGA::GUIEventAdapter::KEY_Down:
            strcat(keyString, "Down");
            break;
        case osgGA::GUIEventAdapter::KEY_Page_Up:
            strcat(keyString, "Page_Up");
            break;
        case osgGA::GUIEventAdapter::KEY_Page_Down:
            strcat(keyString, "Page_Down");
            break;
        case osgGA::GUIEventAdapter::KEY_End:
            strcat(keyString, "End");
            break;
        case osgGA::GUIEventAdapter::KEY_Begin:
            strcat(keyString, "Begin");
            break;
        case osgGA::GUIEventAdapter::KEY_Tab:
            strcat(keyString, "Tab");
            break;
        case osgGA::GUIEventAdapter::KEY_Escape:
            strcat(keyString, "Escape");
            break;
        case osgGA::GUIEventAdapter::KEY_Delete:
            strcat(keyString, "Delete");
            break;
        case osgGA::GUIEventAdapter::KEY_BackSpace:
            strcat(keyString, "Backspace");
            break;
        case osgGA::GUIEventAdapter::KEY_Return:
            strcat(keyString, "Enter");
            break;
        case osgGA::GUIEventAdapter::KEY_Shift_L:
            strcat(keyString, "Shift_L");
            break;
        case osgGA::GUIEventAdapter::KEY_Shift_R:
            strcat(keyString, "Shift_R");
            break;
        case osgGA::GUIEventAdapter::KEY_Alt_L:
            strcat(keyString, "Alt_L");
            break;
        case osgGA::GUIEventAdapter::KEY_Alt_R:
            strcat(keyString, "Alt_R");
            break;
        case osgGA::GUIEventAdapter::KEY_Control_L:
            strcat(keyString, "Control_L");
            break;
        case osgGA::GUIEventAdapter::KEY_Control_R:
            strcat(keyString, "Control_R");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Left:
            strcat(keyString, "KeyPad_Left");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Up:
            strcat(keyString, "KeyPad_Up");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Right:
            strcat(keyString, "KeyPad_Right");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Down:
            strcat(keyString, "KeyPad_Down");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Home:
            strcat(keyString, "KeyPad_Home");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Page_Up:
            strcat(keyString, "KeyPad_Page_Up");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Page_Down:
            strcat(keyString, "KeyPad_Page_Down");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_End:
            strcat(keyString, "KeyPad_End");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Insert:
            strcat(keyString, "KeyPad_Insert");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Begin:
            strcat(keyString, "KeyPad_Begin");
            break;

        case osgGA::GUIEventAdapter::KEY_KP_Multiply:
            strcat(keyString, "KeyPad_Multiply");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Add:
            strcat(keyString, "KeyPad_Add");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Subtract:
            strcat(keyString, "KeyPad_Subtract");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Divide:
            strcat(keyString, "KeyPad_Divide");
            break;
        case osgGA::GUIEventAdapter::KEY_KP_Enter:
            strcat(keyString, "KeyPad_Enter");
            break;
        default:
            strcat(keyString, "Unknown Key");
            break;
        }
    }

    //fprintf(stderr, "sending key: type=%d, keySym=%d, mod=0x%08x (%s) to theCOVER\n", type, keySym, mod, keyString);

    if (theCOVER)
    {
        theCOVER->keyEvent(vrmlType, keyString);
    }

    if (vrmlType == VrmlNodeCOVER::Press && (!strcmp(keyString, "Alt-a") || !strcmp(keyString, "Alt-A")))
    {
        // reconnect audio server
        if (!coVRMSController::instance()->isSlave())
        {
            if (cover->debugLevel(1))
                cerr << "Reconnecting to audio server..." << endl;
            plugin->player->restart();
        }
    }
    if (cover->debugLevel(1))
    {
        if (vrmlType == VrmlNodeCOVER::Release)
            fprintf(stderr, "Vrml97Plugin::key: Release ");
        if (vrmlType == VrmlNodeCOVER::Press)
            fprintf(stderr, "Vrml97Plugin::key: Press ");
        fprintf(stderr, "%s\n", keyString);
    }
    if (vrmlType == VrmlNodeCOVER::Release && (!strcmp(keyString, "Alt-W")))
    {
        std::string filename = "./vrmlOpt.osg";
        // reconnect audio server
        if (!coVRMSController::instance()->isSlave())
        {
			if (System::the->doOptimize())
			{
				osgUtil::Optimizer optimizer;
				optimizer.optimize(plugin->viewer->VRMLRoot, 0xfffffff);
			}

            if (osgDB::writeNodeFile(*plugin->viewer->VRMLRoot, filename.c_str()))
            {
                std::cerr << "Data written to \"" << filename << "\"." << std::endl;
            }
            else
            {
                std::cerr << "Writing to \"" << filename << "\" failed." << std::endl;
            }
        }
    }
}

void Vrml97Plugin::message(int toWhom, int type, int len, const void *buf)
{
    if (len >= strlen("activateTouchSensor0"))
    {
        if (strncmp(((const char *)buf), "activateTouchSensor0", strlen("activateTouchSensor0")) == 0)
        {
            activateTouchSensor(0);
        }
        else if (strncmp(((const char *)buf), "activateTouchSensor1", strlen("activateTouchSensor1")) == 0)
        {
            activateTouchSensor(1);
        }
    }

    if (toWhom != coVRPluginSupport::VRML_EVENT)
        return;

    if (this->vrmlScene)
        vrmlScene->getIncomingSensorEventQueue()->receiveMessage(type, len, buf);
}

void Vrml97Plugin::addNode(osg::Node *node, const RenderObject *)
{
    VrmlNodeCOVISEObject::addNode(node);
}

void Vrml97Plugin::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin Vrml97 coVRGuiToRenderMsg [%s]\n", msg.getString().c_str());

    if (!vrmlScene)
        return;

    if (msg.isValid())
    {
        switch (msg.getType())
        {
        case coGRMsg::GEO_VISIBLE:
        {
            // The same message is handled in VRCoviseConnection (with the same result) so it might be safe to remove here.
            auto &geometryVisibleMsg = msg.as<coGRObjVisMsg>();
            const char *objectName = geometryVisibleMsg.getObjName();

            if (cover->debugLevel(3))
                fprintf(stderr, "in Vrml97 coGRMsg::GEO_VISIBLE object=%s visible=%d\n", objectName, geometryVisibleMsg.isVisible());

            osg::Node *node = VRSceneGraph::instance()->findFirstNode<osg::Node>(objectName, viewer->VRMLRoot);
            if (node)
            {
                if (geometryVisibleMsg.isVisible())
                {
                    node->setNodeMask(node->getNodeMask() | (Isect::Visible));
                }
                else
                {
                    node->setNodeMask(node->getNodeMask() & (~(Isect::Visible| Isect::OsgEarthSecondary)));
                }
            }
        }
        break;
                case coGRMsg::SET_MOVE:
        {
            auto &setMoveMsg = msg.as<coGRObjSetMoveMsg>();
            const char *objectName = setMoveMsg.getObjName();

            if (cover->debugLevel(3))
                fprintf(stderr, "in Vrml97Plugin coGRMsg::SET_MOVE object=%s isMoveable=%d\n", objectName, setMoveMsg.isMoveable());

            osg::Node *node = NULL;
            node = VRSceneGraph::instance()->findFirstNode<osg::Node>(objectName, viewer->VRMLRoot);
            if (node)
            {
                //                if (setMoveMsg.isMoveable())
                //                {
                //                   cover->setNodeIntersectable(node, true);
                //                }
                //                else
                //                {
                //                   cover->setNodeIntersectable(node, false);
                //                }
            }
        }
        break;
                case coGRMsg::SENSOR_EVENT:
        {
            auto &sensorEventMsg = msg.as<coGRObjSensorEventMsg>();
            const char *objectName = sensorEventMsg.getObjName();

            if (cover->debugLevel(3))
                fprintf(stderr, "in Vrml97Plugin coGRMsg::SENSOR_EVENT object=%s sensorID=%d isOver=%d isActive=%d\n", objectName, sensorEventMsg.getSensorId(), sensorEventMsg.isOver(), sensorEventMsg.isActive());

            double dummyPoint[3];
            double dummyMatrix[16];

            //std::cout << "file found" << std::endl;
            vrmlScene->sensitiveEvent(
                viewer->sensors[sensorEventMsg.getSensorId()]->getVrmlObject(),
                System::the->time(),
                sensorEventMsg.isOver(), sensorEventMsg.isActive(),
                dummyPoint,
                dummyMatrix);
        }
        break;
        default:
            break;
        }
    }
}

void Vrml97Plugin::activateTouchSensor(int id)
{
    //fprintf(stderr, "Vrml97Plugin::activateTouchSensor %d\n", id);
    double dummyPoint[3];
    double dummyMatrix[16];

    // return if no viewer (no vrml loaded),
    //  if id is greater then size of sensors
    //  or if id is invalid
    if (viewer == NULL || viewer->sensors.size() <= id || id < 0)
        return;

    vrmlScene->sensitiveEvent(
        viewer->sensors[id]->getVrmlObject(),
        System::the->time(),
        true, true,
        dummyPoint,
        dummyMatrix);

    vrmlScene->sensitiveEvent(
        viewer->sensors[id]->getVrmlObject(),
        System::the->time(),
        true, false,
        dummyPoint,
        dummyMatrix);
}

ui::Element *Vrml97Plugin::getMenuButton(const std::string &buttonName)
{
    if (buttonName.find("activateTouchSensor") == 0)
    {
        int i = atoi(buttonName.substr(19).c_str());
        if (viewer && viewer->sensors.size() > i)
        {
            if (auto a = viewer->sensors[i]->getButton())
            {
                if (!a->callback())
                {
                    a->setCallback([this, i](){
                        activateTouchSensor(i);
                    });
                }
            }
            return viewer->sensors[i]->getButton();
        }
        else
            return NULL;
    }

    return NULL;
}

#if 0
void Vrml97Plugin::menuEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "Vrml97Plugin::menuItem::menuEvent for %s\n", menuItem->getName());

    if (viewer)
    {
        for (int i = 0; i < viewer->sensors.size(); i++)
        {
            if (menuItem == viewer->sensors[i]->getButton())
            {
                activateTouchSensor(i);
                break;
            }
        }
    }
}
#endif


void Vrml97Plugin::preDraw(osg::RenderInfo &renderInfo)  // implementierung von virtual void preDraw(osg::RenderInfo &), definiert in d:\src\covise\src\OpenCOVER\cover\coVRPlugin.h
{
#ifdef HAVE_VRMLNODEPHOTOMETRICLIGHT
        VrmlNodePhotometricLight::updateLightTextures(renderInfo); // note the s
#endif
}

COVERPLUGIN(Vrml97Plugin)
