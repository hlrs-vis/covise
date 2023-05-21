/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			VRCoviseConnection.C 			*
 *									*
 *	Description		covise interface class			*
 *									*
 *	Author			D. Rainer				*
 *									*
 *	Date			23. September 96			*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include <util/common.h>
#include <util/string_util.h>
#include "VRCoviseConnection.h"
#include <cover/coVRCommunication.h>
#include <cover/coCommandLine.h>
#include <cover/coVRShader.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <config/CoviseConfig.h>

#include <util/coTimer.h>
#include <appl/RenderInterface.h>
#include <covise/covise_appproc.h>
#include "VRCoviseObjectManager.h"
#include <api/coUifElem.h>

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRNavigationManager.h>

#include <grmsg/coGRObjVisMsg.h>
#include <grmsg/coGRObjColorObjMsg.h>
#include <grmsg/coGRObjMaterialObjMsg.h>
#include <grmsg/coGRObjShaderObjMsg.h>
#include <grmsg/coGRObjSetTransparencyMsg.h>
#include <grmsg/coGRAnimationOnMsg.h>
#include <grmsg/coGRSetAnimationSpeedMsg.h>
#include <grmsg/coGRSetTimestepMsg.h>
#include <grmsg/coGRSetTrackingParamsMsg.h>
#include <grmsg/coGRObjMoveObjMsg.h>
#include <grmsg/coGRObjTransformSGItemMsg.h>
#include <grmsg/coGRObjSetNameMsg.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace opencover;
using namespace covise;
using namespace grmsg;

VRCoviseConnection *VRCoviseConnection::covconn = NULL;
static std::vector<covise::Message*>waitClusterMessages()
{
	coVRMSController* ms = coVRMSController::instance();

	if (cover->debugLevel(5))
		fprintf(stderr, "\ncoVRMSController::checkAndHandle\n");

	Message* appMsgs[100];
	Message* appMsg;
	int numMessages = 0;
	if (ms->isMaster())
	{
		MARK0("COVER cluster master checking covise messages");
		while ((numMessages < 100) && (appMsg = (CoviseRender::appmod)->check_for_ctl_msg()) != NULL)
		{
			appMsgs[numMessages] = appMsg;
			numMessages++;
		}
		ms->sendSlaves(&numMessages, sizeof(int));
		for (size_t i = 0; i < numMessages; i++)
		{
			MARK1("COVER cluster master send [%s] to cluster slave", covise_msg_types_array[appMsgs[i]->type]);
			ms->sendSlaves(appMsgs[i]);
			MARK0("done");
		}
	}
	else
	{
		MARK0("COVER cluster slave reading covise messages from cluster master");

		//get number of Messages
		if (ms->readMaster(&numMessages, sizeof(int)) < 0)
		{
			cerr << "sync_exit172 myID=" << ms->getID() << endl;
			exit(0);
		}
		for (int i = 0; i < numMessages; i++)
		{
			appMsg = new Message;
			if (ms->readMaster(appMsg) < 0)
			{
				cerr << "sync_exit18 myID=" << ms->getID() << endl;
				exit(0);
			}
			cerr <<  i+1 << ", ";
			MARK1("COVER cluster slave received [%s] from cluster master", covise_msg_types_array[appMsg->type]);
			MARK0("done");
			appMsgs[i] = appMsg;
		}
	}
	return std::vector<covise::Message*>(appMsgs, appMsgs + numMessages);
}
static std::vector<covise::Message*>waitMessages()
{
	std::vector<covise::Message*> msgs;
	while (covise::Message *msg = CoviseRender::check_event())
	{
		msgs.push_back(msg);
	}
	return msgs;
}
static void handleMessage(covise::Message* msg)
{
	CoviseRender::handle_event(msg);// handles the messange and deletes it
}
static bool checkAndHandle()
{
	std::vector<covise::Message*> msgs;
	if (coVRMSController::instance()->isCluster())
	{
		msgs = waitClusterMessages();
	}
	else
	{
		msgs = waitMessages();
	}
	if (msgs.size() == 0)
	{
		return false;
	}
	for (auto msg : msgs)
	{
		handleMessage(msg);
	}
	return true;
}
//static bool checkAndHandle()
//{
//    coVRMSController *ms = coVRMSController::instance();
//    if (!ms->isCluster())
//        return false;
//
//    if (cover->debugLevel(5))
//        fprintf(stderr, "\ncoVRMSController::checkAndHandle\n");
//
//    Message *appMsgs[100];
//    Message *appMsg;
//    int numMessages = 0;
//    if (ms->isMaster())
//    {
//        MARK0("COVER cluster master checking covise messages");
//
//        while ((numMessages < 100) && (appMsg = (CoviseRender::appmod)->check_for_ctl_msg()) != NULL)
//        {
//            appMsgs[numMessages] = appMsg;
//            numMessages++;
//        }
//
//        ms->sendSlaves(&numMessages, sizeof(int));
//
//        for (int i = 0; i < numMessages; i++)
//        {
//            MARK1("COVER cluster master send [%s] to cluster slave", covise_msg_types_array[appMsgs[i]->type]);
//            ms->sendSlaves(appMsgs[i]);
//            MARK0("done");
//            CoviseRender::set_applMsg(appMsgs[i]);
//            CoviseRender::handleControllerMessage(); // handles the messange and deletes it
//        }
//        appMsg = NULL;
//        CoviseRender::set_applMsg(appMsg);
//
//#ifdef HAS_MPI
////if (syncMode == SYNC_MPI)
////{
////MPI_Barrier(appComm);
////}
//#endif
//    }
//    else
//    {
//        MARK0("COVER cluster slave reading covise messages from cluster master");
//
//        //get number of Messages
//        if (ms->readMaster(&numMessages, sizeof(int)) < 0)
//        {
//            cerr << "sync_exit172 myID=" << ms->getID() << endl;
//            exit(0);
//        }
//        for (int i = 0; i < numMessages; i++)
//        {
//            appMsg = new Message;
//            if (ms->readMaster(appMsg) < 0)
//            {
//                cerr << "sync_exit18 myID=" << ms->getID() << endl;
//                exit(0);
//            }
//            MARK1("COVER cluster slave reveived [%s] from cluster master", covise_msg_types_array[appMsg->type]);
//            MARK0("done");
//            CoviseRender::set_applMsg(appMsg);
//            CoviseRender::handleControllerMessage(); //deletes the Message
//        }
//        appMsg = NULL;
//        CoviseRender::set_applMsg(appMsg);
//
//#ifdef HAS_MPI
////if (syncMode == SYNC_MPI)
////{
////MPI_Barrier(appComm);
////}
//#endif
//    }
//    return numMessages > 0;
//}
VRCoviseConnection::VRCoviseConnection()
{
    covconn = this;
    exitFlag = false;
    if (cover->debugLevel(3))
        fprintf(stderr, "new VRCoviseConnection\n");

    CoviseRender::reset();
    CoviseRender::set_module_description("Newest VR-Renderer");
    CoviseRender::add_port(INPUT_PORT, "RenderData", "ColorMap|Geometry|UnstructuredGrid|Points|Spheres|StructuredGrid|Polygons|TriangleStrips|Lines|Float|Vec3", "render geometry");
    CoviseRender::add_port(PARIN, "Viewpoints", "Browser", "Viewpoints");
    CoviseRender::set_port_default("Viewpoints", "./default.vwp");
    CoviseRender::add_port(PARIN, "Viewpoints___filter", "BrowserFilter", "Viewpoints");
    CoviseRender::set_port_default("Viewpoints___filter", "Viewpoints *.vwp/*");
    CoviseRender::add_port(PARIN, "Plugins", "String", "Additional plugins");
    CoviseRender::set_port_default("Plugins", "");

    // only used, when embedded="true" in WindowConfig
    CoviseRender::add_port(PARIN, "WindowID", "IntScalar", "window ID to render to");
    CoviseRender::set_port_default("WindowID", "0");

    if (coVRMSController::instance()->isMaster())
        CoviseRender::init(coCommandLine::argc(), coCommandLine::argv());

    CoviseRender::set_render_callback(VRCoviseConnection::renderCallback, this);
    CoviseRender::set_master_switch_callback(VRCoviseConnection::masterSwitchCallback, this);
    CoviseRender::set_quit_info_callback(VRCoviseConnection::quitInfoCallback, this);
    CoviseRender::set_add_object_callback(VRCoviseConnection::addObjectCallback, this);
    CoviseRender::set_covise_error_callback(VRCoviseConnection::coviseErrorCallback, this);
    CoviseRender::set_delete_object_callback(VRCoviseConnection::deleteObjectCallback, this);

    CoviseRender::set_param_callback(VRCoviseConnection::paramCallback, this);
    CoviseRender::send_ui_message("MODULE_DESC", "Newest VR-Renderer");

	if (coVRMSController::instance()->isCluster())
	{
		coVRCommunication::instance()->setWaitMessagesCallback(waitClusterMessages);
	}
	else
	{
		coVRCommunication::instance()->setWaitMessagesCallback(waitMessages);
	}
	coVRCommunication::instance()->setHandleMessageCallback(handleMessage);
}

VRCoviseConnection::~VRCoviseConnection()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "delete VRCoviseConnection\n");
    coVRCommunication::instance()->setWaitMessagesCallback(nullptr);
    coVRCommunication::instance()->setHandleMessageCallback(nullptr);
}

void VRCoviseConnection::sendQuit()
{

    CoviseRender::send_ui_message("DEL_REQ", "");
}



bool
VRCoviseConnection::update(bool handleOneMessageOnly)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "VRCoviseConnection::update\n");

    bool event = false;

    MARK0("COVER checking messages from controller");
    // check for covise messages and call the appropriate callback
    // covise messages are :
    // quitInfo, addObject, deleteObject
    // masterSwitch, render and param
    // check all pending messages
    // don't check any more after a quitInfo message

    if (coVRMSController::instance()->isCluster())
    {
        event = checkAndHandle();
    }
    else
    {
        while (CoviseRender::check_and_handle_event())
        {
            event = true;
            if (handleOneMessageOnly || exitFlag)
                break;
        }
    }
    if (ObjectManager::instance())
        ObjectManager::instance()->update();

    return event;
}

void
VRCoviseConnection::quitInfoCallback(void *userData, void *callbackData)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::quitInfoCallback\n");

    VRCoviseConnection *thisVRCoviseConnection = (VRCoviseConnection *)userData;

    thisVRCoviseConnection->quitInfo(callbackData);

    thisVRCoviseConnection->exitFlag = true;
}

void
VRCoviseConnection::masterSwitchCallback(void *userData, void *callbackData)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::masterSwitchCallback\n");

    VRCoviseConnection *thisVRCoviseConnection = (VRCoviseConnection *)userData;
    thisVRCoviseConnection->masterSwitch(callbackData);
}

void
VRCoviseConnection::addObjectCallback(void *userData, void *callbackData)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::addObjectCallback\n");

    VRCoviseConnection *thisVRCoviseConnection = (VRCoviseConnection *)userData;

    thisVRCoviseConnection->addObject(callbackData);
}

void
VRCoviseConnection::coviseErrorCallback(void *userData, void *callbackData)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::coviseErrorCallback\n");

    VRCoviseConnection *thisVRCoviseConnection = (VRCoviseConnection *)userData;

    thisVRCoviseConnection->coviseError(callbackData);
}

void
VRCoviseConnection::deleteObjectCallback(void *userData, void *callbackData)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::deleteObjectCallback\n");

    VRCoviseConnection *thisVRCoviseConnection = (VRCoviseConnection *)userData;

    thisVRCoviseConnection->deleteObject(callbackData);
}

void
VRCoviseConnection::paramCallback(bool inMapLoading, void *userData, void *callbackData)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::paramCallback\n");

    VRCoviseConnection *thisVRCoviseConnection = (VRCoviseConnection *)userData;
    thisVRCoviseConnection->localParam(inMapLoading, callbackData);
}

void
VRCoviseConnection::renderCallback(void *userData, void *callbackData)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::renderCallback\n");

    VRCoviseConnection *thisVRCoviseConnection = (VRCoviseConnection *)userData;

    thisVRCoviseConnection->render(callbackData);
}

void
VRCoviseConnection::quitInfo(void *callbackData)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "VRCoviseConnection::quitInfo\n");

    int *quit_flag = (int *)callbackData;
    *quit_flag = 1; // do not quit in appProcess, we do this later

    // set quit flag in VRCoviseConnection object
    OpenCOVER::instance()->setExitFlag(true);
    ;
}

void
VRCoviseConnection::localParam(bool inMapLoading, void *callbackData)
{
    (void)callbackData;

    if (cover->debugLevel(2))
        fprintf(stderr, "\tstartParam\n");

    const char *paramname = CoviseBase::get_reply_param_name();

    // title of module has changed -- ignore
    if (!strcmp(paramname, "SetModuleTitle"))
    {
        return;
    }

    if (!strcmp(paramname, "Plugins"))
    {
        const char *value = NULL;
        if (CoviseRender::get_reply_string(&value) && value)
        {
            std::vector<std::string> plugins = split(value, ',');
            for (size_t i = 0; i < plugins.size(); ++i)
                cover->addPlugin(strip(plugins[i]).c_str());
        }
        return;
    }

    /*    if (!strstr(paramname, "___filter") && strcmp(paramname, "Viewpoints"))
    {
        // ignore non-registered filebrowser filters
        CoviseBase::sendWarning("Received message for non-registered parameter '%s'", paramname);
    }
*/

    if (strcmp(paramname, "WindowID") == 0)
    {
        int64_t windowID;
        CoviseRender::get_reply_int64_scalar(&windowID);
        // TODO: check for valid windowID
        for (int i = 0; i < coVRConfig::instance()->numWindows(); i++)
        {
            if (coVRConfig::instance()->windows[i].embedded)
            {
#ifdef _WINDOWS
                HWND win = (HWND)windowID;
                OpenCOVER::instance()->parentWindow = win;
#else
                OpenCOVER::instance()->parentWindow = windowID;
#endif
            }
        }
    }

    coVRPluginList::instance()->param(paramname, inMapLoading);

    if (cover->debugLevel(2))
        fprintf(stderr, "\tendParam\n");
}

void
VRCoviseConnection::masterSwitch(void *)
{
}

void
VRCoviseConnection::addObject(void *cbPtr)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::addObject\n");

    struct CBdata
    {
        coDistributedObject *obj;
        char *name;
    } *cbData;
    cbData = (CBdata *)cbPtr;
    ObjectManager::instance()->addObject(cbData->name, cbData->obj);
    //VRViewer::instance()->forceCompile();
}

void
VRCoviseConnection::coviseError(void *cbPtr)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::coviseError\n");

    const char *cbData = (const char *)cbPtr;
    ObjectManager::instance()->coviseError(cbData);
}

void
VRCoviseConnection::deleteObject(void *)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::deleteObject: %s\n", CoviseRender::get_object_name());

    ObjectManager::instance()->deleteObject(CoviseRender::get_object_name());
}

void
VRCoviseConnection::render(void *)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::render\n");
    receiveRenderMessage();
}

void
VRCoviseConnection::receiveRenderMessage()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::receiveRenderMessage\n");

    char *tmp, *key;
    // get render message type
    key = CoviseRender::get_render_keyword();
    tmp = CoviseRender::get_render_data();
    
    if (strcmp(key, "AR_VIDEO_FRAME") == 0)
        coVRCommunication::instance()->processARVideoFrame(key, tmp); 

    if (strcmp(key, "INEXEC") == 0 || strcmp(key, "FINISHED") == 0)
    {
        if (coVRPlugin *ak = coVRPluginList::instance()->getPlugin("AKToolbar"))
        {
            // just indicate if a module is running
            ak->message(0, strcmp(key, "FINISHED"), 0, NULL);
        }
    }
    else if (strncmp(key, "GRMSG", 5) == 0)
    {
        cover->guiToRenderMsg(tmp);
        string fullMsg(string("GRMSG\n") + tmp);
        coGRMsg grMsg(fullMsg.c_str());
        if (grMsg.isValid())
        {
            if (grMsg.getType() == coGRMsg::GEO_VISIBLE)
            {
                coGRObjVisMsg geometryVisibleMsg(fullMsg.c_str());
                const char *objectName = geometryVisibleMsg.getObjName();
                if (cover->debugLevel(3))
                    fprintf(stderr, "in CoviseConnection coGRMsg::GEO_VISIBLE object=%s visible=%d\n", objectName, geometryVisibleMsg.isVisible());
                hideObject(objectName, !geometryVisibleMsg.isVisible());
            }

            else if (grMsg.getType() == coGRMsg::COLOR_OBJECT)
            {
                coGRObjColorObjMsg colorObjMsg(fullMsg.c_str());
                const char *objectName = colorObjMsg.getObjName();
                if (cover->debugLevel(3))
                    fprintf(stderr, "in CoviseConnection  coGRMsg::COLOR_OBJECT object=%s\n", objectName);
                int *color = new int[3];
                color[0] = colorObjMsg.getR();
                color[1] = colorObjMsg.getG();
                color[2] = colorObjMsg.getB();
                setColor(objectName, color);

                delete[] color;
            }

            else if (grMsg.getType() == coGRMsg::MATERIAL_OBJECT)
            {
                coGRObjMaterialObjMsg materialObjMsg(fullMsg.c_str());
                const char *objectName = materialObjMsg.getObjName();

                if (cover->debugLevel(3))
                    fprintf(stderr, "in CoviseConnection coGRMsg::MATERIAL_OBJECT object=%s\n", objectName);

                const int *ambient = materialObjMsg.getAmbient();
                const int *diffuse = materialObjMsg.getDiffuse();
                const int *specular = materialObjMsg.getSpecular();
                float shininess = materialObjMsg.getShininess();
                float transparency = materialObjMsg.getTransparency();
                if (cover->debugLevel(3))
                    fprintf(stderr, "coGRMsg::MATERIAL_OBJECT object=%s\n", objectName);
                setMaterial(objectName, ambient, diffuse, specular, shininess, transparency);
            }

            else if (grMsg.getType() == coGRMsg::SET_TRANSPARENCY)
            {
                coGRObjSetTransparencyMsg setTransparencyMsg(fullMsg.c_str());
                const char *objectName = setTransparencyMsg.getObjName();

                if (cover->debugLevel(3))
                    fprintf(stderr, "in VRCoviseConnection  coGRMsg::SET_TRANSPARENCY object=%s\n", objectName);

                setTransparency(objectName, setTransparencyMsg.getTransparency());
            }
            else if (grMsg.getType() == coGRMsg::SHADER_OBJECT)
            {
                coGRObjShaderObjMsg shaderObjMsg(fullMsg.c_str());
                const char *objectName = shaderObjMsg.getObjName();
                const char *shaderName = shaderObjMsg.getShaderName();

                if (cover->debugLevel(3))
                    fprintf(stderr, "in VRCoviseConnection  coGRMsg::SHADER_OBJECT object=%s\n", objectName);

                setShader(objectName, shaderName);
            }
            else if (grMsg.getType() == coGRMsg::MOVE_OBJECT)
            {
                // what is this message used for?
                coGRObjMoveObjMsg moveObjMsg(fullMsg.c_str());
                const char *objectName = moveObjMsg.getObjName();
                const char *moveName = moveObjMsg.getMoveName();
                float x = moveObjMsg.getX();
                float y = moveObjMsg.getY();
                float z = moveObjMsg.getZ();
                if (cover->debugLevel(3))
                    fprintf(stderr, "coGRMsg::MOVE_OBJECT %s object=%s\n", moveName, objectName);
                if (strcmp(moveName, "translate") == 0)
                {
                    coVRNavigationManager::instance()->doGuiTranslate(x, y, z);
                }
                else if (strcmp(moveName, "scale") == 0)
                {
                    bool scale = false;
                    if (x > 0)
                        scale = true;
                    VRSceneGraph::instance()->setScaleFromButton(scale);
                }
                else if (strcmp(moveName, "rotate") == 0)
                {
                    coVRNavigationManager::instance()->doGuiRotate(x, y, z);
                }
            }
            else if (grMsg.getType() == coGRMsg::TRANSFORM_SGITEM)
            {
                coGRObjTransformSGItemMsg transformMsg(fullMsg.c_str());
                const char *objectName = transformMsg.getObjName();
                float row0[4];
                float row1[4];
                float row2[4];
                float row3[4];
                if (cover->debugLevel(3))
                    fprintf(stderr, "coGRMsg::TRANSFORM_OBJECT object=%s\n", objectName);
                for (int i = 0; i < 4; i++)
                {
                    row0[i] = transformMsg.getMatrix(0, i);
                    row1[i] = transformMsg.getMatrix(1, i);
                    row2[i] = transformMsg.getMatrix(2, i);
                    row3[i] = transformMsg.getMatrix(3, i);
                }
                transformSGItem(objectName, row0, row1, row2, row3);
            }
            else if (grMsg.getType() == coGRMsg::SET_NAME)
            {
                coGRObjSetNameMsg setNameMsg(fullMsg.c_str());
                const char *coviseObjectName = setNameMsg.getObjName();
                const char *newName = setNameMsg.getNewName();
                //fprintf(stderr,"COVER got a SET_NAME msg from gui. objectname=%s newname=%s\n", coviseObjectName, newName);

                // if the coviseObjectName contains _SCGR_, it is a geometry loaded from file
                // else it is geometry from a covise module.
                // covise geomtry objects are handled in the plugins and their base class Modulefeedbackmanager
                // because here it is impossible to find the geode in the sg only with the coviseObjectName
                string coname(coviseObjectName);
                if (coname.find("_SCGR_") != string::npos) // this is geometry from a file
                {
                    //fprintf(stderr,"this is geomtry from a file\n");
                    osg::Node *node = VRSceneGraph::instance()->findFirstNode<osg::Node>(coviseObjectName);
                    if (node)
                    {
                        if (node->getNumDescriptions())
                        {
                            // if there is already a description which contains SCGR_ replace it
                            std::vector<std::string> dl = node->getDescriptions();
                            for (size_t i = 0; i < dl.size(); i++)
                            {
                                std::string descr = dl[i];
                                if (descr.find("_SCGR_") != string::npos)
                                {
                                    dl[i] = std::string(newName) + "_SCGR_";
                                }
                            }
                            node->setDescriptions(dl);
                        }
                        else // add a description
                            node->addDescription(string(newName) + "_SCGR_");
                    }
                }
                // else
                //    fprintf(stderr,"covise module geometry is handled in the plugins\n");
            }
        }

        /*
		string fullMsg( string("GRMSG\n")+tmp);
   	coGRMsg grMsg(fullMsg.c_str());
   	if( grMsg.isValid() )
  		{
         
			if( grMsg.getType()==coGRMsg::MOVE_OBJECT )
			{
				coGRObjMoveObjMsg moveObjMsg(fullMsg.c_str());
				const char* objectName = moveObjMsg.getObjName();
				const char* moveName = moveObjMsg.getMoveName();
				float x = moveObjMsg.getX();
				float y = moveObjMsg.getY();
				float z = moveObjMsg.getZ();
				if (cover->debugLevel(3))
					fprintf(stderr,"coGRMsg::MOVE_OBJECT %s object=%s\n",moveName, objectName);
				if (strcmp(moveName, "translate")==0)
				{
					//fprintf(stderr, "<<<<<translate %s %f %f %f\n",objectName,x,y,z);
					VRSceneGraph::sg->doGuiTranslate(x,y,z);
				}
				else if (strcmp(moveName, "scale")==0)
				{
					//fprintf(stderr, "<<<<<scale %s %f\n",objectName,x);
					VRSceneGraph::sg->doGuiScale(x);
				}
				else if (strcmp(moveName, "rotate")==0)
				{
					//fprintf(stderr, "<<<<<<rotate %s %f %f %f\n",objectName,x,y,z);
					VRSceneGraph::sg->doGuiRotate(x,y,z);
				}
			} else if (grMsg.getType()==coGRMsg::BOUNDARIES_OBJECT)
			{
				coGRObjBoundariesObjMsg boundObjMsg(fullMsg.c_str());
				const char* objectName = boundObjMsg.getObjName();
				const char* boundaries = boundObjMsg.getBoundariesName();
				if (cover->debugLevel(3))
					fprintf(stderr, "coGRMsg::BOUNDARIES_OBJECT %s object=%s\n",boundaries,objectName);
				if (strcmp("translate", boundaries)==0)
					VRSceneGraph::sg->setTranslationBoundaries(boundObjMsg.getFront(), boundObjMsg.getBack(), boundObjMsg.getLeft(), boundObjMsg.getRight(), boundObjMsg.getTop(), boundObjMsg.getBottom());
			} 
         
         else if (grMsg.getType()==coGRMsg::COLOR_OBJECT)
			{
				coGRObjColorObjMsg colorObjMsg(fullMsg.c_str());
				const char* objectName = colorObjMsg.getObjName();
				if (cover->debugLevel(3))
					fprintf(stderr, "coGRMsg::COLOR_OBJECT object=%s\n",objectName);
				int *color = new int[3];
				color[0] =colorObjMsg.getR();
				color[1] =colorObjMsg.getG();
				color[2] =colorObjMsg.getB();
				coColoringManager::setColor(objectName, color, 1.0);
				delete []color;
			} 
         
         else if (grMsg.getType()==coGRMsg::SHADER_OBJECT)
			{
				coGRObjShaderObjMsg shaderObjMsg(fullMsg.c_str());
				const char* objectName = shaderObjMsg.getObjName();
				const char* shaderName = shaderObjMsg.getShaderName();
				const char* mapFloat = shaderObjMsg.getParaFloatName();
				const char* mapVec2 = shaderObjMsg.getParaVec2Name();
				const char* mapVec3 = shaderObjMsg.getParaVec3Name();
				const char* mapVec4 = shaderObjMsg.getParaVec4Name();
				const char* mapBool = shaderObjMsg.getParaBoolName();
				const char* mapInt = shaderObjMsg.getParaIntName();
				const char* mapMat2 = shaderObjMsg.getParaMat2Name();
				const char* mapMat3 = shaderObjMsg.getParaMat3Name();
				const char* mapMat4 = shaderObjMsg.getParaMat4Name();
				if (cover->debugLevel(3))
					fprintf(stderr, "coGRMsg::SHADER_OBJECT %s object=%s\n",shaderName ,objectName);
				coColoringManager::setShader(objectName,shaderName,mapFloat,mapVec2,mapVec3,mapVec4,mapInt,mapBool, mapMat2, mapMat3, mapMat4);
			}
         
		}
*/
    }
}

void
VRCoviseConnection::executeCallback(void *, buttonSpecCell *)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::executeCallback\n");
    char buf[400];

    if (coVRMSController::instance()->isMaster())
    {
        if (CoviseRender::get_feedback_info())
        {
            strcpy(buf, CoviseRender::get_feedback_info());
            CoviseRender::set_feedback_info("C");
            CoviseRender::send_feedback_message("EXEC", "");
            CoviseRender::set_feedback_info(buf);
        }
        else
        {
            CoviseRender::set_feedback_info("C");
            CoviseRender::send_feedback_message("EXEC", "");
        }
    }
}

// hides geometry
// needed for menuevent
void
VRCoviseConnection::hideObject(const char *objName, bool hide)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "VRCoviseConnection::hideObject hide=%d\n", hide);

    osg::Node *node;

    node = VRSceneGraph::instance()->findFirstNode<osg::Node>(objName);
    if (node)
    {
        if (!hide)
            node->setNodeMask(node->getNodeMask() | (Isect::Visible));
        else
            node->setNodeMask(node->getNodeMask() & (~(Isect::Visible | Isect::OsgEarthSecondary)));
    }
}

void
VRCoviseConnection::transformSGItem(const char *objName, float *row0, float *row1, float *row2, float *row3)
{
    osg::ref_ptr<osg::Node> node = VRSceneGraph::instance()->findFirstNode<osg::Node>(objName);
    if (!node.valid())
        return;
    if (node->getNumParents() == 0)
        return;

    osg::Matrix m;
    m.set(row0[0], row1[0], row2[0], row3[0],
          row0[1], row1[1], row2[1], row3[1],
          row0[2], row1[2], row2[2], row3[2],
          row0[3], row1[3], row2[3], 1.0);

    if (node->getParent(0) == cover->getObjectsRoot())
    {

        // If we want to transform the "top" node of a vrml/osg/... model, we don't add an additional node.
        // Adding a node would cause problems with VRSceneGraphs m_attachedNode / m_addedNodes which would affect deleting the model.
        // Unfortunately, if the top node is not already a MatrixTransform, transformation will not work.
        osg::ref_ptr<osg::MatrixTransform> mt = dynamic_cast<osg::MatrixTransform *>(node.get());
        if (mt.valid())
            mt->setMatrix(m);
    }
    else
    {

        bool transformNeeded = !m.isIdentity();
        osg::ref_ptr<osg::MatrixTransform> transformNode = dynamic_cast<osg::MatrixTransform *>(node->getParent(0));
        bool transformPresent = transformNode.valid() && (transformNode->getName() == "#_TRANSFORM_SGITEM_#");

        if (transformNeeded)
        {
            if (!transformPresent)
            {
                // insert a new MatrixTransform
                transformNode = new osg::MatrixTransform();
                transformNode->setName("#_TRANSFORM_SGITEM_#");
                while (node->getNumParents() > 0)
                {
                    osg::Group *parent = node->getParent(0);
                    parent->removeChild(node.get());
                    parent->addChild(transformNode.get());
                }
                transformNode->addChild(node.get());
            }
            transformNode->setMatrix(m);
        }
        else
        {
            if (transformPresent)
            {
                // remove existing MatrixTransform
                while (transformNode->getNumParents() > 0)
                {
                    osg::Group *parent = transformNode->getParent(0);
                    parent->removeChild(transformNode);
                    parent->addChild(node.get());
                }
                transformNode->removeChild(node.get());
            }
        }
    }
}

void
VRCoviseConnection::setColor(osg::Node *node, int *color)
{

    if (node)
    {
        osg::Geode *geode = dynamic_cast<osg::Geode *>(node);
        osg::Group *group = dynamic_cast<osg::Group *>(node);
        if (geode)
        {
            VRSceneGraph::instance()->setColor(geode, color, 1.0);
        }
        else if (group)
        {
            for (unsigned int i = 0; i < group->getNumChildren(); i++)
            {
                setColor(group->getChild(i), color);
            }
        }
    }
}

void
VRCoviseConnection::setMaterial(osg::Node *node, const int *ambient, const int *diffuse, const int *specular, float shininess, float transparency)
{
    //    fprintf(stderr, "VRCoviseConnection::setMaterial %f\n", transparency);
    if (node)
    {
        osg::Geode *geode = dynamic_cast<osg::Geode *>(node);
        osg::Group *group = dynamic_cast<osg::Group *>(node);
        if (geode)
        {
            VRSceneGraph::instance()->setMaterial(geode, ambient, diffuse, specular, shininess, transparency);
        }
        else if (group)
        {
            for (unsigned int i = 0; i < group->getNumChildren(); i++)
            {
                setMaterial(group->getChild(i), ambient, diffuse, specular, shininess, transparency);
            }
        }
    }
}

void
VRCoviseConnection::setTransparency(osg::Node *node, float transparency)
{

    if (node)
    {
        osg::Geode *geode = dynamic_cast<osg::Geode *>(node);
        osg::Group *group = dynamic_cast<osg::Group *>(node);
        if (geode)
        {
            VRSceneGraph::instance()->setTransparency(geode, transparency);
        }
        else if (group)
        {
            for (unsigned int i = 0; i < group->getNumChildren(); i++)
            {
                setTransparency(group->getChild(i), transparency);
            }
        }
    }
}

void
VRCoviseConnection::setColor(const char *objectName, int *color)
{
    //fprintf(stderr,"*****VRCoviseConnection::setColor(%s, %d %d %d)\n", objectName, color[0], color[1], color[2]);

    osg::Node *node;
    node = VRSceneGraph::instance()->findFirstNode<osg::Node>(objectName);
    if (node)
    {
        setColor(node, color);
    }
}

void
VRCoviseConnection::setMaterial(const char *objectName, const int *ambient, const int *diffuse, const int *specular, float shininess, float transparency)
{
    //fprintf(stderr,"*****VRCoviseConnection::setMaterial(%s)\n", objectName);

    osg::Node *node;
    node = VRSceneGraph::instance()->findFirstNode<osg::Node>(objectName);
    if (node)
    {
        setMaterial(node, ambient, diffuse, specular, shininess, transparency);
    }
}

void
VRCoviseConnection::setTransparency(const char *objectName, float transparency)
{
    osg::Node *node;
    node = VRSceneGraph::instance()->findFirstNode<osg::Node>(objectName);
    setTransparency(node, transparency);
}

void
VRCoviseConnection::setShader(const char *objectName, const char *shaderName)
{
    osg::Node *node = VRSceneGraph::instance()->findFirstNode<osg::Node>(objectName);
    if (!node)
    {
        return;
    }

    if (!shaderName || strcmp(shaderName, "") == 0)
    {
        coVRShaderList::instance()->remove(node);
        return;
    }

    coVRShader *shader = coVRShaderList::instance()->get(shaderName);
    if (shader)
    {
        shader->apply(node);
    }
}
