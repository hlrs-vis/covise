/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: MADIconnect OpenCOVER Plugin (connects to MADI)              **
 **                                                                          **
 **                                                                          **
 ** Author: D. Wickeroth                                                     **
 **                                                                          **
 ** History:                                                                 **
 ** July 2025  v1                                                           **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "MADIconnect.h"

#include <cover/coVRConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>

#include <config/CoviseConfig.h>

#include <PluginUtil/PluginMessageTypes.h>

#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Vec4>

#include <osg/Node>
#include <osg/NodeVisitor>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <iostream>

using namespace opencover;
using namespace std;
using covise::TokenBuffer;
using covise::coCoviseConfig;

MADIconnect *MADIconnect::plugin = nullptr;

MADIconnect::MADIconnect()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("MADIconnect", cover->ui)
{
    fprintf(stderr, "MADIconnect\n");

    plugin = this;

    dataPath = coCoviseConfig::getEntry("value", "COVER.Plugin.MADIconnect.DataPath", "");
    if (!dataPath.empty())
    {
        char last = dataPath.back();
        if (last == '/' || last == '\\')
            dataPath.pop_back(); // remove trailing slash
    } 
    else
    {
#ifdef _WIN32
        dataPath = std::getenv("USERPROFILE");
        dataPath += "\\data";
#else
        dataPath = std::getenv("HOME");
        dataPath += "/data";
#endif
    }
    cout << "MADIconnect::DataPath: " << dataPath << endl;

    int port = coCoviseConfig::getInt("port", "COVER.Plugin.MADIconnect", 33333);
    toMADI = NULL;
    serverConn = NULL;
    if (coVRMSController::instance()->isMaster())
    {
        cout << "MADI::Creating server connection on port " << port << endl;

        serverConn = new ServerConnection(port, 4711, Message::UNDEFINED);
        if (!serverConn->getSocket())
        {
            cout << "MADI::tried to open server Port " << port << endl;
            cout << "MADI::Creation of server failed!" << endl;
            cout << "MADI::Port-Binding failed! Port already bound?" << endl;
            delete serverConn;
            serverConn = NULL;
        }
        else
        {
            cover->watchFileDescriptor(serverConn->getSocket()->get_id());
        }
    }

    struct linger linger;
	linger.l_onoff = 0;
	linger.l_linger = 0;
	cout << "MADI::Set socket options..." << endl;
	if (serverConn)
	{
		setsockopt(serverConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

		cout << "MADI::Set server to listen mode..." << endl;
		serverConn->listen();
		if (!serverConn->is_connected()) // could not open server port
		{
			fprintf(stderr, "MADI::Could not open server port %d\n", port);
			delete serverConn;
			serverConn = NULL;
		}
	}
    msg = new Message;

    madiMenu = new ui::Menu("MADIconnect", this);
    madiMenu->setText("MADIconnect");

    testAction = new ui::Action(madiMenu, "TestMADI");
    testAction->setText("Test MADI Connection");
    testAction->setCallback([this]() { sendTestMessage(); });
}

bool MADIconnect::destroy()
{
    std::cout << "MADIconnect::destroy called" << std::endl;    
    return true;
}

bool MADIconnect::update()
{
    //std::cout << "MADIconnect::update called" << std::endl;
    return true;
}

void MADIconnect::sendTestMessage()
{
    std::cout << "MADIconnect::sendTestMessage called" << std::endl;
    static int data = 42;

    TokenBuffer tb;
    cout << "MADIconnect::Test 1: " << tb.getData().getLength() << endl;
    tb << data++;
    cout << "MADIconnect::Test 2: " << tb.getData().getLength() << endl;
    tb << data++;
    cout << "MADIconnect::Test 3: " << tb.getData().getLength() << endl;
    tb << data;
    cout << "MADIconnect::Test 4: " << tb.getData().getLength() << endl;

    Message m(tb);
    m.type = (int)MADIconnect::MSG_VIEW_ALL;
    MADIconnect::instance()->sendMessage(m);
    data++;
}

#include <iostream>
#include <iomanip>
#define PRINT_BYTES_HEX(data, len)                                      \
    do {                                                                \
        for (size_t i = 0; i < (len); ++i) {                            \
            std::cout << std::hex << std::setw(2) << std::setfill('0')  \
                      << (static_cast<unsigned>(                        \
                              static_cast<unsigned char>((data)[i])))   \
                      << " ";                                           \
        }                                                               \
        std::cout << std::dec << std::endl;                             \
    } while (0)

void MADIconnect::preFrame()
{
    if (serverConn && serverConn->is_connected() && serverConn->check_for_input()) // we have a server and received a connect
	{
		//   std::cout << "Trying serverConn..." << std::endl;
		toMADI = serverConn->spawn_connection();
		if (toMADI && toMADI->is_connected())
		{
			fprintf(stderr, "Connected to MADI\n");
            cover->watchFileDescriptor(toMADI->getSocket()->get_id());
		}
	}
       
    //Distribute messages to slaves
    char gotMsg = '\0';
    if (coVRMSController::instance()->isMaster())
	{
		double startTime = cover->currentTime();
		while (toMADI && toMADI->check_for_input())
		{
			if (cover->currentTime() > startTime + 1)
				break;
			toMADI->recv_msg(msg);
			if (msg)
			{
                cout << "MADIconnect::Master Received Msg:\n" << msg->sender << " " << msg->type << " " << msg->send_type << " " << msg->data.length() << endl;
                PRINT_BYTES_HEX(msg->data.data(), msg->data.length());

				gotMsg = '\1';
				coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
				coVRMSController::instance()->sendSlaves(msg);                

				cover->sendMessage(this, coVRPluginSupport::TO_SAME_OTHERS, PluginMessageTypes::RRZK_rrxevent + msg->type, msg->data.length(), msg->data.data());
				handleMessage(msg);
			}
			else
			{
				gotMsg = '\0';
				cerr << "could not read message" << endl;
				break;
			}
		}
		gotMsg = '\0';
		coVRMSController::instance()->sendSlaves(&gotMsg, sizeof(char));
	}
	else
	{
		do
		{
			coVRMSController::instance()->readMaster(&gotMsg, sizeof(char));
			if (gotMsg != '\0')
			{
				coVRMSController::instance()->readMaster(msg);

                cout << "MADIconnect::Client Received Msg:\n" << msg->sender << " " << msg->type << " " << msg->send_type << " " << msg->data.length() << endl;
                PRINT_BYTES_HEX(msg->data.data(), msg->data.length());

				handleMessage(msg);
			}
		} while (gotMsg != '\0');
	}    
}

bool MADIconnect::sendMessage(Message &m)
{
	if (toMADI) // false on slaves
	{
		return toMADI->sendMessage(&m);
	}
    return false;
}

class ColorChangerVisitor : public osg::NodeVisitor {
public:
    ColorChangerVisitor(const std::string& targetName, const osg::Vec4& color)
        : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN), _targetName(targetName), _color(color) {}

    virtual void apply(osg::Group& node) override {
        if (node.getName() == _targetName && node.getNumChildren() > 0) {
            osg::ref_ptr<osg::Node> child = node.getChild(0);
            osg::Geode* geode = dynamic_cast<osg::Geode*>(child.get());
            if (geode) {
                std::cout << "Found geode under node: " << _targetName << std::endl;
                setColor(geode);
            }
        }
        traverse(node);
    }

private:
    void setColor(osg::Geode* geode) {
        for (unsigned int i = 0; i < geode->getNumDrawables(); ++i) {
            osg::Geometry* geometry = dynamic_cast<osg::Geometry*>(geode->getDrawable(i));
            if (geometry) {
                osg::ref_ptr<osg::Material> material = new osg::Material;
                material->setDiffuse(osg::Material::FRONT_AND_BACK, _color);
                geode->getOrCreateStateSet()->setAttributeAndModes(material, osg::StateAttribute::ON);
            }
        }
    }

    std::string _targetName;
    osg::Vec4 _color;
};


void MADIconnect::handleMessage(Message *m)
{
    if (!m)
    {
        cerr << "MADIconnect::handleMessage: received NULL message" << endl;
        return;
    }

    enum MessageTypes type = (enum MessageTypes)m->type;

    switch (type)
    {
        case MSG_LOAD_NEURONS:
        {
            TokenBuffer tb(m);
            int numNeurons = 0;
            tb >> numNeurons;

            for(int i = 0; i < numNeurons; ++i){
                string filename;
                tb >> filename;
                string fullPath = dataPath + "/" + filename;
                coVRFileManager::instance()->loadFile(fullPath.c_str(), NULL, cover->getObjectsRoot(), "MADI");
            }            
            break;
        }

        case MSG_SHOW_NEURONS:
        {
            TokenBuffer tb(m);
            int numNeurons = 0;
            tb >> numNeurons;
            for(int i = 0; i < numNeurons; ++i){
                string neuronName;
                tb >> neuronName;
                string fullPath = dataPath + "/" + neuronName;
                osg::Node *node;
                node = VRSceneGraph::instance()->findFirstNode<osg::Node>(fullPath.c_str());
                node->setNodeMask(node->getNodeMask() | (Isect::Visible));

                osg::Group *group = dynamic_cast<osg::Group*>(node);
                if (!group)
                {
                    cerr << "MADIconnect::hideNeurons: Node is not a group: " << fullPath << endl;
                    continue;
                }

                if (group)
                {

                    osg::ref_ptr<osg::Node> child = group->getChild(0);
                    osg::Geode* geode = dynamic_cast<osg::Geode*>(child.get());
                    if (geode) {
                        std::cout << "Found geode under node: " << fullPath << std::endl;
                        geode->setNodeMask(geode->getNodeMask() | (Isect::Visible));
                    }
                    else {
                        std::cout << "No geode found under node: " << fullPath << std::endl;
                    }                    
                }
            }
            break;
        }

        case MSG_HIDE_NEURONS:
        {
            TokenBuffer tb(m);
            int numNeurons = 0;
            tb >> numNeurons;
            for(int i = 0; i < numNeurons; ++i){
                string neuronName;
                tb >> neuronName;
                string fullPath = dataPath + "/" + neuronName;
                osg::Node *node;
                node = VRSceneGraph::instance()->findFirstNode<osg::Node>(fullPath.c_str());
                node->setNodeMask(node->getNodeMask() & (~(Isect::Visible | Isect::OsgEarthSecondary)));
                
                osg::Group *group = dynamic_cast<osg::Group*>(node);
                if (!group)
                {
                    cerr << "MADIconnect::hideNeurons: Node is not a group: " << fullPath << endl;
                    continue;
                }
                
                if (group)
                {

                    osg::ref_ptr<osg::Node> child = group->getChild(0);
                    osg::Geode* geode = dynamic_cast<osg::Geode*>(child.get());
                    if (geode) {
                        std::cout << "Found geode under node: " << fullPath << std::endl;
                        geode->setNodeMask(geode->getNodeMask() & (~(Isect::Visible | Isect::OsgEarthSecondary)));
                    }
                    else {
                        std::cout << "No geode found under node: " << fullPath << std::endl;
                    }                    
                }
            }
            break;
        }

        case MSG_COLOR_NEURONS:
        {
            std::cout << "MADIconnect::MSG_COLOR_NEURONS" << std::endl;
            TokenBuffer tb(m);

            int intColor = 0;
            tb >> intColor;
            std::cout << "MADIconnect::Color: " << intColor << std::endl;

            //TODO: Use VRSceneGraph::instance()->setColor(geode, color, 1.0f);
            //VRSceneGraph::instance()->setColor(geode, color, 1.0);

            float r = ((intColor >> 16) & 0xFF) / 255.0f;
            float g = ((intColor >> 8)  & 0xFF) / 255.0f;
            float b = ( intColor        & 0xFF) / 255.0f;
            
            osg::Vec4 color(r,g,b,1.0f);            
            std::cout << "MADIconnect::Color RGBA: " << color.r() << ", " << color.g() << ", " << color.b() << ", " << color.a() << std::endl;
                
            // Apply color to all neurons
            int numNeurons = 0;
            tb >> numNeurons;
            for(int i = 0; i < numNeurons; ++i){
                string neuronName;
                tb >> neuronName;
                string fullPath = dataPath + "/" + neuronName;                
                ColorChangerVisitor colorChanger(fullPath, color);
                cover->getObjectsRoot()->accept(colorChanger);
            }
            break;
        }

        case MSG_LOAD_VOLUME:
        case MSG_SHOW_VOLUME:
        case MSG_HIDE_VOLUME:
            // Handle other message types as needed
            cout << "MADIconnect::Received message type: " << type << endl;
            break;

        case MSG_VIEW_ALL:
        {
            VRSceneGraph::instance()->viewAll();
            break;
        }

        case MSG_TEST:
        {
            cout << "MADIconnect::Received MSG_TEST" << endl;
            TokenBuffer tb(m);
            cout << "MADIconnect::TB: " << msg->sender << " " << msg->type << " " << msg->send_type << " " << msg->data.length() << endl;
            PRINT_BYTES_HEX(tb.getData().data(), tb.getData().length());                        
            int test;
            tb >> test;
            cout << "MADI::Received test value: " << test << endl;
            break;
        }       
        
        default:
            cerr << "MADIconnect::Unknown message type: " << type << endl;
            // Handle unknown message type
            // You can add your own logic here
            // For now, we just log it
        break;
    }         
}

// this is called if the plugin is removed at runtime
MADIconnect::~MADIconnect()
{
    fprintf(stderr, "Goodbye MADI\n");

    if (serverConn && serverConn->getSocket())
        cover->unwatchFileDescriptor(serverConn->getSocket()->get_id());
	delete serverConn;
	serverConn = NULL;

    if (toMADI && toMADI->getSocket())
        cover->unwatchFileDescriptor(toMADI->getSocket()->get_id());

    delete msg;
    toMADI = NULL;

    if (plugin == this)
        plugin = nullptr;
}

COVERPLUGIN(MADIconnect)
