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
#include <osg/Switch>

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
                //cout << "MADIconnect::Master Received Msg:\n" << msg->sender << " " << msg->type << " " << msg->send_type << " " << msg->data.length() << endl;
                //PRINT_BYTES_HEX(msg->data.data(), msg->data.length());

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

                //cout << "MADIconnect::Client Received Msg:\n" << msg->sender << " " << msg->type << " " << msg->send_type << " " << msg->data.length() << endl;
                //PRINT_BYTES_HEX(msg->data.data(), msg->data.length());

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

class FindNodeAndFirstGeodeVisitor : public osg::NodeVisitor
{
public:
    explicit FindNodeAndFirstGeodeVisitor(const std::string& targetNameSubstring)
        : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN),
          _targetNameSubstring(targetNameSubstring),
          _foundTargetNode(nullptr), _foundGeode(nullptr)
    {
    }

    void apply(osg::Node& node) override
    {
        if (!_foundTargetNode && node.getName().find(_targetNameSubstring) != std::string::npos)
        {
            _foundTargetNode = &node;

            // Start second search from this node
            GeodeFinder geodeFinder;
            node.accept(geodeFinder);
            _foundGeode = geodeFinder.getFoundGeode();

            return; // Stop after finding the first match
        }

        if (!_foundTargetNode)
            traverse(node);
    }

    osg::Node* getFoundNode() const { return _foundTargetNode.get(); }
    osg::Geode* getFoundGeode() const { return _foundGeode.get(); }

private:
    class GeodeFinder : public osg::NodeVisitor
    {
    public:
        GeodeFinder() : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN), _foundGeode(nullptr) {}

        void apply(osg::Geode& geode) override
        {
            if (!_foundGeode)
                _foundGeode = &geode;
        }

        void apply(osg::Node& node) override
        {
            if (!_foundGeode)
                traverse(node);
        }

        osg::Geode* getFoundGeode() const { return _foundGeode.get(); }

    private:
        osg::ref_ptr<osg::Geode> _foundGeode;
    };

    std::string _targetNameSubstring;
    osg::ref_ptr<osg::Node> _foundTargetNode;
    osg::ref_ptr<osg::Geode> _foundGeode;
};

class SceneGraphPrinter : public osg::NodeVisitor
{
public:
    SceneGraphPrinter()
        : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN), _indentLevel(0)
    {
    }

    virtual void apply(osg::Node& node) override
    {
        printNodeInfo(node);
        _indentLevel++;
        traverse(node);
        _indentLevel--;
    }

    virtual void apply(osg::Group& group) override
    {
        printNodeInfo(group);
        _indentLevel++;
        traverse(group);
        _indentLevel--;
    }

private:
    int _indentLevel;

    void printNodeInfo(osg::Node& node)
    {
        std::string indent(_indentLevel * 2, ' ');
        std::string name = node.getName().empty() ? "<unnamed>" : node.getName();
        std::cout << indent << "- " << name
                  << " [Type: " << node.className()
                  << ", Children: " << (node.asGroup() ? node.asGroup()->getNumChildren() : 0)
                  << "]" << std::endl;
    }
};

class SwitchNodeToggler : public osg::NodeVisitor
{
public:
    SwitchNodeToggler(const std::string& targetName, bool show)
        : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN),
          _targetName(targetName),
          _show(show),
          _found(false)
    {
    }

    virtual void apply(osg::Node& node) override
    {
        if (!_found && node.getName() == _targetName)
        {
            osg::Switch* sw = dynamic_cast<osg::Switch*>(&node);
            if (sw)
            {
                std::cout << "Found switch node: " << _targetName << std::endl;
                toggleSwitch(sw);
                _found = true;
                return; // Stop traversing if found
            }
        }

        traverse(node);
    }

    bool wasFound() const { return _found; }

private:
    std::string _targetName;
    bool _show;
    bool _found;

    void toggleSwitch(osg::Switch* sw)
    {
        if (_show)
        {
            sw->setAllChildrenOff();
            for (unsigned int i = 0; i < sw->getNumChildren(); ++i)
            {
                // Show the first child (or all, if you prefer)
                sw->setChildValue(sw->getChild(i), true);
                break;
            }
        }
        else
        {
            sw->setAllChildrenOff(); // Hide all children
        }
    }
};

bool MADIconnect::showNeuron(const std::string &neuronName, bool show)
{
    auto it = loadedNeurons.find(neuronName);
    if (it != loadedNeurons.end())
    {
        //cout << "MADIconnect:: Found neuron: " << neuronName << endl;
        osg::Switch* sw = it->second.get();
        if (sw)
        {
            sw->setAllChildrenOff();
            if (show)
            {
                for (unsigned int i = 0; i < sw->getNumChildren(); ++i)
                {
                    sw->setChildValue(sw->getChild(i), true);
                }
            }
        }
    }
    else
    {
        // Fallback to NodeVisitor if not found
        string switchName = "Switch_" + neuronName;
        SwitchNodeToggler toggler(switchName, show);
        cover->getObjectsRoot()->accept(toggler);
    }
    return true;
}

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

                //1. Get the filename from the message
                string filename;
                tb >> filename;
                string fullPath = dataPath + "/" + filename;

                // 2. Create the osg::Switch node
                osg::ref_ptr<osg::Switch> switchNode = new osg::Switch();
                string switchName = "Switch_" + filename;
                switchNode->setName(switchName);

                // 3. Add it to the root of the scene graph
                osg::Group* root = cover->getObjectsRoot();  // or cover->getScene() depending on your context
                if (root)
                {
                    root->addChild(switchNode.get());
                }
                else
                {
                    std::cerr << "Error: Could not access root scene graph node.\n";
                    return;
                }

                // 3. Load the file into the switch node               
                osg::Node* loadedNode = coVRFileManager::instance()->loadFile(
                    fullPath.c_str(),  // fileName
                    nullptr,           // coTUIFileBrowserButton* fb (can be nullptr)
                    switchNode.get(),  // parent
                    nullptr            // covise_key (can be nullptr)
                );

                if (!loadedNode)
                {
                    std::cerr << "Error: Failed to load file into switch node.\n";
                }
                else
                {
                    //Enable the loaded child
                    switchNode->setAllChildrenOff();  // Turn off all children first
                    switchNode->setChildValue(loadedNode, true);  // Show only this loaded child

                    loadedNeurons[filename] = switchNode; // Store the switch node for later reference
                    std::cout << "MADIconnect::Loaded neuron: " << filename << " into switch: " << switchName << std::endl;
                }                
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
                showNeuron(neuronName, true);
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
                showNeuron(neuronName, false);
            }
            break;
        }

        case MSG_COLOR_NEURONS:
        {
            std::cout << "MADIconnect::MSG_COLOR_NEURONS" << std::endl;
            TokenBuffer tb(m);

            int rgb[3];
            tb >> rgb[0] >> rgb[1] >> rgb[2];
            
            float transparency;
            tb >> transparency;

            std::cout << "MADIconnect::Color RGBA: " << rgb[0] << ", " << rgb[1] << ", " << rgb[2] << ", " << transparency << std::endl;

            // Apply color to all neurons
            int numNeurons = 0;
            tb >> numNeurons;
            for (int i = 0; i < numNeurons; ++i)
            {
                std::string neuronName;
                tb >> neuronName;

                auto it = loadedNeurons.find(neuronName);
                bool found = false;
                
                if (it != loadedNeurons.end())
                {
                    cout << "MADIconnect:: Found neuron: " << neuronName << endl;
                    osg::Switch* sw = it->second.get();
                    if (sw && sw->getNumChildren() > 0)
                    {
                        cout << "MADIconnect:: Found children in switch: " << neuronName << endl;
                        osg::Group* group = dynamic_cast<osg::Group*>(sw->getChild(0));
                        if (group)
                        {
                            cout << "MADIconnect:: Found group." << endl;
                            osg::Geode* geode = dynamic_cast<osg::Geode*>(group->getChild(0));
                            if (geode)
                            {
                                cout << "MADIconnect:: Found geode." << endl;
                                VRSceneGraph::instance()->setColor(geode, rgb, 1.0f);
                                found = true;
                            }
                            else
                            {
                                cout << "MADIconnect:: No geode found in group: " << neuronName << endl;
                            }
                        }
                        else
                        {
                            cout << "MADIconnect:: No group found in switch: " << neuronName << endl;
                        }
                    }
                    else
                    {
                        cout << "MADIconnect:: No children found in switch: " << neuronName << endl;
                    }
                }

                if (!found)
                {
                    // Fallback to node visitor
                    FindNodeAndFirstGeodeVisitor finder(neuronName);
                    cover->getObjectsRoot()->accept(finder);
                    osg::Geode* geode = finder.getFoundGeode();
                    if (geode)
                    {
                        std::cout << "MADIconnect:: Found geode for neuron: " << neuronName << std::endl;
                        VRSceneGraph::instance()->setColor(geode, rgb, 1.0f);
                    }
                    else
                    {
                        std::cout << "MADIconnect:: No geode found for neuron: " << neuronName << std::endl;
                    }
                }
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

            SceneGraphPrinter sceneGraphPrinter;
            cover->getObjectsRoot()->accept(sceneGraphPrinter);
            break;
        }       
        
        default:
            cerr << "MADIconnect::Unknown message type: " << type << endl;
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
