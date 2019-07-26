/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "VRRegisterSceneGraph.h"

#include <grmsg/coGRObjRegisterMsg.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPluginList.h>
#include <net/message.h>
#include <net/message_types.h>

#include <sstream>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Node>
#include <osg/Material>

using namespace std;
using namespace grmsg;
using namespace covise;

namespace opencover
{

VRRegisterSceneGraph *VRRegisterSceneGraph::s_instance = NULL;

VRRegisterSceneGraph *VRRegisterSceneGraph::instance()
{
    if (!s_instance)
        s_instance = new VRRegisterSceneGraph;
    return s_instance;
}

VRRegisterSceneGraph::VRRegisterSceneGraph()
{
    assert(!s_instance);
    registerId = 0;
    sceneGraphAppendixIdString = "SCGR";
    blocked = false;
	active = false;
}

VRRegisterSceneGraph::~VRRegisterSceneGraph()
{
    s_instance = NULL;
}

void VRRegisterSceneGraph::setRegisterStartIndex(int startIndex)
{
    if (blocked)
        return;
    registerId = startIndex;
}

void VRRegisterSceneGraph::registerNode(osg::Node *node, string parent)
{
	if (!active)
		return;
    if (blocked)
        return;
    //fprintf(stderr, "VRRegisterSceneGraph::registerNode %s\n", node->getName().c_str());
    whole_message = "SCENEGRAPH: \t";
    createRegisterMessage(node, parent);
    sendRegisterMessage();
    transparency_message = "SCENEGRAPHPARAMS: \t";
    createTransparencyMessage(node);
    sendTransparencyMessage();
}

void VRRegisterSceneGraph::unregisterNode(osg::Node *node, string parent)
{
	if (!active)
		return;
    if (blocked)
        return;
    whole_message = "SCENEGRAPH: \t";
    // for stl
    createUnregisterMessage(node, parent);
    sendUnregisterMessage();
}

void VRRegisterSceneGraph::createTransparencyMessage(osg::Node *node)
{
    //walk trough all nodes and add the ones with transparency!=1.0 to message
    if (node)
    {
        osg::Geode *geode = dynamic_cast<osg::Geode *>(node);
        osg::Group *group = dynamic_cast<osg::Group *>(node);

        // if its a geode, so it has a material and maybe is transparent
        if (geode)
        {
            float transparency = 1.0;
            osg::ref_ptr<osg::Drawable> drawable;
            osg::ref_ptr<osg::Material> mtl;
            // get transparency of geode
            if (geode->getNumDrawables() > 0)
            {
                drawable = geode->getDrawable(0);
                mtl = (osg::Material *)drawable->getOrCreateStateSet()->getAttribute(osg::StateAttribute::MATERIAL);
                if (mtl)
                {
                    osg::Vec4 color = mtl->getAmbient(osg::Material::FRONT_AND_BACK);
                    transparency = color[3];
                }
            }

            // if geode is transparent, add name of the geode and value of
            // transparency to transparency_message
            if (transparency != 1.0)
            {
                string transparencyStr;
                ostringstream transStream;
                transStream << setprecision(6) << transparency;
                transparencyStr = transStream.str();

                // separate names with tabulators
                // separate node and transparency with ;;
                // set transparency in GUI for part with the same name as geode
                string message = "";
                string name = "";
                name = geode->getName();
                if (name != " " && name != "")
                {
                    size_t found = whole_message.find(name);
                    // transparancy can only be set if the identifier (name)
                    // is in whole_message, too
                    if (found != string::npos)
                        message.append(name);
                }
                //fprintf(stderr, "setTrasn geode of %s to %f\n", name.c_str(), transparency);
                message.append(";;");
                message.append(transparencyStr);
                message.append("\t");
                transparency_message.append(message);
            }
        }
        // if node is a group, walk through all children to find
        // geodes which are transparent
        else if (group)
        {
            for (unsigned int i = 0; i < group->getNumChildren(); i++)
            {
                createTransparencyMessage(group->getChild(i));
            }
        }
    }
}

void VRRegisterSceneGraph::createRegisterMessage(osg::Node *node, string parent)
{
    if (node)
    {
        string nodeName = createName(node->getName());
        // check for vr-prepare's EndOfTree indicator
        bool doNotFollow = (nodeName.find("_EOT") != string::npos) || (nodeName.find("-EOT") != string::npos);
        // append counting register id
        stringstream idString;
        idString << nodeName;
        idString << "_" << sceneGraphAppendixIdString << "_";
        idString << registerId;
        node->setName(idString.str().c_str());
        registerId++;
        //separate names with tabulators
        //separate node, class and parent with ;;
        string message = node->getName();
        message.append(";;");
        message.append(node->className());
        message.append(";;");
        message.append(parent);
        message.append("\t");
        whole_message.append(message);
        osg::Group *group = dynamic_cast<osg::Group *>(node);
        if (group && !doNotFollow) // pfGroup, pfDCS
        {
            for (unsigned int i = 0; i < group->getNumChildren(); i++)
            {
                createRegisterMessage(group->getChild(i), node->getName());
            }
        }
    }
}

void VRRegisterSceneGraph::createUnregisterMessage(osg::Node *node, string parent)
{
    if (node)
    {
        string message = node->getName();
        // check for vr-prepare's EndOfTree indicator
        bool doNotFollow = (message.find("_EOT") != string::npos) || (message.find("-EOT") != string::npos);
        //separate names with tabulators
        //separate node and parent with ;;
        message.append(";;");
        message.append(parent);
        message.append("\t");
        whole_message.append(message);

        osg::Group *group = dynamic_cast<osg::Group *>(node);
        if (group && !doNotFollow) // pfGroup, pfDCS
        {
            for (unsigned int i = 0; i < group->getNumChildren(); i++)
            {
                createUnregisterMessage(group->getChild(i), node->getName());
            }
        }
    }
}

void VRRegisterSceneGraph::sendRegisterMessage()
{
    // only send message, if there is a controller and i am the master
    if (coVRMSController::instance()->isMaster())
    {

        coGRObjRegisterMsg regMsg(whole_message.c_str(), NULL);
        Message grmsg{COVISE_MESSAGE_UI, DataHandle((char*)(regMsg.c_str()), strlen((regMsg.c_str())) + 1, false)};
        coVRPluginList::instance()->sendVisMessage(&grmsg);
    }
}

void VRRegisterSceneGraph::sendUnregisterMessage()
{
    // only send message, if there is a controller and i am the master
    if (coVRMSController::instance()->isMaster())
    {

        coGRObjRegisterMsg regMsg(whole_message.c_str(), NULL, true);
        Message grmsg{ COVISE_MESSAGE_UI, DataHandle((char*)(regMsg.c_str()), strlen((regMsg.c_str())) + 1, false) };
        coVRPluginList::instance()->sendVisMessage(&grmsg);
    }
}

void VRRegisterSceneGraph::sendTransparencyMessage() //toGui
{
    // only send message, if there is a controller and i am the master
    if (coVRMSController::instance()->isMaster())
    {

        coGRKeyWordMsg keyWordMsg(transparency_message.c_str(), false);
        Message grmsg{ COVISE_MESSAGE_UI, DataHandle((char*)(keyWordMsg.c_str()), strlen((keyWordMsg.c_str())) + 1, false) };
        coVRPluginList::instance()->sendVisMessage(&grmsg);
    }
}

string VRRegisterSceneGraph::createName(string oldName)
{
    size_t end = oldName.find_last_of('.');
    size_t start = oldName.find_last_of('/');
    if (start != 0)
        start = start + 1;
    string newName = oldName.substr(start, end - start);
    return newName;
}
}
