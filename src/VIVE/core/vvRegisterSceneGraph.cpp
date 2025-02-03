/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "vvRegisterSceneGraph.h"

#include <grmsg/coGRObjRegisterMsg.h>
#include <grmsg/coGRKeyWordMsg.h>
#include "vvMSController.h"
#include "vvVIVE.h"
#include "vvPluginSupport.h"
#include "vvPluginList.h"
#include <net/message.h>
#include <net/message_types.h>

#include <sstream>
#include <vsg/all.h>

using namespace std;
using namespace grmsg;
using namespace covise;

namespace vive
{

    vvRegisterSceneGraph* vvRegisterSceneGraph::s_instance = NULL;

    vvRegisterSceneGraph* vvRegisterSceneGraph::instance()
    {
        if (!s_instance)
            s_instance = new vvRegisterSceneGraph;
        return s_instance;
    }

    vvRegisterSceneGraph::vvRegisterSceneGraph()
    {
        assert(!s_instance);
        registerId = 0;
        sceneGraphAppendixIdString = "SCGR";
        blocked = false;
        active = false;
    }

    vvRegisterSceneGraph::~vvRegisterSceneGraph()
    {
        s_instance = NULL;
    }

    void vvRegisterSceneGraph::setRegisterStartIndex(int startIndex)
    {
        if (blocked)
            return;
        registerId = startIndex;
    }

    void vvRegisterSceneGraph::registerNode(vsg::Node* node, string parent)
    {
        if (!active)
            return;
        if (blocked)
            return;
        //fprintf(stderr, "vvRegisterSceneGraph::registerNode %s\n", node->getName().c_str());
        whole_message = "SCENEGRAPH: \t";
        createRegisterMessage(node, parent);
        sendRegisterMessage();
        transparency_message = "SCENEGRAPHPARAMS: \t";
        createTransparencyMessage(node);
        sendTransparencyMessage();
    }

    void vvRegisterSceneGraph::unregisterNode(vsg::Node* node, string parent)
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

    void vvRegisterSceneGraph::createTransparencyMessage(vsg::Node* node)
    {
        //walk trough all nodes and add the ones with transparency!=1.0 to message
        if (node)
        {
            vsg::Group* group = dynamic_cast<vsg::Group*>(node);

         /*   // if its a geode, so it has a material and maybe is transparent
            if (geode)
            {
                float transparency = 1.0;
                vsg::ref_ptr<vsg::Drawable> drawable;
                vsg::ref_ptr<vsg::Material> mtl;
                // get transparency of geode
                if (geode->getNumDrawables() > 0)
                {
                    drawable = geode->getDrawable(0);
                    mtl = (vsg::Material*)drawable->getOrCreateStateSet()->getAttribute(vsg::StateAttribute::MATERIAL);
                    if (mtl)
                    {
                        vsg::Vec4 color = mtl->getAmbient(vsg::Material::FRONT_AND_BACK);
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
            else */
            if (group)
            {
                for( const auto &child : group->children)
                {
                    createTransparencyMessage(child);
                }
            }
        }
    }

    void vvRegisterSceneGraph::createRegisterMessage(vsg::Node* node, string parent)
    {
        if (node)
        {
            std::string name;
            if (node->getValue("name", name))
            {
                string nodeName = createName(name);
                // check for vr-prepare's EndOfTree indicator
                bool doNotFollow = (nodeName.find("_EOT") != string::npos) || (nodeName.find("-EOT") != string::npos);
                // append counting register id
                stringstream idString;
                idString << nodeName;
                idString << "_" << sceneGraphAppendixIdString << "_";
                idString << registerId;
                node->setValue("name", idString.str());
                registerId++;
                //separate names with tabulators
                //separate node, class and parent with ;;
                string message = idString.str();
                message.append(";;");
                message.append(node->className());
                message.append(";;");
                message.append(parent);
                message.append("\t");
                whole_message.append(message);
                vsg::Group* group = dynamic_cast<vsg::Group*>(node);
                if (group && !doNotFollow) // pfGroup, pfDCS
                {
                    for (const auto& child : group->children)
                    {
                        createRegisterMessage(child, idString.str());
                    }
                }
            }
        }
    }

    void vvRegisterSceneGraph::createUnregisterMessage(vsg::Node* node, string parent)
    {
        if (node)
        {
            std::string name;
            if (node->getValue("name", name))
            {
                string message = name;
                // check for vr-prepare's EndOfTree indicator
                bool doNotFollow = (message.find("_EOT") != string::npos) || (message.find("-EOT") != string::npos);
                //separate names with tabulators
                //separate node and parent with ;;
                message.append(";;");
                message.append(parent);
                message.append("\t");
                whole_message.append(message);

                vsg::Group* group = dynamic_cast<vsg::Group*>(node);
                if (group && !doNotFollow) // pfGroup, pfDCS
                {
                    for (const auto& child : group->children)
                    {
                        createUnregisterMessage(child, name);
                    }
                }
            }
        }
    }

    void vvRegisterSceneGraph::sendRegisterMessage()
    {
        // only send message, if there is a controller and i am the master
        if (vvMSController::instance()->isMaster())
        {

            coGRObjRegisterMsg regMsg(whole_message.c_str(), NULL);
            vv->sendGrMessage(regMsg);
        }
    }

    void vvRegisterSceneGraph::sendUnregisterMessage()
    {
        // only send message, if there is a controller and i am the master
        if (vvMSController::instance()->isMaster())
        {

            coGRObjRegisterMsg regMsg(whole_message.c_str(), NULL, true);
            vv->sendGrMessage(regMsg);
        }
    }

    void vvRegisterSceneGraph::sendTransparencyMessage() //toGui
    {
        // only send message, if there is a controller and i am the master
        if (vvMSController::instance()->isMaster())
        {

            coGRKeyWordMsg keyWordMsg(transparency_message.c_str(), false);
            vv->sendGrMessage(keyWordMsg);
        }
    }

    string vvRegisterSceneGraph::createName(string oldName)
    {
        size_t end = oldName.find_last_of('.');
        size_t start = oldName.find_last_of('/');
        if (start != 0)
            start = start + 1;
        string newName = oldName.substr(start, end - start);
        return newName;
    }
}
