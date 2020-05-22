/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/SimReference.h>
#include "PLMXMLSimVisitor.h"
#include <cover/coVRSelectionManager.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <net/tokenbuffer.h>

using namespace std;
using namespace opencover;

PLMXMLSimVisitor::PLMXMLSimVisitor(coVRPlugin *plug, osg::Node *node, const char *targetNode)
    : osg::NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
    , plugin(plug)
    , simNode(node)
    , target(targetNode)
{
}
PLMXMLSimVisitor::~PLMXMLSimVisitor()
{
}

void PLMXMLSimVisitor::apply(osg::Group &node)
{
    osg::Referenced *data = node.getUserData();
    SimReference *simRef;

    if ((simRef = dynamic_cast<SimReference *>(data)))
    {
        vector<string> mySimVec = simRef->getSimPath();
        vector<string> mySimNameVec = simRef->getSimName();

        vector<string>::iterator nameIter = mySimNameVec.begin();

        for (vector<string>::iterator i = mySimVec.begin(); i != mySimVec.end(); i++)
        {
            if (*i == target)
            {
                cout << endl << "###############################################################################" << endl;
                std::cout << "until traversing the Scengraph a simulation for Knoten:" << node.getName() << " was found" << std::endl;
                std::cout << "Userdata: " << *i << std::endl;
                cout << "###############################################################################" << endl << endl;
                CAD_SIM_Node[&node] = simNode;
                string nodePath = coVRSelectionManager::generatePath(&node);
                string simPath = coVRSelectionManager::generatePath(simNode);
                //cout<<"Node Path : "<<nodePath<<endl;
                //cout<<"Sim Path  : "<<simPath<<endl;
                covise::TokenBuffer tb;
                tb << nodePath;
                tb << simNode->getName();
                tb << *nameIter;
                cover->sendMessage(plugin, "SGBrowser", PluginMessageTypes::PLMXMLSetSimPair, tb.getData().length(), tb.getData().data()); //gottlieb: message to /covise/src/renderer/OpenCOVER/plugins/general/SGBrowser/SGBrowser.cpp->SGBrowser::message
                // cover->sendMessage ( plugin,"SGBrowser", 12, tb.get_length(),tb.get_data() );
            }
            nameIter++;
        }
    }
    traverse(node);
}
//-----------------------------------------------------------------------------------------------

PLMXMLCadVisitor::PLMXMLCadVisitor(coVRPlugin *plug)
    : osg::NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
    , plugin(plug)
{
}
void PLMXMLCadVisitor::apply(osg::Group &node)
{
    osg::Referenced *data = node.getUserData();
    SimReference *simRef;

    if ((simRef = dynamic_cast<SimReference *>(data)))
    {
        vector<string> simName = simRef->getSimName();
        vector<string> simPath = simRef->getSimPath();
        vector<string>::iterator Iter = simName.begin();
        if ((*Iter).substr((*Iter).find_last_of("_"), (*Iter).size()) == "_Sim")
        {
            string tmpstr((*Iter).substr(0, (*Iter).find_last_of("_")));
            tmpstr.append("_Cad");
            PLMXMLSimVisitor visitor(plugin, &node, tmpstr.c_str());
            cover->getObjectsRoot()->traverse(visitor);
        }
    }
    traverse(node);
}
