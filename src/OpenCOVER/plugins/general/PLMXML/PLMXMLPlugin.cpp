/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: PLMXML Plugin (loads PLMXML documents)                      **
**                                                                          **
**                                                                          **
** Author: U.Woessner                                                       **
**                                                                          **
** History:  		         		                            **
** Nov-01  v1	    				       		            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRTui.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRSelectionManager.h>
#include <config/CoviseConfig.h>

#include "PLMXMLPlugin.h"
#include "PLMXMLParser.h"
#include "PLMXMLSimVisitor.h"

#include <cover/RenderObject.h>
#include <cover/VRRegisterSceneGraph.h>

#include <osg/Group>
#include <osg/MatrixTransform>

using namespace osg;
#include <sys/types.h>
#include <string.h>

#include <PluginUtil/SimReference.h>

#include <PluginUtil/PluginMessageTypes.h>
#include <net/tokenbuffer.h>

PLMXMLPlugin *PLMXMLPlugin::plugin = NULL;
osg::Group *PLMXMLPlugin::currentGroup = NULL;

static FileHandler handlers[] = {
    { NULL,
      PLMXMLPlugin::loadPLMXML,
      PLMXMLPlugin::unloadPLMXML,
      "plmxml" },
    { NULL,
      PLMXMLPlugin::loadPLMXML,
      PLMXMLPlugin::unloadPLMXML,
      "xml" }
};

int PLMXMLPlugin::loadPLMXML(const char *filename, osg::Group *loadParent, const char *)
{
    if (loadParent)
        currentGroup = loadParent;
    else
        currentGroup = cover->getObjectsRoot();
    // read XML document here and add parts to currentGroup

    bool loadAll = coCoviseConfig::isOn("COVER.Plugin.PLMXML.LoadAll", true);
    bool loadSTL = coCoviseConfig::isOn("COVER.Plugin.PLMXML.LoadSTL", false);
    bool loadVRML = coCoviseConfig::isOn("COVER.Plugin.PLMXML.LoadVRML", true);
    bool undoVRMLRotate = coCoviseConfig::isOn("COVER.Plugin.PLMXML.UndoVRMLRotate", true);

    // We need a single, dedicated root node for registration in vr-prepare.
    // If no registration is done, we can skip this part.
    osg::Group *rootGroup = new osg::Group;
    rootGroup->setName("PLMXML root");
    currentGroup->addChild(rootGroup);
    currentGroup = rootGroup;
    VRRegisterSceneGraph::instance()->block();

    PLMXMLParser *parser = new PLMXMLParser();
    parser->loadAll(loadAll);
    parser->loadSTL(loadSTL);
    parser->loadVRML(loadVRML);
    parser->undoVRMLRotate(undoVRMLRotate);
    parser->parse(filename, currentGroup);

    // register nodes
    VRRegisterSceneGraph::instance()->unblock();
    VRRegisterSceneGraph::instance()->registerNode(rootGroup, filename);

    TokenBuffer tb;
    tb << "LoadFiles";
    cout << "Message generated from PLMXML-Plugin!-> create Loadfiles-Button for right mouse menue in SGBrowser" << endl;
    cover->sendMessage(PLMXMLPlugin::plugin, "SGBrowser", PluginMessageTypes::PLMXMLLoadFiles, tb.getData().length(), tb.getData().data()); //gottlieb: message to /covise/src/renderer/OpenCOVER/plugins/general/SGBrowser/SGBrowser.cpp->SGBrowser::message
    return 0;
}

int PLMXMLPlugin::unloadPLMXML(const char *filename, const char *)
{
    (void)filename;

    // TODO remove group and Transform nodes created during loading this xml-file
    return 0;
}

PLMXMLPlugin::PLMXMLPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool PLMXMLPlugin::init()
{
    fprintf(stderr, "PLMXMLPlugin::PLMXMLPlugin\n");
    if (plugin)
    {
        fprintf(stderr, "already have an instance of PLMXMLPlugin !!!\n");
        return false;
    }

    plugin = this;

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);

    return true;
}
//---------------------------------------------------------------------------------------------
void PLMXMLPlugin::addNode(osg::Node *node, RenderObject *render)
{
    if (render != NULL)
    {
        if (render->getAttribute("SIMULATION") != NULL)
        {
            std::string sim_att(render->getAttribute("SIMULATION"));
            std::string cad_att(render->getAttribute("SIMULATION"));
            sim_att.append("_Sim");
            cad_att.append("_Cad");
            string nodePath = coVRSelectionManager::generatePath(node);
            node->setUserData(new SimReference(nodePath.c_str(), sim_att.c_str()));
            PLMXMLSimVisitor visitor(this, node, cad_att.c_str());
            cover->getObjectsRoot()->traverse(visitor);
        }
        if (render->getAttribute("CAD_FILE") != NULL)
        {
            PLMXMLCadVisitor cad_visitor(this);
            cover->getObjectsRoot()->traverse(cad_visitor);
        }
    }
}
// this is called if the plugin is removed at runtime
PLMXMLPlugin::~PLMXMLPlugin()
{
    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);
}

COVERPLUGIN(PLMXMLPlugin)
