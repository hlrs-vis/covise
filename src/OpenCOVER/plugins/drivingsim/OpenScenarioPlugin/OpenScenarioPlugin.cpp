/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/


#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include "OpenScenarioPlugin.h"

#include <DrivingSim/OpenScenario/OpenScenarioBase.h>
#include <DrivingSim/OpenScenario/oscFileHeader.h>
#include <DrivingSim/oddlot/src/io/oscparser.hpp>
#include <DrivingSim/oddlot/src/io/domparser.hpp>
#include <DrivingSim/oddlot/src/io/domwriter.hpp>

using namespace OpenScenario; 
using namespace opencover;

OpenScenarioPlugin *OpenScenarioPlugin::plugin = NULL;

static FileHandler handlers[] = {
    { NULL,
      OpenScenarioPlugin::loadOSC,
      OpenScenarioPlugin::loadOSC,
      NULL,
      "xosc" }
};

OpenScenarioPlugin::OpenScenarioPlugin()
{
	plugin = this;
	osdb = new OpenScenario::OpenScenarioBase();
    fprintf(stderr, "OpenScenario::OpenScenario\n");
}

// this is called if the plugin is removed at runtime
OpenScenarioPlugin::~OpenScenarioPlugin()
{
    fprintf(stderr, "OpenScenarioPlugin::~OpenScenarioPlugin\n");
}


void OpenScenarioPlugin::preFrame()
{
}

COVERPLUGIN(OpenScenarioPlugin)

bool OpenScenarioPlugin::init()
{
    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    //coVRFileManager::instance()->registerFileHandler(&handlers[1]);
	return true;
}

int OpenScenarioPlugin::loadOSC(const char *filename, osg::Group *g, const char *key)
{
	plugin->loadOSCFile(filename,g,key);
}

int OpenScenarioPlugin::loadOSCFile(const char *filename, osg::Group *, const char *key)
{
	if(osdb->loadFile(filename, "OpenSCENARIO", "OpenSCENARIO") == false)
    {
        std::cerr << std::endl;
        std::cerr << "failed to load OpenSCENARIO from file " << filename << std::endl;
        std::cerr << std::endl;
        delete osdb;
        return -1;
    }
		/*
		OpenScenario::OpenScenarioBase *openScenarioBase = projectData_->getOSCBase()->getOpenScenarioBase();
		if (openScenarioBase)
	    {
		OSCParser *oscParser = new OSCParser(openScenarioBase, projectData_);
	
		if (plugin->parseXOSC(filename))
		return 1;
		 else
		return 0;
		}

		if (openScenarioBase_->loadFile(filename.toStdString(), nodeName.toStdString(), fileType.toStdString()) == false)
		{
        qDebug() << "failed to load OpenScenarioBase from file " << filename;
        return false;
		}
		*/
}