/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: BloodPlugin OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "BloodPlugin.h"
#include <cover/coVRPluginSupport.h>
using namespace opencover;
BloodPlugin::BloodPlugin() : ui::Owner("Blood", cover->ui)
{
    fprintf(stderr, "BloodPlugin\n");
}

BloodPlugin *BloodPlugin::inst=nullptr;

// this is called if the plugin is removed at runtime
BloodPlugin::~BloodPlugin()
{
    cover->getObjectsRoot()->removeChild(bloodNode.get());
    for (auto junk = bloodJunks.begin(); junk != bloodJunks.end(); junk++)
    {
        delete *junk;
    }
    bloodJunks.clear();
}

bool BloodPlugin::init()
{
    bloodMenu = new ui::Menu("Blood", this);
    bloodMenu->setText("Blood");

    addBlood = new ui::Action(bloodMenu, "addBlood");
    addBlood->setText("Add Blood");
    addBlood->setCallback([this]() {doAddBlood(); });
    bloodNode = new osg::Group();
    bloodNode->setName("Blood");
    cover->getObjectsRoot()->addChild(bloodNode.get());
        
    return true;
}

bool BloodPlugin::update()
{
    return true;
}

void BloodPlugin::doAddBlood()
{
    //create a bunch of blood on the object
    bloodJunks.push_back(new Blood());
}

BloodPlugin * BloodPlugin::instance()
{
    return inst;
}

COVERPLUGIN(BloodPlugin)
