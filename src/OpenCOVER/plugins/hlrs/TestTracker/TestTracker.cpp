/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: TestTracker OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "TestTracker.h"
#include <cover/coVRPluginSupport.h>
#include "cover/input/input.h"
using namespace opencover;
TestTracker::TestTracker() : ui::Owner("TestTracker", cover->ui)
{
    fprintf(stderr, "TestTracker\n");
}

TestTracker *TestTracker::inst=nullptr;

// this is called if the plugin is removed at runtime
TestTracker::~TestTracker()
{
    cover->getObjectsRoot()->removeChild(trackerNode.get());
}

bool TestTracker::init()
{
    trackerMenu = new ui::Menu("TestTracker", this);
    trackerMenu->setText("TestTracker");

    printButton = new ui::Button(trackerMenu, "print");
    printButton->setText("Print");
    printButton->setState(false);
    printButton->setCallback([this](bool state) {doPrint=state; });
    trackerNode = new osg::Group();
    trackerNode->setName("TestTracker");
    cover->getObjectsRoot()->addChild(trackerNode.get());
    TrackingBody* tbVive = Input::instance()->getBody("ViveHand");
    TrackingBody* tbART = Input::instance()->getBody("ArtHand");
    ButtonDevice* buttonsVive = Input::instance()->getButtons("ViveRight");
    if (tbVive == nullptr)
    {
        fprintf(stderr, "please configure ViveHand");
        return false;
    }
    if (tbART == nullptr)
    {
        fprintf(stderr, "please configure ArtHand");
        return false;
    }
    if (buttonsVive == nullptr)
    {
        fprintf(stderr, "please configure ViveRight");
        return false;
    }
    return true;
}

bool TestTracker::update()
{
    if (doPrint)
    {
    }
    return true;
}

TestTracker * TestTracker::instance()
{
    return inst;
}

COVERPLUGIN(TestTracker)
