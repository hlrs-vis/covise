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
#include <cover/coVRFileManager.h>
#include "cover/input/input.h"
using namespace opencover;
TestTracker::TestTracker() 
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("TestTracker", cover->ui)
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
    tbVive = Input::instance()->getBody("ViveHand");
    tbTracker = Input::instance()->getBody("ViveTracker");
    buttonsVive = Input::instance()->getButtons("ViveRight");
    if (tbVive == nullptr)
    {
        fprintf(stderr, "please configure ViveHand");
        return false;
    }
    if (tbTracker == nullptr)
    {
        fprintf(stderr, "please configure ViveTracker");
        return false;
    }
    if (buttonsVive == nullptr)
    {
        fprintf(stderr, "please configure ViveRight");
        return false;
    }
    vt = new osg::MatrixTransform();
    vt->setName("ViveCoordinateSystem");
    at = new osg::MatrixTransform();
    at->setName("ARTCoordinateSystem");
    vt->addChild(coVRFileManager::instance()->loadIcon("Axis"));
    at->addChild(coVRFileManager::instance()->loadIcon("Axis"));
    trackerNode->addChild(vt);
    trackerNode->addChild(at);
    return true;
}

bool TestTracker::update()
{
    osg::Vec3 vivePos = tbVive->getMat().getTrans();
    osg::Vec3 artPos = tbTracker->getMat().getTrans();
    vt->setMatrix(tbVive->getMat());
    at->setMatrix(tbTracker->getMat());
    if (doPrint || buttonsVive->getButtonState()!=0)
    {
        fprintf(stderr, "Vive: %f %f %f  Tracker:%f %f %f  Diff:%f %f %f\n", vivePos[0], vivePos[1], vivePos[2], artPos[0], artPos[1], artPos[2], vivePos[0]- artPos[0], vivePos[1]- artPos[1], vivePos[2] - artPos[2]);
    }
    return true;
}

TestTracker * TestTracker::instance()
{
    return inst;
}

COVERPLUGIN(TestTracker)
