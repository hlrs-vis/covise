/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BLOOD_PLUGIN_H
#define _BLOOD_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2018 HLRS  **
 **                                                                          **
 ** Description: TestTracker                                                 **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** August 2018 v1	    				                                     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Label.h>

#include <osg/Group>

#include <TestTracker.h>

using namespace opencover;

class TestTracker : public opencover::coVRPlugin, public ui::Owner
{
public:
    TestTracker();
    ~TestTracker(); 
    virtual bool init();
    virtual bool update();
    osg::ref_ptr<osg::Group> trackerNode;
    static TestTracker *instance();
private:
    static TestTracker *inst;
    ui::Menu* trackerMenu = nullptr;
    ui::Button *printButton = nullptr;
    bool doPrint = false;
};
#endif
