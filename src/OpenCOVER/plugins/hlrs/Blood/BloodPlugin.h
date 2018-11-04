/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BLOOD_PLUGIN_H
#define _BLOOD_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2018 HLRS  **
 **                                                                          **
 ** Description: BloodPlugin                                                 **
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

#include <Blood.h>

using namespace opencover;

class BloodPlugin : public opencover::coVRPlugin, public ui::Owner
{
public:
    BloodPlugin();
    ~BloodPlugin(); 
    virtual bool init();
    virtual bool update();
    void doAddBlood();
    osg::ref_ptr<osg::Group> bloodNode;
    std::list<Blood *> bloodJunks;
    static BloodPlugin *instance();
private:
    static BloodPlugin *inst;
    ui::Menu* bloodMenu = nullptr;
    ui::Action* addBlood = nullptr;
};
#endif
