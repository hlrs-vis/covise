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
#include <cover/ui/Owner.h>

#include <osg/Geode>

#include <Blood.h>

namespace opencover {
namespace ui {
class Menu;
class Action;
}
}

using namespace opencover;

class BloodPlugin : public opencover::coVRPlugin, public ui::Owner
{
public:	
	
    Weapon knife;
    std::list<Droplet*> particleList;
    std::list<Droplet*> particlesOnGround;
    
    //member variables for displaying knife
    osg::ref_ptr<osg::MatrixTransform> knifeTransform;
    osg::Matrix knifeBaseTransform;

    BloodPlugin();
    ~BloodPlugin(); 

    virtual bool init();
    virtual bool update();
    void doAddBlood();
    osg::Vec3 particleSlip(Droplet* p);
    
    osg::ref_ptr<osg::Geode> bloodGeode;
    std::list<Blood*> bloodJunks;
    static BloodPlugin *instance();

    osg::Matrix hand;
    osg::Matrix handInObjectsRoot;

private:
    int numParticles = 0;
    
    static BloodPlugin *inst;
    ui::Menu* bloodMenu = nullptr;
    ui::Action* addBlood = nullptr;
};
#endif
