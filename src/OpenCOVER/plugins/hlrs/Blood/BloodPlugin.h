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
#include <cmath>
#include <string>

#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Label.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>

#include <osg/Group>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/AnimationPath>
#include <osg/PositionAttitudeTransform> 
#include <osg/Material>

#include <Blood.h>

using namespace opencover;

class BloodPlugin : public opencover::coVRPlugin, public ui::Owner
{
public:	
    //member variables for displaying knife
    osg::ref_ptr<osg::MatrixTransform> knifeTransform;
    osg::Matrix knifeBaseTransform;

    BloodPlugin();
    ~BloodPlugin(); 

    virtual bool init();
    virtual bool update();
    osg::Vec3 particleSlip(Droplet* p);
    
    void doAddBlood();
    osg::ref_ptr<osg::Group> bloodNode;
    std::list<Blood*> bloodJunks;
    static BloodPlugin *instance();

    // Droplet particle;
    Weapon knife;
    std::list<Droplet*> particleList;
    std::list<Droplet*> particlesOnGround;

private:
    int numParticles = 0;
    
    static BloodPlugin *inst;
    ui::Menu* bloodMenu = nullptr;
    ui::Action* addBlood = nullptr;
};
#endif
