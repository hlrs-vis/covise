/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BLOOD_H
#define BLOOD_H
/****************************************************************************\ 
 **                                                            (C)2018 HLRS  **
 **                                                                          **
 ** Description: some blood drops                                            **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** August 2018  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <PluginUtil/coSphere.h>
#include <vector>

class Drop
{
public:
    osg::Vec3 pos;
    osg::Vec3 v;
    float mass;
    float d;
    double getCW() { return 0.15; };
    bool sticking=true;
};

class Blood 
{
    const double densityOfAir = 1.18;
    const double densityOfBlood = 1055.0;
    const double friction = 0.1;
public:
    Blood();
    ~Blood();
    void integrate(float dt, osg::Vec3 vObj);
    std::vector<Drop> drops;
    opencover::coSphere *sphere;
    osg::Vec3 a;
};
#endif
