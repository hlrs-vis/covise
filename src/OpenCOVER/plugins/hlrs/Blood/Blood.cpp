/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Hello OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Blood.h"
#include "BloodPlugin.h"
#include <cover/coVRPluginSupport.h>
using namespace opencover;
Blood::Blood()
{
    sphere = new coSphere();

    a.set(0, 0, -9.81);

    sphere->setMaxRadius(100);
    sphere->setRenderMethod(coSphere::RENDER_METHOD_ARB_POINT_SPRITES);    //Doesn't work properly on AMD RADEON 7600M

    BloodPlugin::instance()->bloodNode->addChild(sphere);
}

// this is called if the plugin is removed at runtime
Blood::~Blood()
{
    BloodPlugin::instance()->bloodNode->removeChild(sphere);
}

void Blood::integrate(float dt, osg::Vec3 vObj)
{
    for (int i = 0; i < drops.size(); i++)
    {
        Drop *drop = &drops[i];
        if (drop->sticking)
        {
            osg::Vec3 as = -(drop->v-vObj) * friction*dt;
            drop->v = drop->v + as;
        }
        drop->v = drop->v + a;
        drop->pos = drop->pos + (drop->v * dt);
    }
}

