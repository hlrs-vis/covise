/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: osgBulletPlugin OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "osgBullet.h"
#include <cover/coVRPluginSupport.h>
using namespace opencover;
osgBulletPlugin::osgBulletPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "osgBulletPlugin World\n");
}

// this is called if the plugin is removed at runtime
osgBulletPlugin::~osgBulletPlugin()
{
    fprintf(stderr, "Goodbye\n");
}

COVERPLUGIN(osgBulletPlugin)
