/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DISTORT_CONFIG_PLUGIN_H
#define _DISTORT_CONFIG_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <kernel/coVRPlugin.h>
#include <kernel/coVRPluginSupport.h>
#include "SceneVis.h"
using namespace covise;
using namespace opencover;

class DistortConfig : public coVRPlugin
{
public:
    DistortConfig();
    ~DistortConfig();

    // this will be called in PreFrame

    void preFrame();

    bool init();
    bool load();

private:
};
#endif
