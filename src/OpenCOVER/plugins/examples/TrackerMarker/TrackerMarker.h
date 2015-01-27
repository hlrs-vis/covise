/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRACKERMARKER_PLUGIN_H
#define _TRACKERMARKER_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: TrackerMarker Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

class TrackerMarkerPlugin : public coVRPlugin
{
    osg::Group *group;

public:
    TrackerMarkerPlugin();
    ~TrackerMarkerPlugin();
    bool init();

    // this will be called in PreFrame
    void preFrame();
};
#endif
