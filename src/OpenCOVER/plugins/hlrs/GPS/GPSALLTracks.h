/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GPS_PLUGIN_GPSALLTRACKS_H
#define _GPS_PLUGIN_GPSALLTRACKS_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: GPS OpenCOVER Plugin (is polite)                            **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner, T.Gudat		                                     **
 **                                                                          **
 ** History:  								     **
 ** June 2008  v1	    				       		     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>

using namespace opencover;
using namespace covise;
namespace opencover
{
class coVRLabel;
}
class GPSPoint;
class Track;


//AllTracks
class GPSALLTracks
{
public:
    GPSALLTracks();
    GPSALLTracks(osg::Group *parent);
    ~GPSALLTracks();
    void addTrack(Track *p);
    void drawBirdView();
    osg::ref_ptr<osg::Group> TrackGroup;
    std::list<Track*> allTracks;
};
#endif
