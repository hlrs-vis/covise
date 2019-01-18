/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GPS_PLUGIN_GPSALLPOINTS_H
#define _GPS_PLUGIN_GPSALLPOINTS_H
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
class GPSAllPoints;
class GPSALLTracks;


//All GPSPoints
class GPSALLPoints
{
public:
    GPSALLPoints();
    GPSALLPoints(osg::Group *parent);
    ~GPSALLPoints();
    void drawBirdView();
    void addPoint(GPSPoint *p);
    osg::ref_ptr<osg::Group> PointGroup;
    std::list<GPSPoint*> allPoints;
};
#endif
