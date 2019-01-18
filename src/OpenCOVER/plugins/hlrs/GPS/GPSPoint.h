/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GPS_PLUGIN_GPSPOINT_H
#define _GPS_PLUGIN_GPSPOINT_H
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
#include <xercesc/dom/DOM.hpp>

using namespace opencover;
using namespace covise;
namespace opencover
{
class coVRLabel;
}


//Single Point
class GPSPoint
{
private:
    enum pointType { Trackpoint, Good, Medium ,Bad,Angst,Text,Foto,Sprachaufnahme,OtherChoice};
    pointType PT;
    double longitude;
    double latitude;
    double altitude;
    double time;
    float speed;
    std::string text;

public:
    GPSPoint();
    GPSPoint(osg::Group *parent);
    ~GPSPoint();  
    pointType gettype (void);
    void setPointData (double x, double y, double z, double time, float v, std::string &name);
    void setIndex(int i);
    void drawSphere();
    void drawDetail();
    osg::Vec3 getCoArray();
    float getSpeed();
    void readFile(xercesc::DOMElement *node);
    osg::ref_ptr<osg::Geode> Point;
    

};


#endif
