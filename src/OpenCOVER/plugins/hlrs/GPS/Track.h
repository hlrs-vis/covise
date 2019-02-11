/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GPS_PLUGIN_TRACK_H
#define _GPS_PLUGIN_TRACK_H
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
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Material>

#include <vector>
#include <array>

using namespace opencover;
using namespace covise;
namespace opencover
{
class coVRLabel;
}

class GPSPoint;

//Single Track
class Track
{
public:
    Track();
    ~Track();
    void setIndex(int i);
    void addPoint(double x, double y,double v);
    void drawBirdView();
    void readFile(xercesc::DOMElement *node);
    void drawTrack();
    void transformxy();
    void transformxy2();
    osg::ref_ptr<osg::Group> SingleTrack;
    static osg::ref_ptr<osg::Material> globalDefaultMaterial;
    std::vector<std::array<double,4>> PointsVec;
    osg::ref_ptr<osg::Geode> geode;
};

#endif
