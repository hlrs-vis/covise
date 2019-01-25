/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: GPS OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner, T.Gudat	                                             **
 **                                                                          **
 ** History:  								     **
 ** June 2008  v1	    				       		     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "GPSPoint.h"
#include "GPSALLPoints.h"

#include <osg/Group>
#include <osg/Switch>
#include <osg/Geode>

using namespace opencover;

// GPSALLPoints
GPSALLPoints::GPSALLPoints()
{
    PointGroup = new osg::Group();
    PointGroup->setName("All Points of file");
    //fprintf(stderr, "--- GPSALLPoints created ---\n");
}
GPSALLPoints::~GPSALLPoints()
{
    for (auto *x : allPoints){
        delete x;
    }

    fprintf(stderr, "GPSALLPoints deleted\n");
}
void GPSALLPoints::addPoint(GPSPoint *p)
{
    p->setIndex(allPoints.size());
    allPoints.push_back(p);
    PointGroup->addChild(p->Point);
    //fprintf(stderr, "Point added to ap\n");
}
void GPSALLPoints::drawBirdView()
{
    for (std::list<GPSPoint*>::iterator it = allPoints.begin(); it != allPoints.end(); it++){
        (*it)->drawSphere();
        //(*it)->drawDetail();
    }
}
