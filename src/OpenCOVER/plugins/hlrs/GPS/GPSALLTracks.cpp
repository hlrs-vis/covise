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
#include "GPSALLTracks.h"
#include "Track.h"

#include <osg/Group>
#include <osg/Switch>

using namespace opencover;

// GPSALLTracks
GPSALLTracks::GPSALLTracks()
{
    TrackGroup = new osg::Group();
    TrackGroup->setName("All Tracks of file");
    //fprintf(stderr, "--- GPSALLTracks created ---\n");
}

GPSALLTracks::~GPSALLTracks()
{
    for (auto *x : allTracks){
        delete x;
    }
    fprintf(stderr, "GPSALLPoints deleted\n");
}
void GPSALLTracks::addTrack(Track *t)
{
    t->setIndex(allTracks.size());
    allTracks.push_back(t);
    TrackGroup->addChild(t->SingleTrack);
    //fprintf(stderr, "Track added to at\n");
}
void GPSALLTracks::drawBirdView()
{
    for (std::list<Track*>::iterator it = allTracks.begin(); it != allTracks.end(); it++){
        //(*it)->drawBirdView();
        (*it)->drawTrack();
    }
}
