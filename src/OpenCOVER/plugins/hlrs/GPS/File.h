/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GPS_PLUGIN_FILE_H
#define _GPS_PLUGIN_FILE_H
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
#include "GPSALLPoints.h"
#include "GPSALLTracks.h"

using namespace opencover;
using namespace covise;
namespace opencover
{
class coVRLabel;
}
class GPSAllPoints;
class GPSALLTracks;

//For single files
class File
{
public:
    File();
    File(const char *filename);
    ~File();
    void readFile(const std::string &filename);
    void addAllTracks(GPSALLTracks *p);
    void addAllPoints(GPSALLPoints *p);
    std::string name;
    osg::ref_ptr<osg::Group> FileGroup;
    osg::ref_ptr<osg::Switch> SwitchPoints;
    osg::ref_ptr<osg::Switch> SwitchTracks;
    osg::ref_ptr<osg::Group> TrackGroup;

    std::list<GPSALLTracks*> fileAllTracks;
    std::list<GPSALLPoints*> fileAllPoints;
};
#endif
