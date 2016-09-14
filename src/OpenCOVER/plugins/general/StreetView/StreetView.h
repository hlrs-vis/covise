/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef STREET_VIEW_H
#define STREET_VIEW_H

/****************************************************************************\ 
 **                                                            (C)2016 HLRS  **
 **                                                                          **
 ** Description: Streetview Plugin				                             **
 **                                                                          **
 **                                                                          **
 ** Author: M.Guedey		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Sep-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>

class IndexParser;

class StreetView : public opencover::coVRPlugin
{
public:
    StreetView();
    ~StreetView();
    void preFrame();
	bool init();

private:
	IndexParser *indexParser;
	osg::ref_ptr<osg::Node> stationNode;
};
#endif
