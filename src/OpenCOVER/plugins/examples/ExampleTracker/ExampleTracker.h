/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ExampleTracker_PLUGIN_H
#define _ExampleTracker_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: ExampleTracker OpenCOVER Plugin (demonstrates how to implement a tracker plugin)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>

class ExampleTracker : public opencover::coVRPlugin
{
public:
    ExampleTracker();
    ~ExampleTracker();
    virtual unsigned int button(int station);
    virtual void getMatrix(int station, osg::Matrix &mat);
};
#endif
