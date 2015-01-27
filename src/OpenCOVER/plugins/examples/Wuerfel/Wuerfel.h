/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Wuerfel_PLUGIN_H
#define _Wuerfel_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Wuerfel OpenCOVER Plugin (draws a cube)                          **
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
#include <osg/Geode>

class Wuerfel : public opencover::coVRPlugin
{
public:
    Wuerfel();
    ~Wuerfel();
    virtual bool destroy();

private:
    osg::ref_ptr<osg::Geode> basicShapesGeode;
};
#endif
