/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _OrientationIndicator_PLUGIN_H
#define _OrientationIndicator_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2024 RRZK  **
 **                                                                          **
 ** Description: OrientationIndicator OpenCOVER Plugin (draws axis showing   **
 **              showing the orientation of the loaded model)                **
 **                                                                          **
 ** Author: D.Wickeroth                                                      **
 **                                                                          **
 ** History:  								                                 **
 ** March 2024  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <osg/Geode>
#include <osg/MatrixTransform>

class OrientationIndicator : public opencover::coVRPlugin
{
public:
    OrientationIndicator();
    ~OrientationIndicator();
    virtual bool destroy();
    virtual void preFrame();
    

private:
    osg::ref_ptr<osg::Geode> oi_Geode;
    osg::ref_ptr<osg::MatrixTransform> oi_Trans;
    osg::Vec3f oi_offset;
};
#endif
