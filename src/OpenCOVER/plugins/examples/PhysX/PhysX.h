/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PhysX_PLUGIN_H
#define _PhysX_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: PhysX OpenCOVER Plugin (draws a cube)                          **
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

class PhysX : public opencover::coVRPlugin
{
public:
    PhysX();
    ~PhysX();
    virtual bool destroy();

    virtual void key(int type, int keySym, int mod) override;
   
private:
    osg::ref_ptr<osg::Geode> basicShapesGeode;
    osg::observer_ptr<osg::Group> _root;
};
#endif
