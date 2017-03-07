/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TesselationTest_PLUGIN_H
#define TesselationTest_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: TesselationTest Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>

class TesselationTestPlugin : public opencover::coVRPlugin
{
public:
    TesselationTestPlugin();
    ~TesselationTestPlugin();

    // this will be called in PreFrame
    void preFrame();

    virtual bool init();

private:
    osg::ref_ptr<osg::Geode> m_geode;
    osg::ref_ptr<osg::Geode> createIcosahedron();
};
#endif
