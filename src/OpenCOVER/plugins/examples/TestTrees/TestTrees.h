/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TestTrees_PLUGIN_H
#define _TestTrees_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2013 HLRS  **
 **                                                                          **
 ** Description: TestTrees Plugin (testInstancecRenderer)                    **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Aug-2013  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
using namespace opencover;

class TestTrees : public coVRPlugin
{
public:
    TestTrees();
    ~TestTrees();

    bool init();

    // this will be called in PreFrame
    void preFrame();

private:
};
#endif
