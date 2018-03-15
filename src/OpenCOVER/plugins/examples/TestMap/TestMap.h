/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TestMap_PLUGIN_H
#define _TestMap_PLUGIN_H

#include <cover/coVRPluginSupport.h>


#include <cover/coTabletUI.h>
#include "TestMap.h"

using namespace opencover;

class TestMap : public coVRPlugin, public coTUIListener
{
public:
    TestMap();
    ~TestMap();

    bool init();

    // this will be called in PreFrame
    void preFrame();

private:
    void tabletPressEvent(coTUIElement *);
    void tabletEvent(coTUIElement *);

    coTUIMap *tuiMap;
    coTUITab *testMapTab;
};
#endif
