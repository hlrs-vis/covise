/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


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

#include "TestTrees.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <PluginUtil/coInstanceRenderer.h>

TestTrees::TestTrees()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "TestTrees::TestTrees\n");
}

// this is called if the plugin is removed at runtime
TestTrees::~TestTrees()
{
    fprintf(stderr, "TestTrees::~TestTrees\n");
}

bool TestTrees::init()
{
    int id1 = coInstanceRenderer::instance()->addObject("share/covise/materials/baum-1.png", 422 / 100.0, 853 / 100.0);
    int id2 = coInstanceRenderer::instance()->addObject("share/covise/materials/baum-2.png", 5.0, 5.0);
    int id3 = coInstanceRenderer::instance()->addObject("share/covise/materials/baum-3.png", 512 / 100.0, 1024 / 100.0);
    int id4 = coInstanceRenderer::instance()->addObject("share/covise/materials/baum-4.png", 848 / 100.0, 1708 / 100.0);
    int id5 = coInstanceRenderer::instance()->addObject("share/covise/materials/baum-5.png", 422 / 100.0, 2048 / 100.0);
    int id6 = coInstanceRenderer::instance()->addObject("share/covise/materials/baum-6.png", 512 / 100.0, 1024 / 100.0);
    int id7 = coInstanceRenderer::instance()->addObject("share/covise/materials/baum-7.png", 477 / 100.0, 496 / 100.0);
    int id8 = coInstanceRenderer::instance()->addObject("share/covise/materials/baum-8.png", 487 / 100.0, 498 / 100.0);
    int id9 = coInstanceRenderer::instance()->addObject("share/covise/materials/baum-9.png", 490 / 100.0, 496 / 100.0);
    int numTrees = 10000;
    osg::Vec3Array *va = new osg::Vec3Array(numTrees);
    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id1);

    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id1);

    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id2);

    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id3);

    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id4);

    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id5);

    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id6);

    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id7);

    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id8);

    for (int i = 0; i < numTrees; i++)
    {
        (*va)[i] = osg::Vec3(1000.0 * rand() / RAND_MAX, 1000.0 * rand() / RAND_MAX, 0);
    }
    coInstanceRenderer::instance()->addInstances(*va, id9);
    return true;
}

void
TestTrees::preFrame()
{
}

COVERPLUGIN(TestTrees)
