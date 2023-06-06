/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include "TesselationTestPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

#include <osg/Geometry>
#include <osg/PatchParameter>
#include <cover/coVRConfig.h>

using namespace opencover;

TesselationTestPlugin::TesselationTestPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "TesselationTestPlugin::TesselationTestPlugin\n");
}

bool TesselationTestPlugin::init()
{
    m_geode = createIcosahedron();
    return cover->getObjectsRoot()->addChild(m_geode);
}

osg::ref_ptr<osg::Geode> TesselationTestPlugin::createIcosahedron()
{
    osg::Geode *geode = new osg::Geode();
    geode->setName("Icosahedron");
    osg::Geometry *geometry = new osg::Geometry();
    const unsigned int Faces[] = {
        2, 1, 0,
        3, 2, 0,
        4, 3, 0,
        5, 4, 0,
        1, 5, 0,

        11, 6, 7,
        11, 7, 8,
        11, 8, 9,
        11, 9, 10,
        11, 10, 6,

        1, 2, 6,
        2, 3, 7,
        3, 4, 8,
        4, 5, 9,
        5, 1, 10,

        2, 7, 6,
        3, 8, 7,
        4, 9, 8,
        5, 10, 9,
        1, 6, 10
    };
    int IndexCount = sizeof(Faces) / sizeof(Faces[0]);
    const float Verts[] = {
        0.000f, 0.000f, 1.000f,
        0.894f, 0.000f, 0.447f,
        0.276f, 0.851f, 0.447f,
        -0.724f, 0.526f, 0.447f,
        -0.724f, -0.526f, 0.447f,
        0.276f, -0.851f, 0.447f,
        0.724f, 0.526f, -0.447f,
        -0.276f, 0.851f, -0.447f,
        -0.894f, 0.000f, -0.447f,
        -0.276f, -0.851f, -0.447f,
        0.724f, -0.526f, -0.447f,
        0.000f, 0.000f, -1.000f
    };

    int VertexCount = sizeof(Verts) / sizeof(float);
    osg::Vec3Array *vertices = new osg::Vec3Array();
    for (int i = 0; i < VertexCount; i += 3)
    {
        vertices->push_back(osg::Vec3(Verts[i], Verts[i + 1], Verts[i + 2]));
    }

    geometry->setVertexArray(vertices);
    geometry->addPrimitiveSet(new osg::DrawElementsUInt(osg::PrimitiveSet::PATCHES, IndexCount, Faces));

    geode->addDrawable(geometry);

    return geode;
}

// this is called if the plugin is removed at runtime
TesselationTestPlugin::~TesselationTestPlugin()
{
    fprintf(stderr, "TesselationTestPlugin::~TesselationTestPlugin\n");
    cover->getObjectsRoot()->removeChild(m_geode);
}

void
TesselationTestPlugin::preFrame()
{
}

COVERPLUGIN(TesselationTestPlugin)
