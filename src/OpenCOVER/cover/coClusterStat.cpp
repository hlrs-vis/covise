/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <ostream>

#include "coClusterStat.h"
#include <util/coTypes.h>

#include "coVRPluginSupport.h"
#include "coVRConfig.h"
#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osgDB/ReadFile>
#include <osg/Program>
#include <osg/Shader>
#include <osg/StateSet>
#include <osg/Point>
#include <osg/ShadeModel>
#include <OpenVRUI/coUIElement.h>

using namespace vrui;
using namespace opencover;
coStatGraph::coStatGraph(int Color)
{
    (void)Color;

    maxValue = 0.0001;

    coord = new osg::Vec3Array(NUM_SAMPLES * 2);
    normal = new osg::Vec3Array(1);
    coordIndex = new ushort[(NUM_SAMPLES - 1) * 4];
    normalIndex = new osg::ShortArray(1);
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        zCoords[i] = 1;
        (*coord)[i * 2].set(i, 0, zCoords[i]);
        (*coord)[i * 2 + 1].set(i, 0, 0);
    }
    for (int i = 1; i < NUM_SAMPLES - 1; i++)
    {
        coordIndex[i * 4] = (i - 1) * 2;
        coordIndex[i * 4 + 1] = (i - 1) * 2 + 1;
        coordIndex[i * 4 + 2] = i * 2 + 1;
        coordIndex[i * 4 + 3] = i * 2;
    }

    geoset1 = new osg::Geometry();
    geoset1->setVertexArray(coord);
    geoset1->setNormalArray(normal);
    geoset1->setNormalBinding(osg::Geometry::BIND_OVERALL);
    geoset1->addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4 * NUM_SAMPLES, coordIndex));

    geoset1->setUseDisplayList(false);
    geode = new osg::Geode();
    geode->addDrawable(geoset1);
    graph = new osg::MatrixTransform();
    graph->addChild(geode);
}

void coStatGraph::updateValues()
{
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        (*coord)[i * 2][2] = (zCoords[i] / max) * 10;
    }
}

void coStatGraph::updateMinMax(double val)
{
    max = 0.001;
    sum = 0.001;
    int i;
    if (val > 6000)
        return;
    if (val > 60)
        val = 60;
    else if (val < 0)
        val = 0;

    for (i = NUM_SAMPLES - 1; i > 0; i--)
    {
        zCoords[i] = zCoords[i - 1];
        sum += zCoords[i];
        if (zCoords[i] > max)
        {
            max = zCoords[i];
        }
    }
    zCoords[0] = val;
    sum += zCoords[0];
    if (zCoords[0] > max)
    {
        max = zCoords[0];
    }
}

coClusterStat::coClusterStat(int id)
{
    ID = id;
    renderGraph = new coStatGraph(coUIElement::RED);
    sendGraph = new coStatGraph(coUIElement::GREEN);
    recvGraph = new coStatGraph(coUIElement::WHITE);
    graph = new osg::MatrixTransform();
    osg::Matrix m;
    m.makeScale((cover->getSceneSize() / 3000) * (1800 / NUM_SAMPLES), (cover->getSceneSize() / 3000) * (1800 / NUM_SAMPLES), (cover->getSceneSize() / 3000) * (1800 / NUM_SAMPLES));
    graph->addChild(renderGraph->graph);
    graph->addChild(sendGraph->graph);
    graph->addChild(recvGraph->graph);
    m.makeTranslate(-NUM_SAMPLES / 2.0, 0.0, -(id * 45 + 0.0));
    renderGraph->graph->postMult(m);
    m.makeTranslate(-NUM_SAMPLES / 2.0, 0.0, -(id * 45 + 15));
    sendGraph->graph->postMult(m);
    m.makeTranslate(-NUM_SAMPLES / 2.0, 0.0, -(id * 45 + 30));
    recvGraph->graph->postMult(m);
}

coClusterStat::~coClusterStat()
{
}

void coClusterStat::updateMinMax(double renderTime, double netSend, double netRecv)
{
    renderGraph->updateMinMax(renderTime);
    sendGraph->updateMinMax(netSend);
    recvGraph->updateMinMax(netRecv);
}

void coClusterStat::updateValues()
{
    renderGraph->updateValues();
    sendGraph->updateValues();
    recvGraph->updateValues();
}

void coClusterStat::show()
{
    cover->getScene()->addChild(graph);
    std::cerr << "show" << std::endl;
}

void coClusterStat::hide()
{
    cover->getScene()->removeChild(graph);
    std::cerr << "hide" << std::endl;
}
