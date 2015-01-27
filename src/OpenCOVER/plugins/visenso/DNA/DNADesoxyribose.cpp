/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DNADesoxyribose.h"
#include "DNABaseUnitConnectionPoint.h"
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>

using namespace opencover;

DNADesoxyribose::DNADesoxyribose(osg::Matrix m, int num)
    : DNABaseUnit(m, 100.0, "DESOXYRIBOSE", "dna/Desoxyribose", 55.0, num)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nDNADesoxyribose::DNADesoxyribose\n");

    // add icon as geometry
    createGeometry();

    // add connections to list
    p1 = new DNABaseUnitConnectionPoint(this, "desoxybirose1", osg::Vec3(-0.2, -0.1, 0), osg::Vec3(0, 0, -1), "phosphat1");
    p2 = new DNABaseUnitConnectionPoint(this, "desoxybirose2", osg::Vec3(-0.1, -0.55, 0), osg::Vec3(0, 0, -1), "phosphat2");
    b1 = new DNABaseUnitConnectionPoint(this, "desoxybirose3", osg::Vec3(1, -0.1, 0), osg::Vec3(0, 0, 1), "base1");
    addConnectionPoint(p1);
    addConnectionPoint(p2);
    addConnectionPoint(b1);
}

DNADesoxyribose::~DNADesoxyribose()
{
    delete p1;
    delete p2;
    delete b1;
}

void DNADesoxyribose::createGeometry()
{
    DNABaseUnit::createGeometry();
    // read geometry from file and scale it
    string connection1 = geofilename_ + "Connection1";
    string connection2 = geofilename_ + "Connection2";
    string connection3 = geofilename_ + "Connection3";
    connectionGeom1 = coVRFileManager::instance()->loadIcon(connection1.c_str());
    connectionGeom2 = coVRFileManager::instance()->loadIcon(connection2.c_str());
    connectionGeom3 = coVRFileManager::instance()->loadIcon(connection3.c_str());
    connectionGeom1->setNodeMask(0x0);
    connectionGeom2->setNodeMask(0x0);
    connectionGeom3->setNodeMask(0x0);
    // add geometry to node
    scaleTransform->addChild(connectionGeom1.get());
    scaleTransform->addChild(connectionGeom2.get());
    scaleTransform->addChild(connectionGeom3.get());
}

void DNADesoxyribose::showConnectionGeom(bool b, string connName)
{
    if (connName.compare("desoxybirose1") == 0)
    {
        if (b)
            connectionGeom2->setNodeMask(0xfffffff);
        else
            connectionGeom2->setNodeMask(0x0);
    }
    else if (connName.compare("desoxybirose2") == 0)
    {
        if (b)
            connectionGeom3->setNodeMask(0xfffffff);
        else
            connectionGeom3->setNodeMask(0x0);
    }
    else if (connName.compare("desoxybirose3") == 0)
    {
        if (b)
            connectionGeom1->setNodeMask(0xfffffff);
        else
            connectionGeom1->setNodeMask(0x0);
    }
}
