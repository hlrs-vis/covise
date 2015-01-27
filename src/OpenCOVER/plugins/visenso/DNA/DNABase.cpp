/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DNABase.h"
#include "DNABaseUnitConnectionPoint.h"
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>

using namespace opencover;

DNABase::DNABase(osg::Matrix m, float size, const char *interactorName, std::string geofilename, float boundingRadius, bool left, int num)
    : DNABaseUnit(m, size, interactorName, geofilename, boundingRadius, num)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nDNABase::DNABase %d\n", left);

    // add icon as geometry
    createGeometry();

    // add connections to list
    if (left)
        d1 = new DNABaseUnitConnectionPoint(this, "base1", osg::Vec3(-0.65, 0, 0.05), osg::Vec3(0, 1, 0), "desoxybirose3");
    else
        d1 = new DNABaseUnitConnectionPoint(this, "base1", osg::Vec3(0.65, 0, 0.05), osg::Vec3(0, 1, 0), "desoxybirose3");

    d1->setRotation(!left);
    addConnectionPoint(d1);
}

DNABase::~DNABase()
{
    delete d1;
}

void DNABase::createGeometry()
{
    DNABaseUnit::createGeometry();
    // read geometry from file and scale it
    string connection = geofilename_ + "Connections";
    connectionGeom = coVRFileManager::instance()->loadIcon(connection.c_str());
    connectionGeom->setNodeMask(0x0);
    // add geometry to node
    scaleTransform->addChild(connectionGeom.get());
}

void DNABase::showConnectionGeom(bool b, string connName)
{
    if (connName.compare("base1") != 0)
    {
        if (b)
            connectionGeom->setNodeMask(0xfffffff);
        else
            connectionGeom->setNodeMask(0x0);
    }
}
