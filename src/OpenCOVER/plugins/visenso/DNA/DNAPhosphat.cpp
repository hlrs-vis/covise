/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DNAPhosphat.h"
#include "DNABaseUnitConnectionPoint.h"

#include <cover/coVRPluginSupport.h>

using namespace opencover;

DNAPhosphat::DNAPhosphat(osg::Matrix m, int num)
    : DNABaseUnit(m, 150.0, "PHOSPHAT", "dna/Phosphat", 80.0, num)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nDNAPhosphat::DNAPhosphat\n");

    // add icon as geometry
    createGeometry();

    // add connections to list
    d1 = new DNABaseUnitConnectionPoint(this, "phosphat1", osg::Vec3(0.0, 0.55, 0), osg::Vec3(0, -1, 0), "desoxybirose1");
    d2 = new DNABaseUnitConnectionPoint(this, "phosphat2", osg::Vec3(0.0, -0.85, 0), osg::Vec3(0, -1, 0), "desoxybirose2");
    addConnectionPoint(d1);
    addConnectionPoint(d2);
}

DNAPhosphat::~DNAPhosphat()
{
    delete d1;
    delete d2;
}
