/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OSGResponseSurface.h"
#include "cover/VRSceneGraph.h"
using namespace opencover;

OSGResponseSurface::OSGResponseSurface(double (*setresponse)(double *), double *setlowerbound, double *setupperbound, bool *setinteger, double h)
{
    lineGeometry = new osg::Geometry;
    this->addDrawable(lineGeometry);
    lineVertices = new osg::Vec3Array;
    lineGeometry->setVertexArray(lineVertices);

    osg::DrawElementsUInt *lineBase;

    double x[3] = { 0, 0, 0 };
    int nvertx0 = 0;
    int nvertx1 = 0;

    for (x[0] = setlowerbound[0]; x[0] < setupperbound[0]; x[0] += h)
    {
        nvertx1 = 0;
        for (x[1] = setlowerbound[1]; x[1] < setupperbound[1]; x[1] += h)
        {
            x[2] = (*setresponse)(x);
            lineVertices->push_back(osg::Vec3(x[0], x[1], x[2]));
            ++nvertx1;
        }
        ++nvertx0;
    }

	lineBase = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, nvertx0*(nvertx1 - 1));
    for (int i = 0; i < nvertx0; ++i)
    {
        for (int j = 0; j < (nvertx1 - 1); ++j)
        {
            lineBase->push_back(i * nvertx1 + j);
            lineBase->push_back(i * nvertx1 + j + 1);
        }
    }
	lineGeometry->addPrimitiveSet(lineBase);

	lineBase = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, nvertx1*(nvertx0 - 1));
    for (int i = 0; i < nvertx1; ++i)
    {
        for (int j = 0; j < (nvertx0 - 1); ++j)
        {
            lineBase->push_back(j * nvertx1 + i);
            lineBase->push_back((j + 1) * nvertx1 + i);
        }
    }
	lineGeometry->addPrimitiveSet(lineBase);
    lineGeometry->setStateSet(VRSceneGraph::instance()->loadUnlightedGeostate());
}
