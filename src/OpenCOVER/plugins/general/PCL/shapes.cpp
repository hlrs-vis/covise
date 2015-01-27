/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * shapes.cpp
 *
 *  Created on: Nov 10, 2012
 *      Author: asher
 */

#include <shapes.h>
#include <osg/Geometry>
#include <osg/Array>

static bool
osgpcl::buildArrowGeometry(osg::Geometry *geom)
{
    // Create an arrow pointing in the +z direction.
    const float sD(.05); // shaft diameter
    const float hD(.075); // head diameter
    const float len(1.); // length
    const float sh(.65); // length from base to start of head

    osg::Vec3Array *verts(new osg::Vec3Array);
    verts->resize(22);
    geom->setVertexArray(verts);

    osg::Vec3Array *norms(new osg::Vec3Array);
    norms->resize(22);
    geom->setNormalArray(norms);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    {
        osg::Vec4Array *osgC = new osg::Vec4Array;
        osgC->push_back(osg::Vec4(1., 1., 1., 1.));
        geom->setColorArray(osgC);
        geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    }

    // Shaft
    (*verts)[0] = osg::Vec3(sD, 0., 0.);
    (*verts)[1] = osg::Vec3(sD, 0., sh);
    (*verts)[2] = osg::Vec3(0., -sD, 0.);
    (*verts)[3] = osg::Vec3(0., -sD, sh);
    (*verts)[4] = osg::Vec3(-sD, 0., 0.);
    (*verts)[5] = osg::Vec3(-sD, 0., sh);
    (*verts)[6] = osg::Vec3(0., sD, 0.);
    (*verts)[7] = osg::Vec3(0., sD, sh);
    (*verts)[8] = osg::Vec3(sD, 0., 0.);
    (*verts)[9] = osg::Vec3(sD, 0., sh);

    (*norms)[0] = osg::Vec3(1., 0., 0.);
    (*norms)[1] = osg::Vec3(1., 0., 0.);
    (*norms)[2] = osg::Vec3(0., -1., 0.);
    (*norms)[3] = osg::Vec3(0., -1., 0.);
    (*norms)[4] = osg::Vec3(-1., 0., 0.);
    (*norms)[5] = osg::Vec3(-1., 0., 0.);
    (*norms)[6] = osg::Vec3(0., 1., 0.);
    (*norms)[7] = osg::Vec3(0., 1., 0.);
    (*norms)[8] = osg::Vec3(1., 0., 0.);
    (*norms)[9] = osg::Vec3(1., 0., 0.);

    geom->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLE_STRIP, 0, 10));

    // Head
    (*verts)[10] = osg::Vec3(hD, -hD, sh);
    (*verts)[11] = osg::Vec3(hD, hD, sh);
    (*verts)[12] = osg::Vec3(0., 0., len);
    osg::Vec3 norm = ((*verts)[11] - (*verts)[10]) ^ ((*verts)[12] - (*verts)[10]);
    norm.normalize();
    (*norms)[10] = norm;
    (*norms)[11] = norm;
    (*norms)[12] = norm;

    (*verts)[13] = osg::Vec3(hD, hD, sh);
    (*verts)[14] = osg::Vec3(-hD, hD, sh);
    (*verts)[15] = osg::Vec3(0., 0., len);
    norm = ((*verts)[14] - (*verts)[13]) ^ ((*verts)[15] - (*verts)[13]);
    norm.normalize();
    (*norms)[13] = norm;
    (*norms)[14] = norm;
    (*norms)[15] = norm;

    (*verts)[16] = osg::Vec3(-hD, hD, sh);
    (*verts)[17] = osg::Vec3(-hD, -hD, sh);
    (*verts)[18] = osg::Vec3(0., 0., len);
    norm = ((*verts)[17] - (*verts)[16]) ^ ((*verts)[18] - (*verts)[16]);
    norm.normalize();
    (*norms)[16] = norm;
    (*norms)[17] = norm;
    (*norms)[18] = norm;

    (*verts)[19] = osg::Vec3(-hD, -hD, sh);
    (*verts)[20] = osg::Vec3(hD, -hD, sh);
    (*verts)[21] = osg::Vec3(0., 0., len);
    norm = ((*verts)[20] - (*verts)[19]) ^ ((*verts)[21] - (*verts)[19]);
    norm.normalize();
    (*norms)[19] = norm;
    (*norms)[20] = norm;
    (*norms)[21] = norm;

    geom->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLES, 10, 12));

    return (true);
}
