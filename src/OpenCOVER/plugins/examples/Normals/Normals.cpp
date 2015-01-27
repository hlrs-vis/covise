/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// code by Don Burns
// available at http://www.openscenegraph.org/pub/osgnormals.zip

//#define DEBUG 1
#ifdef DEBUG
#include <iostream>
#endif
#include "Normals.h"

#include <cover/coVRPluginSupport.h>

using namespace osg;
using namespace opencover;

namespace osgUtil
{

Normals::Normals(Node *node, float scale, Mode mode)
{
    MakeNormalsVisitor mnv(scale);
    mnv.setMode(mode);
    node->accept(mnv);

    ref_ptr<Vec3Array> coords = mnv.getCoords();
    ref_ptr<Vec4Array> colors = new Vec4Array;
    if (mode == SurfaceNormals)
        colors->push_back(Vec4(1, 0, 0, 1));
    else if (mode == VertexNormals)
        colors->push_back(Vec4(0, 1, 0, 1));

    ref_ptr<Geometry> geom = new Geometry;
    geom->setVertexArray(coords.get());
    geom->setColorArray(colors.get());
    geom->setColorBinding(Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, coords->size()));

    StateSet *sset = new StateSet;
    sset->setMode(GL_LIGHTING, StateAttribute::OFF);
    geom->setStateSet(sset);
    addDrawable(geom.get());
}

Normals::MakeNormalsVisitor::MakeNormalsVisitor(float normalScale, Normals::Mode mode)
    : NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
    , _normal_scale(normalScale)
    , _mode(mode)
{
    _local_coords = new Vec3Array;
}

void Normals::MakeNormalsVisitor::apply(Geode &geode)
{
    for (unsigned int i = 0; i < geode.getNumDrawables(); i++)
    {
        Geometry *geom = dynamic_cast<Geometry *>(geode.getDrawable(i));
        if (geom)
        {
            Vec3Array *coords = dynamic_cast<Vec3Array *>(geom->getVertexArray());
            if (coords == 0L)
                continue;

            Vec3Array *normals = dynamic_cast<Vec3Array *>(geom->getNormalArray());
            if (normals == 0L)
                continue;

            Geometry::AttributeBinding binding = geom->getNormalBinding();
            if (binding == Geometry::BIND_OFF)
                continue;

            Geode *g = &geode;
            Matrix *mat = cover->getWorldCoords(g);
            Matrix nmat = *mat;
            nmat.setTrans(Vec3(0., 0., 0.));

            if (binding == Geometry::BIND_OVERALL)
            {
                Vec3 v(0, 0, 0);
                Vec3 n = normals->front();

                Vec3Array::iterator coord_index = coords->begin();
                while (coord_index != coords->end())
                    v += *(coord_index++);
                v /= (float)(coords->size());
                v = v * *mat;

                n *= _normal_scale;
                n = n * nmat;
                _local_coords->push_back(v);
                _local_coords->push_back((v + n));
            }
            else // BIND_PER_PRIMTIVE_SET, BIND_PER_PRIMTITIVE, BIND_PER_VERTEX
            {
                Geometry::PrimitiveSetList &primitiveSets = geom->getPrimitiveSetList();
                Geometry::PrimitiveSetList::iterator itr;

                Vec3Array::iterator coord_index = coords->begin();
                Vec3Array::iterator normals_index = normals->begin();

                for (itr = primitiveSets.begin(); itr != primitiveSets.end(); ++itr)
                {
#ifdef DEBUG
                    _printPrimitiveType((*itr).get());
#endif
                    if (binding == Geometry::BIND_PER_PRIMITIVE_SET)
                    {
                        Vec3 v(0, 0, 0);
                        Vec3 n = *(normals_index++);
                        int ni = (*itr)->getNumIndices();
                        for (int i = 0; i < ni; i++)
                            v += *(coord_index++);
                        v /= (float)(ni);
                        v = v * *mat;

                        n *= _normal_scale;
                        n = n * nmat;
                        _local_coords->push_back(v);
                        _local_coords->push_back((v + n));
                    }
                    else
                    {
                        switch ((*itr)->getMode())
                        {
                        case (PrimitiveSet::LINES):
                            for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                            {
                                _processPrimitive(1, coord_index,
                                                  normals_index, binding, *mat, nmat);
                                coord_index += 1;
                                normals_index += 1;
                            }
                            break;

                        case (PrimitiveSet::TRIANGLES):
                            for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                            {
                                _processPrimitive(3, coord_index,
                                                  normals_index, binding, *mat, nmat);
                                coord_index += 3;
                                normals_index += 3;
                            }
                            break;

                        case (PrimitiveSet::TRIANGLE_STRIP):
                            for (unsigned int j = 0; j < (*itr)->getNumIndices() - 2; j++)
                            {
                                _processPrimitive(3, coord_index,
                                                  normals_index, binding, *mat, nmat);
                                coord_index++;
                                normals_index++;
                            }
                            coord_index += 2;
                            if (binding == Geometry::BIND_PER_VERTEX)
                                normals_index += 2;
                            break;

                        case (PrimitiveSet::TRIANGLE_FAN):
                            break;

                        case (PrimitiveSet::QUADS):
                            for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                            {
                                _processPrimitive(4, coord_index,
                                                  normals_index, binding, *mat, nmat);
                                coord_index += 4;
                                normals_index += 4;
                            }
                            break;

                        case (PrimitiveSet::POLYGON):
                        {
                            DrawArrayLengths *draw = dynamic_cast<DrawArrayLengths *>(itr->get());
                            if (!draw)
                                break;
                            for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                            {
                                int nv = (*draw)[j];
                                _processPrimitive(nv, coord_index,
                                                  normals_index, binding, *mat, nmat);
                                coord_index += nv;
                                normals_index += nv;
                            }
                        }
                        break;
                        case (PrimitiveSet::QUAD_STRIP):
                            break;
                        default:
                            break;
                        }
                    }
                }
            }
        }
    }
    traverse(geode);
}

void Normals::MakeNormalsVisitor::_processPrimitive(unsigned int nv,
                                                    Vec3Array::iterator coords,
                                                    Vec3Array::iterator normals,
                                                    Geometry::AttributeBinding binding,
                                                    Matrix &mat, Matrix &nmat)
{

    Vec3 v(0, 0, 0);
    Vec3 n(0, 0, 0);
    if (_mode == SurfaceNormals)
    {
        if (binding == Geometry::BIND_PER_VERTEX)
        {
            for (unsigned int i = 0; i < nv; i++)
                n += *(normals++);
            n /= (float)(nv);
        }

        for (unsigned int i = 0; i < nv; i++)
            v += *(coords++);
        v /= (float)(nv);
        v = v * mat;

        n *= _normal_scale;
        n = n * nmat;

        _local_coords->push_back(v);
        _local_coords->push_back((v + n));
    }
    else if (_mode == VertexNormals)
    {
        for (unsigned int i = 0; i < nv; i++)
        {
            v = *(coords++);
            v = v * mat;

            n = *(normals++);
            n *= _normal_scale;
            n = n * nmat;

            _local_coords->push_back(v);
            _local_coords->push_back((v + n));
        }
    }
}

void Normals::_printPrimitiveType(osg::PrimitiveSet *pset)
{
#ifdef DEBUG
    std::cout << (pset->getMode() == PrimitiveSet::POINTS ? "POINTS" : pset->getMode() == PrimitiveSet::LINES ? "LINES" : pset->getMode() == PrimitiveSet::LINE_STRIP ? "LINE_STRIP" : pset->getMode() == PrimitiveSet::LINE_LOOP ? "LINE_LOOP" : pset->getMode() == PrimitiveSet::TRIANGLES ? "TRIANGLES" : pset->getMode() == PrimitiveSet::TRIANGLE_STRIP ? "TRIANGLE_STRIP" : pset->getMode() == PrimitiveSet::TRIANGLE_FAN ? "TRIANGLE_FAN" : pset->getMode() == PrimitiveSet::QUADS ? "QUADS" : pset->getMode() == PrimitiveSet::QUAD_STRIP ? "QUAD_STRIP" : pset->getMode() == PrimitiveSet::POLYGON ? "POLYGON" : "Dunno") << std::endl;
#else
    (void)pset;
#endif
}
}
