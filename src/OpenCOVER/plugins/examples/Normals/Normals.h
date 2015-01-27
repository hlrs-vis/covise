/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_UTIL_NORMALS_H
#define OSG_UTIL_NORMALS_H

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/NodeVisitor>

namespace osgUtil
{

class Normals : public osg::Geode
{
public:
    enum Mode
    {
        SurfaceNormals,
        VertexNormals
    };

    Normals(osg::Node *node, float scale = 1.0, Mode mode = SurfaceNormals);

private:
    class MakeNormalsVisitor : public osg::NodeVisitor
    {
    public:
        MakeNormalsVisitor(float normalScale = 1.0, Normals::Mode = Normals::SurfaceNormals);

        void setMode(Mode mode)
        {
            _mode = mode;
        }

        void apply(osg::Geode &geode);

        osg::Vec3Array *getCoords()
        {
            return _local_coords.get();
        }

    private:
        osg::ref_ptr<osg::Vec3Array> _local_coords;
        float _normal_scale;
        Mode _mode;

        void _processPrimitive(unsigned int nv,
                               osg::Vec3Array::iterator coords,
                               osg::Vec3Array::iterator normals,
                               osg::Geometry::AttributeBinding binding,
                               osg::Matrix &mat, osg::Matrix &nmat);
    };

    // For debugging
    static void _printPrimitiveType(osg::PrimitiveSet *pset);
};

class SurfaceNormals : public Normals
{
public:
    SurfaceNormals(Node *node, float scale = 1.0)
        : Normals(node, scale, Normals::SurfaceNormals)
    {
    }
};

class VertexNormals : public Normals
{
public:
    VertexNormals(Node *node, float scale = 1.0)
        : Normals(node, scale, Normals::VertexNormals)
    {
    }
};
}

#endif
