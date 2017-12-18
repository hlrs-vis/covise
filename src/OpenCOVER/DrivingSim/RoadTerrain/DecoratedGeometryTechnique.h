/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __DecoratedGeometryTechnique_h
#define __DecoratedGeometryTechnique_h

#include <osgTerrain/GeometryTechnique>
#include <osgTerrain/TerrainTile>
#include <iostream>
#include <osg/Version>

struct BoundingArea
{
    BoundingArea(const osg::Vec2d &min = osg::Vec2d(), const osg::Vec2d &max = osg::Vec2d())
        : _min(min)
        , _max(max)
    {
    }

    bool contains(const osg::Vec2d &point) const
    {
        return (_min.x() <= point.x() && point.x() <= _max.x()) && (_min.y() <= point.y() && point.y() <= _max.y());
    }

    osg::Vec2d _min, _max;
};

class DecoratedGeometryTechnique : public osgTerrain::GeometryTechnique
{
public:
    DecoratedGeometryTechnique(const std::vector<BoundingArea> &voidBoundingAreaVector_ = std::vector<BoundingArea>(0), osg::StateSet *treeStateSet_ = NULL, osg::StateSet *buildingStateSet_ = NULL, const std::vector<osg::ref_ptr<osg::Geode> > &shapeFileVector_ = std::vector<osg::ref_ptr<osg::Geode> >())
        : GeometryTechnique()
        , voidBoundingAreaVector(voidBoundingAreaVector_)
        , treeStateSet(treeStateSet_)
        , buildingStateSet(buildingStateSet_)
        , shapeFileVector(shapeFileVector_)
    {
    }

    DecoratedGeometryTechnique(const DecoratedGeometryTechnique &gt, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY)
        : GeometryTechnique(gt, copyop)
        , voidBoundingAreaVector(gt.voidBoundingAreaVector)
        , treeStateSet(gt.treeStateSet)
        , buildingStateSet(gt.buildingStateSet)
        , shapeFileVector(gt.shapeFileVector)
    {
    }

    virtual void update(osgUtil::UpdateVisitor *uv);
    virtual void traverse(osg::NodeVisitor &nv);

#if OPENSCENEGRAPH_SOVERSION < 72
    virtual void generateGeometry(osgTerrain::Locator *, const osg::Vec3d &);
#else
    virtual void generateGeometry(BufferData &buffer, osgTerrain::Locator *, const osg::Vec3d &);
#endif

private:
    virtual ~DecoratedGeometryTechnique()
    {
    }

    const std::vector<BoundingArea> &voidBoundingAreaVector;

    osg::ref_ptr<osg::StateSet> treeStateSet;
    osg::ref_ptr<osg::StateSet> buildingStateSet;

    std::vector<osg::ref_ptr<osg::Geode> > shapeFileVector;
};

#endif
