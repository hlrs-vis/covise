/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __LevelTerrainCallback_h
#define __LevelTerrainCallback_h

#include <osgTerrain/TerrainTile>
#include "ogrsf_frmts.h"
#include "mtrand.h"
#include <cover/coVRShader.h>
#include <osg/ref_ptr>

#include "DecoratedGeometryTechnique.h"

/** Callback for post processing loaded TerrainTile, and for filling in missing elements such as external external imagery.*/
struct LevelTerrainCallback : public osgTerrain::TerrainTile::TileLoadedCallback
{
    //LevelTerrainCallback(osg::Vec3d offset_, const std::vector<OGRLayer*>& = std::vector<OGRLayer*>());
    LevelTerrainCallback(osg::Vec3d offset_, const std::vector<BoundingArea> &areaVector, const std::vector<osg::ref_ptr<osg::Geode> > &shapeFileVector);

    bool deferExternalLayerLoading() const
    {
        return false;
    }

    void loaded(osgTerrain::TerrainTile *tile, const osgDB::ReaderWriter::Options *options) const;

protected:
    osg::Vec3d offset;
    //const std::vector<OGRLayer*>& layers;

    //std::vector< osg::ref_ptr<osg::Node> > treeNodeVector;

    //mutable MTRand mt;

    std::vector<BoundingArea> voidBoundingAreaVector;
    std::vector<osg::ref_ptr<osg::Geode> > shapeFileVector;

    osg::ref_ptr<osg::StateSet> treeStateSet;
    opencover::coVRShader *treeShader;

    osg::ref_ptr<osg::StateSet> buildingStateSet;
};

#endif
