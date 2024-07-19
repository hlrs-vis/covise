/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CutGeometry_h
#define __CutGeometry_h

#include <osgTerrain/TerrainTile>
#include "ogrsf_frmts.h"
#include <cover/coVRShader.h>
#include <osg/ref_ptr>
class GeoDataLoader;


/** Callback for post processing loaded TerrainTile, and for filling in missing elements such as external external imagery.*/
struct CutGeometry : public osgTerrain::TerrainTile::TileLoadedCallback
{
    //CutGeometry(osg::Vec3d offset_, const std::vector<OGRLayer*>& = std::vector<OGRLayer*>());
    CutGeometry(GeoDataLoader *plugin);

    bool deferExternalLayerLoading() const
    {
        return false;
    }

    void loaded(osgTerrain::TerrainTile *tile, const osgDB::ReaderWriter::Options *options) const;

protected:
    GeoDataLoader* plugin;
    //const std::vector<OGRLayer*>& layers;

    //std::vector< osg::ref_ptr<osg::Node> > treeNodeVector;


    osg::ref_ptr<osg::StateSet> treeStateSet;
    opencover::coVRShader *treeShader;

    osg::ref_ptr<osg::StateSet> buildingStateSet;
};

#endif
