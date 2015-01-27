/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __RoadFootprint_h
#define __RoadFootprint_h

#include <osgTerrain/TerrainTile>

/** Callback for post processing loaded TerrainTile, and for filling in missing elements such as external external imagery.*/
struct RoadFootprintTileLoadedCallback : public osgTerrain::TerrainTile::TileLoadedCallback
{
    bool deferExternalLayerLoading() const
    {
        return false;
    }

    void loaded(osgTerrain::TerrainTile *tile, const osgDB::ReaderWriter::Options *options) const;
};

#endif
