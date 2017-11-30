/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ROAD_TERRAIN_LOADER_H
#define ROAD_TERRAIN_LOADER_H
/****************************************************************************\
 **                                                            (C)2011 HLRS  **
 **                                                                          **
 ** Description: RoadTerrain Plugin                                          **
 **                                                                          **
 **                                                                          **
 ** Author: Florian Seybold		                                            **
 **                                                                          **
 ** History:  								                                         **
 ** Nov-01  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <osg/Texture2D>
#include <vector>
#include <osg/PositionAttitudeTransform>

#include <VehicleUtil/RoadSystem/RoadSystem.h>
#include "ogrsf_frmts.h"
#include "DecoratedGeometryTechnique.h"

class ROADTERRAINPLUGINEXPORT RoadTerrainLoader
{
public:
    RoadTerrainLoader();
    ~RoadTerrainLoader();

    bool loadTerrain(std::string filename, osg::Vec3d offset,
                     const std::vector<BoundingArea> &voidBoundingAreaVector = std::vector<BoundingArea>(),
                     const std::vector<std::string> &shapeFileVector = std::vector<std::string>());
    bool addLayer(std::string filename);

    static RoadTerrainLoader *instance();

private:
    static RoadTerrainLoader *s_instance;
    osg::PositionAttitudeTransform *roadGroup;
    xercesc::DOMElement *rootElement;

    bool terrainLoaded;
    bool layerLoaded;

    std::vector<OGRLayer *> layerVector;
};
#endif
