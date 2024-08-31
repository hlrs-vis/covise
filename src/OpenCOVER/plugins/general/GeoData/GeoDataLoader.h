/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GEO_DATA_LOADER_H
#define GEO_DATA_LOADER_H
/****************************************************************************\
 **                                                            (C)2024 HLRS  **
 **                                                                          **
 ** Description: GeoDataLoader Plugin                                        **
 **                                                                          **
 **                                                                          **
 ** Author: Uwe Wössner 		                                             **
 **                                                                          **
 ** History:  								                                 **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <osg/Texture2D>
#include <vector>
#include <osg/PositionAttitudeTransform>
#include <osgTerrain/Terrain>
#include <cover/coVRPlugin.h>
#include "CutGeometry.h"
#include <proj.h>

#ifndef RAD_TO_DEG
#define RAD_TO_DEG    57.295779513082321
#define DEG_TO_RAD   .017453292519943296
#endif


class  GeoDataLoader: public opencover::coVRPlugin
{
public:
    GeoDataLoader();
    bool init();
    ~GeoDataLoader();

    bool loadTerrain(std::string filename, osg::Vec3d offset);
    bool addLayer(std::string filename);

    static GeoDataLoader *instance();
    osg::Vec3 offset;
    float NorthAngle;
    void parseCoordinates(const std::string& jsonData);


private:
    static GeoDataLoader *s_instance;
    PJ_CONTEXT* ProjContext;
    PJ* ProjInstance;


};
#endif
