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
 ** Author: Uwe W�ssner 		                                             **
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
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Button.h>
#include <cover/ui/SelectionList.h>
#include <filesystem>

namespace fs = std::filesystem;

#ifndef RAD_TO_DEG
#define RAD_TO_DEG    57.295779513082321
#define DEG_TO_RAD   .017453292519943296
#endif


class  GeoDataLoader: public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    GeoDataLoader();
    bool init();
    ~GeoDataLoader();

    osg::ref_ptr<osg::Node> loadTerrain(std::string filename, osg::Vec3d offset);
    bool addLayer(std::string filename);

    static GeoDataLoader *instance();
    osg::Vec3 offset;
    float NorthAngle;
    void parseCoordinates(const std::string& jsonData);
    virtual bool update();
    virtual void message(int toWhom, int type, int length, const void* data);
    void setSky(int num);


private:
    static GeoDataLoader *s_instance;
    PJ_CONTEXT* ProjContext;
    PJ* ProjInstance;

    osg::ref_ptr<osg::MatrixTransform> rootNode;
    osg::ref_ptr<osg::MatrixTransform> skyRootNode;
    osg::ref_ptr<osg::Node> skyNode;
    std::map<std::string, osg::ref_ptr<osg::Node>> loadedTerrains;
    std::map<std::string, osg::ref_ptr<osg::Node>> loadedBuildings;
    opencover::ui::Menu* geoDataMenu;
    opencover::ui::Menu* terrainMenu;
    opencover::ui::Menu* buildingMenu;
    opencover::ui::Button* skyButton;
    opencover::ui::EditField* location;
    opencover::ui::SelectionList* skys;
    float northAngle;
    std::string terrainFile;
    std::string skyPath;
    int defaultSky;


};
#endif
