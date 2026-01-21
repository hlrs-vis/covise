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
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Button.h>
#include <cover/ui/SelectionList.h>
#include <filesystem>
#include <optional>

namespace fs = std::filesystem;

#ifndef RAD_TO_DEG
#define RAD_TO_DEG    57.295779513082321
#define DEG_TO_RAD   .017453292519943296
#endif
class skyEntry
{
public:
    skyEntry(const std::string& n, const std::string& fn);
    ~skyEntry();
    skyEntry(const skyEntry &se);
    std::string name;
    std::string fileName;
    osg::ref_ptr<osg::Node> skyNode;
};

class  GeoDataLoader: public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    GeoDataLoader();
    bool init();
    ~GeoDataLoader();

    osg::ref_ptr<osg::Node> loadTerrain(std::string filename, osg::Vec3d localOffset);
    bool addLayer(std::string filename);

    static GeoDataLoader *instance();
    osg::Vec3 rootOffset{0.0, 0.0, 0.0};
    float trueNorthDegree = 0.0f;
    float NorthAngle;

    struct geoLocation
    {
        double latitude;
        double longitude;
        double easting;
        double northing;
        double height;
        std::string displayName;
    };

    virtual bool update();
    virtual void message(int toWhom, int type, int length, const void* data);
    void setSky(int num);
    void setSky(std::string fileName);
    void setRootTransform(const osg::Vec3& offset, float trueNorthDeg);
    std::optional<geoLocation> parseCoordinates(const std::string& jsonData);
    void jumpToLocation(const osg::Vec3d &worldPos);

private:
    static GeoDataLoader *s_instance;
    PJ_CONTEXT* ProjContext;
    PJ* ProjInstance;

    osg::ref_ptr<osg::MatrixTransform> rootNode;
    osg::ref_ptr<osg::MatrixTransform> skyRootNode;
    osg::ref_ptr<osg::Node> currentSkyNode;
    osg::ref_ptr<osg::Node> skyNode;
    osg::ref_ptr<osg::Node> terrainNode = nullptr;
    std::map<std::string, osg::ref_ptr<osg::Node>> loadedTerrains;
    std::map<std::string, osg::ref_ptr<osg::Node>> loadedBuildings;

    opencover::ui::Menu* geoDataMenu;
    opencover::ui::Menu* terrainMenu;
    opencover::ui::Menu* buildingMenu;
    opencover::ui::Group* skyGroup;
    opencover::ui::Group* originGroup;
    opencover::ui::Group* locationGroup;

    opencover::ui::Button* skyButton;
    opencover::ui::Button* applyOffset;
    opencover::ui::EditField* location;
    opencover::ui::EditField* easting;
    opencover::ui::EditField* northing;
    opencover::ui::EditField* height;
    opencover::ui::EditField* trueNorth;
    opencover::ui::SelectionList* datasetList;
    opencover::ui::SelectionList* skys;
    std::list<skyEntry> skyEntries;

    struct DatasetInfo
    {
        std::string name;
        double easting;
        double northing;
        double height;
        double trueNorth;
    };
    std::vector<DatasetInfo> datasets;

    float northAngle;
    std::string terrainFile;
    std::string skyPath;
    int defaultSky;

    std::string tempEastingText;
    std::string tempNorthingText;
    std::string tempHeightText;
    std::string tempTrueNorthText;
};
#endif
