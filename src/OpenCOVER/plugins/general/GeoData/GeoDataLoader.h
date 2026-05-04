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
#include <osg/TexMat>
#include <osg/StateSet>
#include <cover/coVRPlugin.h>
#include <cover/coVRShader.h>
#include <proj.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Button.h>
#include <cover/ui/Label.h>
#include <cover/ui/SelectionList.h>
#include <filesystem>
#include <optional>

#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <cover/coIntersection.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
class GeoDataLoader;

class editTerrain : public vrui::coCombinedButtonInteraction
{
public:
    editTerrain(GeoDataLoader *g);
    // enable interaction
    void enableIntersection();
    // disable interaction
    void disableIntersection();

private:
    virtual void createGeometry();


    // check whether interactor is enabled
    bool isEnabled();


    osg::ref_ptr<osg::Node> geometryNode; ///< Geometry node
    osg::ref_ptr<osg::MatrixTransform> moveTransform;
    osg::ref_ptr<osg::MatrixTransform> scaleTransform;

    osg::ref_ptr<osg::StateSet> _selectedHl, _intersectedHl, _oldHl;
    bool _standardHL = true;
    bool _intersectionEnabled = false;

    GeoDataLoader *gdl = nullptr;
};

namespace fs = std::filesystem;

#ifndef RAD_TO_DEG
#define RAD_TO_DEG 57.295779513082321
#define DEG_TO_RAD .017453292519943296
#endif
class skyEntry
{
public:
    enum skyType
    {
        texture = 0,
        geometry
    };
    skyEntry(const std::string &n, const std::string &fn, double lon, double lat);
    ~skyEntry();
    skyEntry(const skyEntry &se);
    std::string name;
    std::string fileName;
    osg::ref_ptr<osg::Node> skyNode;
    osg::ref_ptr<osg::Texture2D> skyTexture;
    skyType type = geometry;
    double skyLongitude;
    double skyLatitude;
    double skyTrueNorth;
};

class EditInfo
{
    std::string fileName;
};

class GeoDataLoader : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    struct DatasetInfo
    {
        std::string name;
        double easting;
        double northing;
        double altitude;
        double trueNorth;
    };

    GeoDataLoader();
    bool init();
    ~GeoDataLoader();

    osg::ref_ptr<osg::Node> loadTerrain(std::string filename, osg::Vec3d localOffset);

    static GeoDataLoader *instance();
    osg::Vec3 rootOffset { 0.0, 0.0, 0.0 };
    float trueNorthDegree = 0.0f;
    float NorthAngle;
    osg::ref_ptr<osg::Geode> TexturedSphere;
    osg::TexMat *texMat;
    opencover::coVRShader *shader = nullptr;
    opencover::coVRUniform *topUniform = nullptr;
    opencover::coVRUniform *bottomUniform = nullptr;
    opencover::coVRUniform *floorColorUniform = nullptr;
    editTerrain *editInteraction;

    struct geoLocation
    {
        double latitude;
        double longitude;
        double easting;
        double northing;
        double altitude;
        std::string displayName;
    };

    virtual bool update();
    virtual void message(int toWhom, int type, int length, const void *data);
    void setSky(int num);
    void setSky(std::string fileName);
    void setTop(float t);
    void setBottom(float b);
    void setFloorColor(osg::Vec4 fc);
    void setRootTransform(const osg::Vec3 &offset, float trueNorthDeg);
    std::optional<geoLocation> parseCoordinates(const std::string &jsonData);
    void jumpToLocation(const osg::Vec3d &worldPos);

    const std::vector<DatasetInfo> &getDatasets() const { return datasets; }

    void doInteraction();

private:
    static GeoDataLoader *s_instance;
    PJ_CONTEXT *ProjContext;
    PJ *ProjInstance;

    osg::ref_ptr<osg::MatrixTransform> rootNode;
    osg::ref_ptr<osg::MatrixTransform> skyRootNode;
    osg::ref_ptr<osg::Node> currentSkyNode;
    osg::ref_ptr<osg::Node> skyNode;
    osg::ref_ptr<osg::Node> terrainNode = nullptr;
    osg::ref_ptr<osg::Node> buildingNode = nullptr;
    std::map<std::string, osg::ref_ptr<osg::Node>> loadedTerrains;
    std::map<std::string, osg::ref_ptr<osg::Node>> loadedBuildings;
    bool showTerrain = true;
    bool showBuildings = true;
    osg::Node *oldIntersectedNode;
    std::list<EditInfo> editInfos;
    void doDelete();
    void doReplace();
    void doUndo();

    opencover::ui::Menu *geoDataMenu;
    opencover::ui::Group *geoObjectGroup;
    opencover::ui::Group *skyGroup;
    opencover::ui::Group *originGroup;
    opencover::ui::Group *locationGroup;
    opencover::ui::Group *visibilityGroup;
    opencover::ui::Group *editGroup;

    opencover::ui::Button *editButton;
    opencover::ui::Action *deleteSelected;
    opencover::ui::Action *replace;
    opencover::ui::Action *undo;
    opencover::ui::EditField *selectionRadius;
    opencover::ui::Label *selectionName;


    opencover::ui::Button *terrainVisibilityButton;
    opencover::ui::Button *buildingVisibilityButton;
    opencover::ui::Button *skyButton;
    opencover::ui::Button *applyOffset;
    opencover::ui::Button *saveOffsetToConfig;
    opencover::ui::EditField *location;
    opencover::ui::EditField *easting;
    opencover::ui::EditField *northing;
    opencover::ui::EditField *altitude;
    opencover::ui::EditField *trueNorth;
    opencover::ui::SelectionList *datasetList;
    opencover::ui::SelectionList *skys;
    opencover::ui::Slider *skyNorthSlider = nullptr;
    std::vector<skyEntry> skyEntries;

    std::vector<DatasetInfo> datasets;

    float northAngle;
    std::string terrainFile;
    std::string skyPath;
    int defaultSky;

    std::string tempEastingText;
    std::string tempNorthingText;
    std::string tempAltitudeText;
    std::string tempTrueNorthText;
};
#endif
