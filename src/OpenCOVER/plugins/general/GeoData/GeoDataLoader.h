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
#include <proj.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Button.h>
#include <cover/ui/Label.h>
#include <cover/ui/SelectionList.h>
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

class EditInfo
{
    std::string fileName;
};

struct regionEntry
{
    std::string region_name;
    std::string terrain_path;
    std::string lod_path;
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
    void setRootTransform(const osg::Vec3 &origin, float trueNorthDeg = 0.f);

    static GeoDataLoader *instance();
    osg::Vec3 rootOffset { 0.0, 0.0, 0.0 };
    float trueNorthDegree = 0.0f;
    editTerrain *editInteraction;

    virtual void message(int toWhom, int type, int length, const void *data);

    const std::vector<DatasetInfo> &getDatasets() const { return datasets; }

    void doInteraction();
    bool update();

    void setRegionEnabled(const std::string &region_name, bool enabled);
    void setAllRegionsEnabled(bool enabled);

    void setShowBuildings(bool state);
    void setShowTerrain(bool state);

private:
    void applyOffset();

    static GeoDataLoader *s_instance;
    PJ_CONTEXT *ProjContext;
    PJ *ProjInstance;

    osg::ref_ptr<osg::Node> terrainNode = nullptr;
    osg::ref_ptr<osg::Node> buildingNode = nullptr;
    std::map<std::string, osg::ref_ptr<osg::Node>> loadedTerrains;
    std::map<std::string, osg::ref_ptr<osg::Node>> loadedBuildings;
    std::map<std::string, regionEntry> regions;
    std::map<std::string, opencover::ui::Button *> regionButtons;
    bool showTerrain = true;
    bool showBuildings = true;
    osg::Node *oldIntersectedNode;
    std::list<EditInfo> editInfos;
    void doDelete();
    void doReplace();
    void doUndo();

    opencover::ui::Menu *geoDataMenu;
    opencover::ui::Group *geoObjectGroup;
    opencover::ui::Group *originGroup;
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
    opencover::ui::Button *applyOffsetButton;
    opencover::ui::Button *saveOffsetToConfig;
    opencover::ui::EditField *easting;
    opencover::ui::EditField *northing;
    opencover::ui::EditField *altitude;
    opencover::ui::EditField *trueNorth;
    opencover::ui::SelectionList *datasetList;

    std::vector<DatasetInfo> datasets;

    float northAngle;
    std::string terrainFile;

    std::string tempEastingText;
    std::string tempNorthingText;
    std::string tempAltitudeText;
    std::string tempTrueNorthText;
};
#endif
