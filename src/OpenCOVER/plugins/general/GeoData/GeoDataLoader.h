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
#include <cover/coVRLabel.h>
#include <osg/Texture2D>
#include <string>
#include <optional>
#include <mutex>
#include <set>
#include <vector>
#include <osg/Node>
#include <osg/NodeVisitor>
#include <osg/FrameStamp>
#include <osg/Referenced>
#include <osg/ref_ptr>
#include <osg/PositionAttitudeTransform>
#include <osgTerrain/Terrain>
#include <osg/PagedLOD>
#include <osg/TexMat>
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

namespace vrui
{
class coNavInteraction;
}

class EditInfo
{
    std::string fileName;
};

#include "PlaceLabel.h"

struct regionEntry
{
    std::string region_name;
    std::string terrain_path;
    std::string lod_path;
    std::string labels_path;
};

struct PlaceLabelGroup
{
    osg::ref_ptr<osg::Node> node;
    std::vector<std::shared_ptr<PlaceLabel>> labels;
};

class GeoDataLoader : public opencover::coVRPlugin,
                      public opencover::ui::Owner
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

    osg::ref_ptr<osg::Node> loadTerrain(std::string filename, bool isTerrain = true);

    static GeoDataLoader *instance();

    virtual void message(int toWhom, int type, int length, const void *data);

    const std::vector<DatasetInfo> &getDatasets() const { return datasets; }

    void doInteraction();
    virtual bool update();

    std::string requestNodeFile(const std::string &fileName, osg::NodePath &nodePath, float priority,
                                const osg::FrameStamp *framestamp, osg::ref_ptr<osg::Referenced> &databaseRequest,
                                const osg::Referenced *options) override;

    void setRegionEnabled(const std::string &region_name, bool enabled);
    void setAllRegionsEnabled(bool enabled);

    void setShowBuildings(bool state);
    void setShowTerrain(bool state);
    void setShowLabels(bool state);
    void setAutoViewDistance(bool state);
    void setTilePickingEnabled(bool state);

    std::optional<PlaceLabelGroup> loadLabels(const std::string &file);

private:
    void applyOffset();

    static GeoDataLoader *s_instance;
    PJ_CONTEXT *ProjContext;
    PJ *ProjInstance;

    std::map<std::string, osg::ref_ptr<osg::Node>> loadedTerrains;
    std::map<std::string, osg::ref_ptr<osg::Node>> loadedBuildings;
    std::map<std::string, PlaceLabelGroup> loadedLabels;

    std::map<std::string, regionEntry> regions;
    std::map<std::string, opencover::ui::Button *> regionButtons;

    std::map<std::string, std::string> replacements;
    mutable std::mutex replacementsMutex;
    std::set<std::string> omittedTiles;
    bool showTerrain = true;
    bool showBuildings = true;
    bool showLabels = false;

    std::list<EditInfo> editInfos;

    struct TileSelection
    {
        std::string name;
        std::string fileName;
        osg::ref_ptr<osg::PagedLOD> pagedLod;
        osg::ref_ptr<osg::Node> contentNode;
        unsigned int childIndex = 0;
        osg::ref_ptr<osg::Node> originalChild;
        unsigned int visibleNodeMask = ~0u;
        bool omitted = false;
        bool selected = false;
        osg::ref_ptr<osg::Node> undoChild;
        std::string undoFileName;
        bool undoOmitted = false;
        bool undoSelected = false;
        unsigned int undoNodeMask = ~0u;
        bool hasUndo = false;
    };
    std::vector<TileSelection> tiles;
    int selectedTile = -1;
    bool tilePickingEnabled = false;
    vrui::coNavInteraction *tileSelectionInteraction = nullptr;
    void selectTile(const osg::NodePath &hitPath);
    void clearTileSelections();
    void toggleTileSelection(int index);
    void updateTileSelectionHighlight(TileSelection &tile);
    void stageTileReplacement(const std::string &fileName, const std::string &replacementFileName);
    std::optional<std::string> replacementFor(const std::string &fileName) const;
    void applyTileEditRules(osg::Node *node);
    void doDelete();
    void doReplace();
    void doUndo();

    opencover::ui::Group *editGroup;

    opencover::ui::Button *editButton;
    opencover::ui::Action *deleteSelected;
    opencover::ui::Action *replace;
    opencover::ui::Action *undo;
    opencover::ui::EditField *selectionRadius;
    opencover::ui::Label *selectionName;
    opencover::ui::SelectionList *tileList;
    opencover::ui::EditField *replacementFile;
    opencover::ui::Action *exportTile;
    opencover::ui::Button *saveTileReplacementsToConfig;

    opencover::ui::Menu *geoDataMenu;

    opencover::ui::Group *visibilityGroup;
    opencover::ui::Button *terrainVisibilityButton;
    opencover::ui::Button *buildingsVisibilityButton;
    opencover::ui::Button *labelsVisibilityButton;
    opencover::ui::Button *autoViewDistanceButton;

    opencover::ui::Group *originGroup;
    opencover::ui::SelectionList *datasetList;
    opencover::ui::Button *saveOffsetToConfig;
    opencover::ui::EditField *easting;
    opencover::ui::EditField *northing;
    opencover::ui::EditField *altitude;
    opencover::ui::EditField *trueNorth;

    opencover::ui::Group *geoObjectGroup;

    std::vector<DatasetInfo> datasets;

    float northAngle;
    std::string terrainFile;

    std::string tempEastingText;
    std::string tempNorthingText;
    std::string tempAltitudeText;
    std::string tempTrueNorthText;

    bool autoViewDistance = false;
};
#endif
