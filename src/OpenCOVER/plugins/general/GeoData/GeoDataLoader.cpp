/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2024 HLRS  **
 **                                                                          **
 ** Description: GeoDataLoader Plugin                                        **
 **                                                                          **
 **                                                                          **
 ** Author: Uwe Woessner 		                                              **
 **                                                                          **
 ** History:  								                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "GeoDataLoader.h"

#include <algorithm>
#include <iterator>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coIntersection.h>
#include <OpenVRUI/coNavInteraction.h>
#include <OpenVRUI/coInteractionManager.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <memory>
#include <osg/ref_ptr>

#include <gdal.h>
#include <ogrsf_frmts.h>

#include <vrml97/vrml/VrmlNamespace.h>
#include <VrmlNodeGeoData.h>
#include <cover/ui/Manager.h>

#include <config/CoviseConfig.h>
#include <geodata/GeoData.h>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osg/Node>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/CullFace>
#include <osg/Texture2D>
#include <osg/TexMat>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/LOD>
#include <osg/ShapeDrawable>
#include <osgUtil/Optimizer>
#include <osgTerrain/Terrain>
#include <osgViewer/Renderer>
#include <iostream>
#include <cover/coVRLabel.h>
#include <string>
#include "PlaceLabel.h"
#include "cover/coVRConfig.h"
#include "VrmlNodeGeoreferenced.h"
#include <PluginUtil/PluginMessageTypes.h>
#include <stdio.h>
#include <utility>

using namespace vrui;
using namespace opencover;

// #define GEODATA_VERBOSE

GeoDataLoader *GeoDataLoader::s_instance = nullptr;

GeoDataLoader::GeoDataLoader()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("GeoData", cover->ui)
{
    assert(s_instance == nullptr);
    s_instance = this;

    VrmlNamespace::addBuiltIn(VrmlNode::defineType<VrmlNodeGeoreferenced>());
}
bool GeoDataLoader::init()
{
    vrml::VrmlNamespace::addBuiltIn(vrml::VrmlNode::defineType<VrmlNodeGeoData>());

    geoDataMenu = dynamic_cast<ui::Menu *>(cover->ui->getByPath("Manager.GeoData"));
    if (!geoDataMenu)
    {
        geoDataMenu = new ui::Menu("GeoData", cover->ui);
        geoDataMenu->setText("GeoData");
        geoDataMenu->allowRelayout(true);
    }
    geoDataMenu->setVisible(true);

    auto configFile = config();

    originGroup = new ui::Group(geoDataMenu, "originGroup");
    originGroup->setText("Origin");
    originGroup->allowRelayout(true);

    // selection list for offsets for different datasets
    datasetList = new ui::SelectionList(originGroup, "datasets");
    datasetList->setText("Choose Datasets");
    datasetList->append("None");
    datasets.clear();

    auto datasetEntries = configFile->array<opencover::config::Section>("", "datasets");

    for (size_t i = 0; i < datasetEntries->size(); i++)
    {
        opencover::config::Section entry = (*datasetEntries)[i];

        DatasetInfo dataset;
        dataset.name = entry.value<std::string>("", "name")->value();

        double altitude = entry.value<double>("", "altitude")->value();
        double trueNorth = entry.value<double>("", "trueNorth")->value();
        double latitude = entry.value<double>("", "latitude")->value();
        double longitude = entry.value<double>("", "longitude")->value();

        if (latitude || longitude)
        {
            auto enu = GeoData::instance()->globalToReference(osg::Vec3d(longitude, latitude, altitude));
            dataset.easting = enu.x();
            dataset.northing = enu.y();
            dataset.altitude = enu.z();
        }
        else
        {
            dataset.easting = entry.value<double>("", "easting")->value();
            dataset.northing = entry.value<double>("", "northing")->value();
            dataset.altitude = altitude;
            dataset.trueNorth = trueNorth;
        }

        datasets.push_back(dataset);
        datasetList->append(dataset.name);
    }

    auto replacementEntries = configFile->array<opencover::config::Section>("", "replacements");

    for (size_t i = 0; i < replacementEntries->size(); i++)
    {
        opencover::config::Section entry = (*replacementEntries)[i];

        std::string fileName = entry.value<std::string>("", "fileName")->value();
        std::string replacementFileName = entry.value<std::string>("", "replaceBy")->value();
        std::lock_guard<std::mutex> lock(replacementsMutex);
        replacements[fileName] = replacementFileName;
    }

    datasetList->select(0, true); // select "None" by default
    datasetList->setCallback([this](int selection)
        {
        if (selection == 0) // "None"
        {
            easting->setValue(0.0);
            northing->setValue(0.0);
            altitude->setValue(0.0);
            trueNorth->setValue(0.0);

            tempEastingText = "0.0";
            tempNorthingText = "0.0";
            tempAltitudeText = "0.0";
            tempTrueNorthText = "0.0";
        } 
        else
        {
            int index = selection - 1;
            if (index < 0 || index >= datasets.size())
            {
                easting->setValue(0.0);
                northing->setValue(0.0);
                altitude->setValue(0.0);
                trueNorth->setValue(0.0);

                tempEastingText = "0.0";
                tempNorthingText = "0.0";
                tempAltitudeText = "0.0";
                tempTrueNorthText = "0.0";
            }
            else
            {
                const DatasetInfo& dataset = datasets[index];
                easting->setValue(dataset.easting);
                northing->setValue(dataset.northing);
                altitude->setValue(dataset.altitude);
                trueNorth->setValue(dataset.trueNorth);
                tempEastingText = std::to_string(dataset.easting);
                tempNorthingText = std::to_string(dataset.northing);
                tempAltitudeText = std::to_string(dataset.altitude);
                tempTrueNorthText = std::to_string(dataset.trueNorth);
            }
        }
        applyOffset(); });

    tempEastingText = "0.0";
    tempNorthingText = "0.0";
    tempAltitudeText = "0.0";
    tempTrueNorthText = "0.0";

    easting = new ui::EditField(originGroup, "easting");
    easting->setText("Easting (m):");
    easting->setCallback([this](std::string val)
        {
        this->tempEastingText = val;
        applyOffset(); });

    northing = new ui::EditField(originGroup, "northing");
    northing->setText("Northing (m):");
    northing->setCallback([this](std::string val)
        {
        this->tempNorthingText = val;
        applyOffset(); });

    altitude = new ui::EditField(originGroup, "altitude");
    altitude->setText("Altitude (m):");
    altitude->setCallback([this](std::string val)
        {
        this->tempAltitudeText = val;
        applyOffset(); });

    trueNorth = new ui::EditField(originGroup, "trueNorth");
    trueNorth->setText("True North (°):");
    trueNorth->setCallback([this](std::string val)
        {
        this->tempTrueNorthText = val;
        applyOffset(); });

    // TODO add option to rename dataset either with edit field or with editable selection list
    saveOffsetToConfig = new ui::Button(originGroup, "saveOffsetToConfig");
    saveOffsetToConfig->setText("Save to Config");
    saveOffsetToConfig->setCallback([this](bool state)
        {
            if (!state) return;

            auto configFile = config();
            if (!configFile) return;

            int selectedDataset = datasetList->selectedIndex();
            int datasetIndex = 0;
            std::string newName;

            if (selectedDataset <= 0) // "None" selected, save as new dataset
            {
                newName = "Dataset_" + std::to_string(datasets.size() + 1);
                datasetIndex = datasets.size();
            }
            else // existing dataset selected, overwrite
            {
                datasetIndex = selectedDataset - 1; // adjust for "None" entry
                newName = datasets[datasetIndex].name;
            }

            std::string baseKey = "datasets[" + std::to_string(datasetIndex) + "]";
            try
            {
                double newEasting  = tempEastingText.empty()  ? 0.0 : std::stod(tempEastingText);
                double newNorthing = tempNorthingText.empty() ? 0.0 : std::stod(tempNorthingText);
                double newAltitude = tempAltitudeText.empty() ? 0.0 : std::stod(tempAltitudeText);
                double newNorth    = tempTrueNorthText.empty() ? 0.0 : std::stod(tempTrueNorthText);

                auto nameValuePtr = configFile->value<std::string>(baseKey, "name", "");
                *nameValuePtr = newName;

                *configFile->value<double>(baseKey, "easting", 0.0) = newEasting;
                *configFile->value<double>(baseKey, "northing", 0.0) = newNorthing;
                *configFile->value<double>(baseKey, "altitude", 0.0) = newAltitude;
                *configFile->value<double>(baseKey, "trueNorth", 0.0) = newNorth;
                *configFile->value<double>(baseKey, "latitude", 0.0) = 0.0;
                *configFile->value<double>(baseKey, "longitude", 0.0) = 0.0;

                if (configFile->save())
                {
                    if (selectedDataset <= 0) // "None" selected, add new entry to list
                    {
                        DatasetInfo newDs;
                        newDs.name = newName;
                        newDs.easting = newEasting;
                        newDs.northing = newNorthing;
                        newDs.altitude = newAltitude;
                        newDs.trueNorth = newNorth;
                        
                        datasets.push_back(newDs);
                        datasetList->append(newName);
                        datasetList->select(datasets.size());
                        std::cerr << "GeoData: Created NEW dataset: " << newName << std::endl;
                    }
                    else // existing dataset selected, update values
                    {
                        datasets[datasetIndex].easting = newEasting;
                        datasets[datasetIndex].northing = newNorthing;
                        datasets[datasetIndex].altitude = newAltitude;
                        datasets[datasetIndex].trueNorth = newNorth;
                        std::cerr << "GeoData: Updated dataset: " << newName << std::endl;
                    }
                }
                else
                {
                    std::cerr << "GeoData: Failed to save config file!" << std::endl;
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "GeoData Error: Invalid input" << std::endl;
            }

            saveOffsetToConfig->setState(false); });

    visibilityGroup = new ui::Group(geoDataMenu, "visibility");
    visibilityGroup->setText("Toggle Visibility");
    visibilityGroup->allowRelayout(true);

    terrainVisibilityButton = new ui::Button(visibilityGroup, "terrainVisibility");
    terrainVisibilityButton->setText("Terrain");
    terrainVisibilityButton->setState(showTerrain);
    terrainVisibilityButton->setCallback([this](bool state)
        { setShowTerrain(state); });

    buildingsVisibilityButton = new ui::Button(visibilityGroup, "buildingVisibility");
    buildingsVisibilityButton->setText("Buildings");
    buildingsVisibilityButton->setState(showBuildings);
    buildingsVisibilityButton->setCallback([this](bool state)
        { setShowBuildings(state); });

    labelsVisibilityButton = new ui::Button(visibilityGroup, "labelsVisibility");
    labelsVisibilityButton->setText("Labels");
    labelsVisibilityButton->setState(showLabels);
    labelsVisibilityButton->setCallback([this](bool state)
        { setShowLabels(state); });

    autoViewDistanceButton = new ui::Button(visibilityGroup, "autoViewDistance");
    autoViewDistanceButton->setText("Automatic view distance");
    autoViewDistanceButton->setState(autoViewDistance);
    autoViewDistanceButton->setCallback([this](bool state)
        { setAutoViewDistance(state); });

    // create Button for each region in config
    geoObjectGroup = new ui::Group(geoDataMenu, "GeoObjects");
    geoObjectGroup->setText("Geo-Objects");
    geoObjectGroup->allowRelayout(true);

    auto terrainEntries = configFile->array<opencover::config::Section>("", "regions");

    for (size_t i = 0; i < terrainEntries->size(); i++)
    {

        opencover::config::Section terrainEntry = (*terrainEntries)[i];

        std::string region_name = terrainEntry.value<std::string>("", "name")->value();
        std::string terrain_path = terrainEntry.value<std::string>("", "terrainPath")->value();
        std::string lod_path = terrainEntry.value<std::string>("", "lodPath")->value();
        std::string labels_path = terrainEntry.value<std::string>("", "labelsPath")->value();

        regions.emplace(region_name, regionEntry { region_name, terrain_path, lod_path, labels_path });

        if (terrain_path != "" || lod_path != "" || labels_path != "")
        {
            ui::Button *regionButton = new ui::Button(geoObjectGroup, region_name);
            regionButton->setText(region_name);
            regionButton->setState(false);
            regionButton->setCallback([this, region_name](bool state)
                { setRegionEnabled(region_name, state); });
            regionButtons[region_name] = regionButton;
        }
    }

    editGroup = new ui::Group(geoDataMenu, "edit");
    editGroup->setText("Edit");
    editGroup->allowRelayout(true);
    tileSelectionInteraction = new vrui::coNavInteraction(vrui::coInteraction::ButtonA,
        "GeoDataTileSelection", vrui::coInteraction::High);
    editButton = new ui::Button(editGroup, "editTerrain");
    editButton->setText("editTerrain");
    editButton->setState(false);
    editButton->setCallback([this](bool state)
        {
        if (state)
        {
            setTilePickingEnabled(true);
        }
        else
        {
            setTilePickingEnabled(false);
            } });

    tileList = new ui::SelectionList(editGroup, "tiles");
    tileList->setText("Selected tiles");
    tileList->append("No tile selected");
    tileList->select(0, true);
    tileList->setCallback([this](int index)
        {
            if (index > 0 && index <= static_cast<int>(tiles.size()))
            {
                selectedTile = index - 1;
                auto &tile = tiles[selectedTile];
                selectionName->setText(tile.name);
            }
        });

    replacementFile = new ui::EditField(editGroup, "replacementFile");
    replacementFile->setText("Replacement tile file:");

    saveTileReplacementsToConfig = new ui::Button(editGroup, "saveTileReplacementsToConfig");
    saveTileReplacementsToConfig->setText("Save tile replacements to Config");
    saveTileReplacementsToConfig->setState(false);
    saveTileReplacementsToConfig->setCallback([this](bool state)
        {
            if (!state)
                return;
            auto configFile = config();
            if (configFile && configFile->save())
                std::cerr << "GeoData: Saved tile replacements to config" << std::endl;
            else
                std::cerr << "GeoData: Failed to save tile replacements to config" << std::endl;
            saveTileReplacementsToConfig->setState(false);
        });

    deleteSelected = new ui::Action(editGroup, "deleteSelected");
    deleteSelected->setText("Omit selected tile");
    deleteSelected->setCallback([this]()
        { doDelete(); });

    replace = new ui::Action(editGroup, "replace");
    replace->setText("Replace selected tile");
    replace->setCallback([this]()
        { doReplace(); });

    undo = new ui::Action(editGroup, "undo");
    undo->setText("undo");
    undo->setCallback([this]()
        { doUndo(); });

    exportTile = new ui::Action(editGroup, "exportTile");
    exportTile->setText("Export selected tile to FBX");
    exportTile->setCallback([this]()
        {
            if (selectedTile < 0)
                return;
            auto &tile = tiles[selectedTile];
            std::string output = tile.fileName;
            const auto dot = output.find_last_of('.');
            const auto slash = output.find_last_of("/\\");
            if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
                output.erase(dot);
            output += ".fbx";
            if (!tile.pagedLod || tile.childIndex >= tile.pagedLod->getNumChildren())
                return;
            auto selectionManager = coVRSelectionManager::instance();
            if (tile.selected && tile.contentNode)
                selectionManager->removeNode(tile.contentNode.get());
            osg::Node *node = tile.pagedLod->getChild(tile.childIndex);
            const bool exported = osgDB::writeNodeFile(*node, output);
            updateTileSelectionHighlight(tile);
            if (!exported)
                std::cerr << "GeoData: Failed to export selected tile to FBX: " << output << std::endl;
        });

    selectionName = new ui::Label(editGroup, "name");
    selectionName->setText("nothing is selected");

    return true;
}

// this is called if the plugin is removed at runtime
GeoDataLoader::~GeoDataLoader()
{
    if (tilePickingEnabled)
        setTilePickingEnabled(false);
    clearTileSelections();
    delete tileSelectionInteraction;
    tileSelectionInteraction = nullptr;
    s_instance = nullptr;

    delete geoObjectGroup;
    delete originGroup;
    delete visibilityGroup;
    geoDataMenu->setVisible(geoDataMenu->numChildren() > 0);
}

void GeoDataLoader::setRegionEnabled(const std::string &region_name, bool enabled)
{
    const auto &it = regions.find(region_name);
    if (it == regions.end())
    {
        // No region found with this name
        return;
    }
    const auto &region = it->second;

    auto terrainRoot = GeoData::instance()->terrainRoot();

    if (enabled)
    {
        if (loadedTerrains.find(region_name) == loadedTerrains.end() && !region.terrain_path.empty())
        {
            auto terrainNode = loadTerrain(region.terrain_path, true);
            if (terrainNode)
            {
                loadedTerrains[region_name] = terrainNode;
                terrainRoot->addChild(terrainNode);
                terrainNode->setName(region_name);
                terrainNode->setNodeMask(showTerrain ? 0xffffffff : 0x0);
            }
        }

        if (loadedBuildings.find(region_name) == loadedBuildings.end() && !region.lod_path.empty())
        {
            auto buildingNode = loadTerrain(region.lod_path, false);
            if (buildingNode)
            {
                loadedBuildings[region_name] = buildingNode;
                terrainRoot->addChild(buildingNode);
                buildingNode->setName(region_name);
                buildingNode->setNodeMask(showBuildings ? 0xffffffff : 0x0);
            }
        }

        if (loadedLabels.find(region_name) == loadedLabels.end() && !region.labels_path.empty())
        {
            auto labelGroup = loadLabels(region.labels_path);
            if (labelGroup)
            {
                loadedLabels[region_name] = labelGroup.value();
                terrainRoot->addChild(labelGroup->node);
                labelGroup->node->setName(region_name);
                labelGroup->node->setNodeMask(showLabels ? 0xffffffff : 0x0);
            }
        }
    }
    else
    {
        if (loadedTerrains.find(region_name) != loadedTerrains.end())
        {
            terrainRoot->removeChild(loadedTerrains[region_name]);
            loadedTerrains.erase(region_name);
        }

        if (loadedBuildings.find(region_name) != loadedBuildings.end())
        {
            terrainRoot->removeChild(loadedBuildings[region_name]);
            loadedBuildings.erase(region_name);
        }

        if (loadedLabels.find(region_name) != loadedLabels.end())
        {
            terrainRoot->removeChild(loadedLabels[region_name].node);
            loadedLabels.erase(region_name);
        }
    }

    // We can assume that the button exists, because the region was found above.
    regionButtons[region_name]->setState(enabled);
}

void GeoDataLoader::setAllRegionsEnabled(bool enabled)
{
    for (auto &[region_name, _] : regions)
    {
        setRegionEnabled(region_name, enabled);
    }
}

void GeoDataLoader::setShowBuildings(bool state)
{
    showBuildings = state;
    buildingsVisibilityButton->setState(state);
    for (const auto &[_, buildings] : loadedBuildings)
    {
        buildings->setNodeMask(state ? 0xffffffff : 0x0);
    }
}

void GeoDataLoader::setShowTerrain(bool state)
{
    showTerrain = state;
    terrainVisibilityButton->setState(state);
    for (const auto &[_, terrain] : loadedTerrains)
    {
        terrain->setNodeMask(state ? 0xffffffff : 0x0);
    }
}

void GeoDataLoader::setShowLabels(bool state)
{
    showLabels = state;
    labelsVisibilityButton->setState(state);
    for (const auto &[_, labelGroup] : loadedLabels)
    {
        labelGroup.node->setNodeMask(state ? 0xffffffff : 0x0);
    }
}

void GeoDataLoader::setAutoViewDistance(bool state)
{
    autoViewDistance = state;
    autoViewDistanceButton->setState(state);
}

void GeoDataLoader::setTilePickingEnabled(bool state)
{
    if (tilePickingEnabled == state)
        return;
    tilePickingEnabled = state;
    if (state)
    {
        coIntersection::instance()->isectAllNodes(true);
        if (tileSelectionInteraction && !tileSelectionInteraction->isRegistered())
            coInteractionManager::the()->registerInteraction(tileSelectionInteraction);
    }
    else
    {
        if (tileSelectionInteraction && tileSelectionInteraction->isRegistered())
            coInteractionManager::the()->unregisterInteraction(tileSelectionInteraction);
        coIntersection::instance()->isectAllNodes(false);
    }
}

void GeoDataLoader::clearTileSelections()
{
    auto *selectionManager = coVRSelectionManager::instance();
    bool changed = false;
    for (auto &tile : tiles)
    {
        if (tile.selected && tile.contentNode)
        {
            selectionManager->removeNode(tile.contentNode.get());
            tile.selected = false;
            changed = true;
        }
    }
    if (changed)
    {
        selectionManager->selectionChanged();
        selectionManager->pickedObjChanged();
    }
}

void GeoDataLoader::toggleTileSelection(int index)
{
    if (index < 0 || index >= static_cast<int>(tiles.size()))
        return;
    auto &tile = tiles[index];
    if (!tile.pagedLod || tile.childIndex >= tile.pagedLod->getNumChildren())
        return;

    auto *selectionManager = coVRSelectionManager::instance();
    if (tile.selected)
    {
        selectionManager->removeNode(tile.contentNode.get());
        tile.selected = false;
        selectionManager->selectionChanged();
        selectionManager->pickedObjChanged();
        return;
    }

    osg::Node *node = tile.pagedLod->getChild(tile.childIndex);
    tile.contentNode = node;
    selectionManager->setSelectionWire(0);
    selectionManager->setSelectionColor(1.0f, 0.0f, 0.0f);
    selectionManager->addSelection(tile.pagedLod.get(), node);
    tile.selected = true;
    selectionManager->pickedObjChanged();
}

void GeoDataLoader::updateTileSelectionHighlight(TileSelection &tile)
{
    if (!tile.selected || !tile.pagedLod || tile.childIndex >= tile.pagedLod->getNumChildren())
        return;
    auto *selectionManager = coVRSelectionManager::instance();
    if (tile.contentNode)
        selectionManager->removeNode(tile.contentNode.get());
    osg::Node *node = tile.pagedLod->getChild(tile.childIndex);
    tile.contentNode = node;
    selectionManager->setSelectionWire(0);
    selectionManager->setSelectionColor(1.0f, 0.0f, 0.0f);
    selectionManager->addSelection(tile.pagedLod.get(), node);
    selectionManager->pickedObjChanged();
}

void GeoDataLoader::stageTileReplacement(const std::string &fileName, const std::string &replacementFileName)
{
    auto configFile = config();
    if (!configFile)
        return;

    auto entries = configFile->array<opencover::config::Section>("", "replacements");
    size_t entryIndex = entries->size();
    for (size_t i = 0; i < entries->size(); ++i)
    {
        opencover::config::Section entry = (*entries)[i];
        if (entry.value<std::string>("", "fileName")->value() == fileName)
        {
            entryIndex = i;
            break;
        }
    }

    const std::string section = "replacements[" + std::to_string(entryIndex) + "]";
    *configFile->value<std::string>(section, "fileName", "") = fileName;
    *configFile->value<std::string>(section, "replaceBy", "") = replacementFileName;
    {
        std::lock_guard<std::mutex> lock(replacementsMutex);
        replacements[fileName] = replacementFileName;
    }
}

std::optional<std::string> GeoDataLoader::replacementFor(const std::string &fileName) const
{
    std::lock_guard<std::mutex> lock(replacementsMutex);
    for (const auto &replacement : replacements)
    {
        if (!replacement.first.empty() && fileName.find(replacement.first) != std::string::npos)
            return replacement.second;
    }
    return std::nullopt;
}

std::string GeoDataLoader::requestNodeFile(const std::string &fileName, osg::NodePath &nodePath, float priority,
                                           const osg::FrameStamp *framestamp,
                                           osg::ref_ptr<osg::Referenced> &databaseRequest,
                                           const osg::Referenced *options)
{
    (void)nodePath;
    (void)priority;
    (void)framestamp;
    (void)databaseRequest;
    (void)options;
    if (auto replacement = replacementFor(fileName))
    {
        if (replacement->empty())
            {
            return "/data/empty.obj";        
        }
        return *replacement;
    }
    return fileName;
}

void GeoDataLoader::selectTile(const osg::NodePath &hitPath)
{
    for (int pathIndex = static_cast<int>(hitPath.size()) - 2; pathIndex >= 0; --pathIndex)
    {
        auto *pagedLod = dynamic_cast<osg::PagedLOD *>(hitPath[pathIndex]);
        if (!pagedLod)
            continue;
        osg::Node *child = hitPath[pathIndex + 1];
        unsigned int childIndex = 0;
        while (childIndex < pagedLod->getNumChildren() && pagedLod->getChild(childIndex) != child)
            ++childIndex;
        if (childIndex < pagedLod->getNumChildren() && childIndex < pagedLod->getNumFileNames())
        {
            const std::string fileName = pagedLod->getFileName(childIndex);
            if (!fileName.empty())
            {
                auto found = std::find_if(tiles.begin(), tiles.end(), [pagedLod, childIndex](const TileSelection &tile)
                    { return tile.pagedLod == pagedLod && tile.childIndex == childIndex; });
                if (found == tiles.end())
                {
                    TileSelection tile;
                    tile.name = fileName;
                    tile.fileName = fileName;
                    tile.pagedLod = pagedLod;
                    tile.childIndex = childIndex;
                    tile.contentNode = pagedLod->getChild(childIndex);
                    tile.originalChild = tile.contentNode;
                    tile.visibleNodeMask = tile.contentNode->getNodeMask();
                    tiles.push_back(tile);
                    tileList->append(fileName);
                    found = std::prev(tiles.end());
                }

                selectedTile = static_cast<int>(std::distance(tiles.begin(), found));
                tileList->select(selectedTile + 1, true);
                toggleTileSelection(selectedTile);
                selectionName->setText(found->name);
                return;
            }
        }
    }
}

void GeoDataLoader::applyTileEditRules(osg::Node *node)
{
    if (!node)
        return;
    if (auto *pagedLod = dynamic_cast<osg::PagedLOD *>(node))
    {
        for (unsigned int i = 0; i < pagedLod->getNumFileNames(); ++i)
        {
            const std::string fileName = pagedLod->getFileName(i);
            if (omittedTiles.count(fileName))
                pagedLod->setFileName(i, "");
            else if (auto replacement = replacementFor(fileName))
                pagedLod->setFileName(i, *replacement);
        }
    }
    if (auto *group = node->asGroup())
    {
        for (unsigned int i = 0; i < group->getNumChildren(); ++i)
            applyTileEditRules(group->getChild(i));
    }
}

void GeoDataLoader::message(int toWhom, int type, int length, const void *data)
{
    const char *messageData = (const char *)data;
    if (type == PluginMessageTypes::LoadTerrain)
        loadTerrain(messageData + 20, true); // 20= opencover://terrain/
    else if (type == PluginMessageTypes::GeoDataSetRegionEnabled)
        setRegionEnabled(std::string(messageData), true);
}

bool GeoDataLoader::update()
{
    if (autoViewDistance)
    {
        double scale = cover->getScale();
        osg::Matrix currentObjectsTransform = cover->getObjectsXform()->getMatrix();
        double z = std::abs(currentObjectsTransform.getTrans().z()) * scale;

        double farValue = 10 * z;
        farValue = std::clamp(farValue, 1e7, 1e10);

        double nearValue = farValue / 1e6;
        nearValue = std::clamp(nearValue, 10.0, 500.0);
        coVRConfig::instance()->setNearFar(nearValue, farValue);
    }

    if (tilePickingEnabled && tileSelectionInteraction && tileSelectionInteraction->wasStarted()
        && cover->getIntersectedNode())
        selectTile(cover->getIntersectedNodePath());

    return false;
}
GeoDataLoader *GeoDataLoader::instance()
{
    if (!s_instance)
        s_instance = new GeoDataLoader;
    return s_instance;
}

osg::ref_ptr<osg::Node> GeoDataLoader::loadTerrain(std::string filename, bool isTerrain)
{
    VRViewer *viewer = VRViewer::instance();
    osgDB::DatabasePager *pager = viewer->getDatabasePager();

#ifdef GEODATA_VERBOSE
    std::cout << "DatabasePager:" << std::endl;
    std::cout << "\t num threads: " << pager->getNumDatabaseThreads() << std::endl;
    std::cout << "\t getDatabasePagerThreadPause: " << (pager->getDatabasePagerThreadPause() ? "yes" : "no") << std::endl;
    std::cout << "\t getDoPreCompile: " << (pager->getDoPreCompile() ? "yes" : "no") << std::endl;
    std::cout << "\t getApplyPBOToImages: " << (pager->getApplyPBOToImages() ? "yes" : "no") << std::endl;
    std::cout << "\t getTargetMaximumNumberOfPageLOD: " << pager->getTargetMaximumNumberOfPageLOD() << std::endl;
#endif

    viewer->setIncrementalCompileOperation(new osgUtil::IncrementalCompileOperation());
    // pager->setIncrementalCompileOperation(new osgUtil::IncrementalCompileOperation());
    osgUtil::IncrementalCompileOperation *coop = pager->getIncrementalCompileOperation();
    if (coop)
    {
        for (int screenIt = 0; screenIt < coVRConfig::instance()->numScreens(); ++screenIt)
        {
            osgViewer::Renderer *renderer = dynamic_cast<osgViewer::Renderer *>(coVRConfig::instance()->channels[screenIt].camera->getRenderer());
            if (renderer)
            {
                renderer->getSceneView(0)->setAutomaticFlush(false);
                renderer->getSceneView(1)->setAutomaticFlush(false);
            }
        }

        coop->setTargetFrameRate(60);
        coop->setMinimumTimeAvailableForGLCompileAndDeletePerFrame(0.001);
        coop->setMaximumNumOfObjectsToCompilePerFrame(2);
        coop->setConservativeTimeRatio(0.1);
        coop->setFlushTimeRatio(0.1);

#ifdef GEODATA_VERBOSE
        std::cout << "IncrementedCompileOperation:" << std::endl;
        std::cout << "\t is active: " << (coop->isActive() ? "yes" : "no") << std::endl;
        std::cout << "\t target frame rate: " << coop->getTargetFrameRate() << std::endl;
        std::cout << "\t getMinimumTimeAvailableForGLCompileAndDeletePerFrame: " << coop->getMinimumTimeAvailableForGLCompileAndDeletePerFrame() << std::endl;
        std::cout << "\t getMaximumNumOfObjectsToCompilePerFrame: " << coop->getMaximumNumOfObjectsToCompilePerFrame() << std::endl;
        std::cout << "\t flush time ratio: " << coop->getFlushTimeRatio() << std::endl;
        std::cout << "\t conservative time ratio: " << coop->getConservativeTimeRatio() << std::endl;
#endif
    }

    osg::Node *terrain = nullptr;
    if (auto replacementFileName = replacementFor(filename))
    {
        if (replacementFileName->empty())
            return nullptr;
        terrain = osgDB::readNodeFile(*replacementFileName);
    }
    else
    {
        terrain = osgDB::readNodeFile(filename);
    }

    if (terrain)
    {
        applyTileEditRules(terrain);
        terrain->setDataVariance(osg::Object::DYNAMIC);

        osg::StateSet *terrainStateSet = terrain->getOrCreateStateSet();
        terrainStateSet->setMode(GL_LIGHTING, isTerrain ? osg::StateAttribute::OFF : osg::StateAttribute::ON);
        terrainStateSet->setMode(GL_LIGHT0, isTerrain ? osg::StateAttribute::OFF : osg::StateAttribute::ON);

        if (isTerrain)
        {
            osg::Material *material = new osg::Material();
            material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
            material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1));
            material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0, 0, 1));
            material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0, 0, 1));
            material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0, 0, 1));
            material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
            material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);
            terrainStateSet->setAttribute(material, osg::StateAttribute::ON);
        }

        terrain->setNodeMask(terrain->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    }

    return terrain;
}

void GeoDataLoader::applyOffset()
{
    double originEasting = 0.0;
    double originNorthing = 0.0;
    double originAltitude = 0.0;
    double trueNorth = 0.0;
    if (tempEastingText != "")
    {
        originEasting = std::stod(tempEastingText);
    }
    if (tempNorthingText != "")
    {
        originNorthing = std::stod(tempNorthingText);
    }
    if (tempAltitudeText != "")
    {
        originAltitude = std::stod(tempAltitudeText);
    }
    if (tempTrueNorthText != "")
    {
        trueNorth = std::stod(tempTrueNorthText);
    }

    osg::Vec3 origin = osg::Vec3(originEasting, originNorthing, originAltitude);
    GeoData::instance()->setProjectTransform(origin, trueNorth);
}

std::optional<PlaceLabelGroup> GeoDataLoader::loadLabels(const std::string &file)
{
    GDALAllRegister();

    GDALDatasetUniquePtr dataset(GDALDataset::Open(file.c_str(), GDAL_OF_VECTOR));
    if (dataset == nullptr)
    {
        return std::nullopt;
    }

    std::vector<std::shared_ptr<PlaceLabel>> labels;
    osg::ref_ptr<osg::Group> node = new osg::Group;
    auto font = configString("General", "font", "arial.ttf")->value();

    for (OGRLayer *layer : dataset->GetLayers())
    {
        for (auto &feature : *layer)
        {
            std::string name = feature->GetFieldAsString("name");
            if (name.empty())
                continue;

            const OGRGeometry *geometry = feature->GetGeometryRef();
            if (!geometry)
                continue;

            if (wkbFlatten(geometry->getGeometryType()) != wkbPoint)
                continue;

            const OGRPoint *poPoint = geometry->toPoint();

            double altitude = feature->GetFieldAsDouble("altitude");
            int size = feature->GetFieldAsInteger("size");

            osg::Vec3d global(poPoint->getX(), poPoint->getY(), altitude);
            auto local = GeoData::instance()->globalToReference(global);

            labels.emplace_back(std::make_shared<PlaceLabel>(name, local, node, font, size));
        }
    }

    return PlaceLabelGroup { node, labels };
}

COVERPLUGIN(GeoDataLoader)

void GeoDataLoader::doDelete()
{
    auto *selectionManager = coVRSelectionManager::instance();
    bool selectionChanged = false;
    size_t omittedCount = 0;
    int firstOmittedTile = -1;

    for (size_t i = 0; i < tiles.size(); ++i)
    {
        auto &tile = tiles[i];
        if (!tile.selected || !tile.pagedLod || tile.childIndex >= tile.pagedLod->getNumChildren())
            continue;

        tile.undoSelected = true;
        if (tile.contentNode)
        {
            selectionManager->removeNode(tile.contentNode.get());
            selectionChanged = true;
        }
        tile.selected = false;
        tile.undoChild = tile.pagedLod->getChild(tile.childIndex);
        tile.contentNode = tile.undoChild;
        tile.undoFileName = tile.pagedLod->getFileName(tile.childIndex);
        tile.undoOmitted = tile.omitted;
        tile.undoNodeMask = tile.undoChild->getNodeMask();
        tile.hasUndo = true;
        tile.omitted = true;
        omittedTiles.insert(tile.fileName);
        stageTileReplacement(tile.fileName, "");
        tile.pagedLod->setFileName(tile.childIndex, "");
        tile.pagedLod->getChild(tile.childIndex)->setNodeMask(0);
        if (firstOmittedTile < 0)
            firstOmittedTile = static_cast<int>(i);
        ++omittedCount;
    }

    if (omittedCount == 0)
        return;
    if (selectionChanged)
    {
        selectionManager->selectionChanged();
        selectionManager->pickedObjChanged();
    }
    selectionManager->invalidateHitList();
    selectedTile = firstOmittedTile;
    tileList->select(selectedTile + 1, true);
    selectionName->setText(std::to_string(omittedCount) + " tile(s) omitted; save config to persist");
}
void GeoDataLoader::doReplace()
{
    if (selectedTile < 0 || selectedTile >= static_cast<int>(tiles.size()))
        return;
    auto &tile = tiles[selectedTile];
    const std::string replacementPath = replacementFile->value();
    if (replacementPath.empty() || !tile.pagedLod || tile.childIndex >= tile.pagedLod->getNumChildren())
        return;

    osg::ref_ptr<osg::Node> replacementNode = osgDB::readNodeFile(replacementPath);
    if (!replacementNode)
    {
        std::cerr << "GeoData: Failed to read replacement tile: " << replacementPath << std::endl;
        return;
    }

    auto *selectionManager = coVRSelectionManager::instance();
    tile.undoSelected = tile.selected;
    if (tile.selected && tile.contentNode)
    {
        selectionManager->removeNode(tile.contentNode.get());
        tile.selected = false;
        selectionManager->selectionChanged();
    }
    tile.undoChild = tile.pagedLod->getChild(tile.childIndex);
    tile.contentNode = tile.undoChild;
    tile.undoFileName = tile.pagedLod->getFileName(tile.childIndex);
    tile.undoOmitted = tile.omitted;
    tile.undoNodeMask = tile.undoChild->getNodeMask();
    tile.hasUndo = true;
    replacementNode->setName(tile.name);
    replacementNode->setNodeMask(tile.omitted ? tile.visibleNodeMask : tile.undoNodeMask);
    tile.pagedLod->setChild(tile.childIndex, replacementNode.get());
    tile.pagedLod->setFileName(tile.childIndex, replacementPath);
    tile.contentNode = replacementNode;
    tile.omitted = false;
    tile.selected = tile.undoSelected;
    omittedTiles.erase(tile.fileName);
    {
        std::lock_guard<std::mutex> lock(replacementsMutex);
        replacements[tile.fileName] = replacementPath;
    }
    if (tile.selected)
        updateTileSelectionHighlight(tile);
    tileList->select(selectedTile + 1, true);
    selectionName->setText(tile.name + " (replaced)");
}
void GeoDataLoader::doUndo()
{
    if (selectedTile < 0 || selectedTile >= static_cast<int>(tiles.size()))
        return;
    auto &tile = tiles[selectedTile];
    if (!tile.hasUndo || !tile.pagedLod || tile.childIndex >= tile.pagedLod->getNumChildren())
        return;

    auto *selectionManager = coVRSelectionManager::instance();
    const bool hadSelection = tile.selected;
    if (hadSelection && tile.contentNode)
    {
        selectionManager->removeNode(tile.contentNode.get());
        selectionManager->selectionChanged();
    }
    tile.selected = false;
    tile.pagedLod->setChild(tile.childIndex, tile.undoChild.get());
    tile.pagedLod->setFileName(tile.childIndex, tile.undoFileName);
    tile.undoChild->setNodeMask(tile.undoNodeMask);
    tile.contentNode = tile.undoChild;
    tile.selected = tile.undoSelected;
    tile.omitted = tile.undoOmitted;
    if (tile.omitted)
        omittedTiles.insert(tile.fileName);
    else
        omittedTiles.erase(tile.fileName);
    stageTileReplacement(tile.fileName, tile.undoOmitted ? "" : tile.undoFileName);
    tile.hasUndo = false;
    if (tile.selected)
        updateTileSelectionHighlight(tile);
    else if (hadSelection)
        selectionManager->pickedObjChanged();
    tileList->select(selectedTile + 1, true);
    selectionName->setText(tile.name);
}
