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

#include <cover/coVRPluginSupport.h>
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
#include <PluginUtil/PluginMessageTypes.h>
#include <stdio.h>
#include <utility>

using namespace vrui;
using namespace opencover;

GeoDataLoader *GeoDataLoader::s_instance = nullptr;

GeoDataLoader::GeoDataLoader()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("GeoData", cover->ui)
{
    assert(s_instance == nullptr);
    s_instance = this;
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
            auto enu = GeoData::instance()->globalToReference(osg::Vec3(longitude, latitude, altitude));
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
    editInteraction = new editTerrain(this);

    editButton = new ui::Button(editGroup, "editTerrain");
    editButton->setText("editTerrain");
    editButton->setState(false);
    editButton->setCallback([this](bool state)
        {
        if (state)
        {
            editInteraction->enableIntersection();
        }
        else
        {
            editInteraction->disableIntersection();
            } });

    deleteSelected = new ui::Action(editGroup, "deleteSelected");
    deleteSelected->setText("delete");
    deleteSelected->setCallback([this]()
        { doDelete(); });

    replace = new ui::Action(editGroup, "replace");
    replace->setText("replace");
    replace->setCallback([this]()
        { doReplace(); });

    undo = new ui::Action(editGroup, "undo");
    undo->setText("undo");
    undo->setCallback([this]()
        { doUndo(); });

    selectionName = new ui::Label(editGroup, "name");
    selectionName->setText("nothing is selected");

    return true;
}

// this is called if the plugin is removed at runtime
GeoDataLoader::~GeoDataLoader()
{
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
            auto terrainNode = loadTerrain(region.terrain_path, osg::Vec3d(0, 0, 0));
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
            auto buildingNode = loadTerrain(region.lod_path, osg::Vec3d(0, 0, 0));
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

void GeoDataLoader::message(int toWhom, int type, int length, const void *data)
{
    const char *messageData = (const char *)data;
    if (type == PluginMessageTypes::LoadTerrain)
        loadTerrain(messageData + 20, osg::Vec3d(0, 0, 0)); // 20= opencover://terrain/
}

bool GeoDataLoader::update()
{
    if (editInteraction->isRunning())
    {
        // get the intersected node
        if (cover->getIntersectedNode())
        {
            // position label with name
            // nameLabel_->setPosition(cover->getIntersectionHitPointWorld());
            if (cover->getIntersectedNode() != oldIntersectedNode)
            {
                // get the node name
                string nodeName;
                // char *labelName = NULL;

                // show only node names under objects root
                // so check if node is under objects root
                osg::Node *currentNode = cover->getIntersectedNode();
                while (currentNode != NULL)
                {
                    if (currentNode == cover->getObjectsRoot())
                    {
                        /* if (showGeodeName_)
                         {
                             // first look for a node description beginning with _SCGR_
                             std::vector<std::string> dl = cover->getIntersectedNode()->getDescriptions();
                             for (size_t i = 0; i < dl.size(); i++)
                             {
                                 std::string descr = dl[i];
                                 if (descr.find("_SCGR_") != string::npos)
                                 {
                                     nodeName = dl[i];
                                     // fprintf(stderr,"found description %s\n", nodeName.c_str());
                                     break;
                                 }
                             }
                             if (nodeName.empty())
                             { // if there is no description we take the node name
                                 nodeName = cover->getIntersectedNode()->getName();
                                 // fprintf(stderr,"taking the node name %s\n", nodeName.c_str());
                             }
                         }
                         else // show name of the dcs above
                         {
                             osg::Node *parentDcs = cover->getIntersectedNode()->getParent(0);
                             nodeName = parentDcs->getName();
                             if (nodeName.empty())
                             {
                                 // if the dcs has no name it could be a helper node
                                 if ((parentDcs->getNumDescriptions() > 0) && (parentDcs->getDescription(1) == "SELECTIONHELPER"))
                                 {
                                     nodeName = parentDcs->getParent(0)->getName();
                                 }
                             }
                         }*/
                        break;
                    }
                    if (currentNode->getNumParents() > 0)
                        currentNode = currentNode->getParent(0);
                    else
                        currentNode = NULL;
                }

                // show label
                if (!nodeName.empty())
                {
                    std::string onam = nodeName;
                    // eliminate _SCGR_
                    if (nodeName.find("_SCGR_") != string::npos)
                    {
                        nodeName = nodeName.substr(0, nodeName.rfind("_SCGR_"));
                    }
                    // eliminate 3D studio max name -faces
                    if (nodeName.find("-FACES") != string::npos)
                    {
                        nodeName = nodeName.substr(0, nodeName.rfind("-FACES"));
                    }
                    // eliminate 3D studio max name _faces
                    if (nodeName.find("_FACES") != string::npos)
                    {
                        nodeName = nodeName.substr(0, nodeName.rfind("_FACES"));
                    }
                    // eliminate numbers at the end
                    while ((nodeName.length() > 0) && (nodeName[nodeName.length() - 1] >= '0') && (nodeName[nodeName.length() - 1] <= '9'))
                    {
                        nodeName.erase(nodeName.length() - 1, 1);
                    }

                    // replace underlines with blanks
                    if (nodeName.length() > 0 && nodeName[nodeName.length() - 1] == '_')
                    {
                        nodeName.erase(nodeName.length() - 1);
                    }

                    // check if we now have an empty string (containing spaces only)
                    if (nodeName.length() > 0)
                    {
                        selectionName->setText(nodeName);

                        // delete []labelName;
                        oldIntersectedNode = cover->getIntersectedNode();
                    }
                    else
                    {
                        selectionName->setText("-");
                        oldIntersectedNode = NULL;
                    }
                }
                else
                {
                    selectionName->setText("-");
                    oldIntersectedNode = NULL;
                }
            }
            else
            {
                selectionName->setText("-");
                oldIntersectedNode = NULL;
            }
        }
        else
        {
            selectionName->setText("-");
            oldIntersectedNode = NULL;
        }
    }
    return false; // don't request that scene be re-rendered
}

GeoDataLoader *GeoDataLoader::instance()
{
    if (!s_instance)
        s_instance = new GeoDataLoader;
    return s_instance;
}

osg::ref_ptr<osg::Node> GeoDataLoader::loadTerrain(std::string filename, osg::Vec3d localOffset)
{
    VRViewer *viewer = VRViewer::instance();
    osgDB::DatabasePager *pager = viewer->getDatabasePager();
    std::cout << "DatabasePager:" << std::endl;
    std::cout << "\t num threads: " << pager->getNumDatabaseThreads() << std::endl;
    std::cout << "\t getDatabasePagerThreadPause: " << (pager->getDatabasePagerThreadPause() ? "yes" : "no") << std::endl;
    std::cout << "\t getDoPreCompile: " << (pager->getDoPreCompile() ? "yes" : "no") << std::endl;
    std::cout << "\t getApplyPBOToImages: " << (pager->getApplyPBOToImages() ? "yes" : "no") << std::endl;
    std::cout << "\t getTargetMaximumNumberOfPageLOD: " << pager->getTargetMaximumNumberOfPageLOD() << std::endl;
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

        std::cout << "IncrementedCompileOperation:" << std::endl;
        std::cout << "\t is active: " << (coop->isActive() ? "yes" : "no") << std::endl;
        std::cout << "\t target frame rate: " << coop->getTargetFrameRate() << std::endl;
        std::cout << "\t getMinimumTimeAvailableForGLCompileAndDeletePerFrame: " << coop->getMinimumTimeAvailableForGLCompileAndDeletePerFrame() << std::endl;
        std::cout << "\t getMaximumNumOfObjectsToCompilePerFrame: " << coop->getMaximumNumOfObjectsToCompilePerFrame() << std::endl;
        std::cout << "\t flush time ratio: " << coop->getFlushTimeRatio() << std::endl;
        std::cout << "\t conservative time ratio: " << coop->getConservativeTimeRatio() << std::endl;
    }

    osg::Node *terrain = osgDB::readNodeFile(filename);

    if (terrain)
    {
        terrain->setDataVariance(osg::Object::DYNAMIC);

        osg::StateSet *terrainStateSet = terrain->getOrCreateStateSet();
        terrainStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
        terrainStateSet->setMode(GL_LIGHT0, osg::StateAttribute::ON);

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

            osg::Vec3f global(poPoint->getX(), poPoint->getY(), altitude);
            auto local = GeoData::instance()->globalToReference(global);

            labels.emplace_back(std::make_shared<PlaceLabel>(name, local, node, size));
        }
    }

    return PlaceLabelGroup { node, labels };
}

COVERPLUGIN(GeoDataLoader)

void editTerrain::createGeometry()
{

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.0);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    mySphereDrawable->setColor(osg::Vec4(0.6f, 0.6f, 0.6f, 1.0f));
    geometryNode = new osg::Geode();
    scaleTransform->addChild(geometryNode.get());
    moveTransform->addChild(scaleTransform.get());
    ((osg::Geode *)geometryNode.get())->addDrawable(mySphereDrawable);
    geometryNode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
    geometryNode->setNodeMask(geometryNode->getNodeMask() | (Isect::Pick) | (Isect::Intersection));
}

editTerrain::editTerrain(GeoDataLoader *g)
    : vrui::coCombinedButtonInteraction(vrui::coInteraction::ButtonA, "editTerrain", vrui::coInteraction::InteractionPriority::Medium)
{
    gdl = g;
    _selectedHl = new osg::StateSet();
    _intersectedHl = new osg::StateSet();

    if (_standardHL)
    {
        // set default materials
        osg::Material *selMaterial = new osg::Material();
        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(1.0, 0.3, 0.0, 1.0f));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(1.0, 0.3, 0.0, 1.0f));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);
        osg::Material *isectMaterial = new osg::Material();
        isectMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.6, 0.6, 0.0, 1.0f));
        isectMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.6, 0.6, 0.0, 1.0f));
        isectMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        isectMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        isectMaterial->setColorMode(osg::Material::OFF);
        _selectedHl->setAttribute(selMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
        _intersectedHl->setAttribute(isectMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
    }
    /*
        _selectedHl->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
        _selectedHl->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);*/
}

void editTerrain::enableIntersection()
{
    coInteractionManager::the()->registerInteraction(this);
}

void editTerrain::disableIntersection()
{
    // interactor is normally unregistered in miss
    // if we disable intersection miss is not called anymore
    // therefore we make sure that the interactor is unregistered
    if (registered)
    {
        coInteractionManager::the()->unregisterInteraction(this);
    }

    resetState();
}

void GeoDataLoader::doDelete()
{
}
void GeoDataLoader::doReplace()
{
}
void GeoDataLoader::doUndo()
{
}
