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
#include <vrml97/vrml/VrmlNamespace.h>
#include <VrmlNodeGeoData.h>

#include <config/CoviseConfig.h>

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
#include <string>
#include "HTTPClient/CURL/methods.h"
#include "HTTPClient/CURL/request.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/rapidjson.h>
#include "cover/coVRConfig.h"
#include <PluginUtil/PluginMessageTypes.h>
#include <stdio.h>

using namespace vrui;
using namespace opencover;
#include <exiv2/exiv2.hpp>
namespace opencover
{
namespace ui
{
    class Menu;
    class Label;
    class Group;
    class Button;
    class EditField;
}
}

using namespace covise;
using namespace opencover;

skyEntry::skyEntry(const std::string &n, const std::string &fn, double lon, double lat)
{
    name = n;
    fileName = fn;
    skyLongitude = lon;
    skyLatitude = lat;
}
skyEntry::~skyEntry()
{
}
skyEntry::skyEntry(const skyEntry &se)
{
    name = se.name;
    fileName = se.fileName;
    skyNode = se.skyNode;
    skyTexture = se.skyTexture;
    type = se.type;
    skyLongitude = se.skyLongitude;
    skyLatitude = se.skyLatitude;
    skyTrueNorth = se.skyTrueNorth;
}

std::string name;
std::string fileName;
osg::ref_ptr<osg::Node> skyNode;
osg::ref_ptr<osg::Texture2D> skyTexture;

std::string getCoordinates(const std::string &address)
{
    using namespace opencover::httpclient::curl;
    std::string readBuffer;

    CURL *curl = curl_easy_init();
    if (!curl)
    {
        return "[ERROR] Failed to initialize CURL.";
    }

    char *encoded = curl_easy_escape(curl, address.c_str(), address.length());
    if (!encoded)
    {
        curl_easy_cleanup(curl);
        return "[ERROR] Failed to encode address.";
    }

    // Nominatim request URL
    std::string url = "https://nominatim.openstreetmap.org/search?q=" + std::string(encoded) + "&format=json&limit=1";
    curl_free(encoded);

    GET getRequest(url);
    Request::Options options = {
        { CURLOPT_USERAGENT, "Covise Plugin GeoDataLoader (https://github.com/hlrs-vis/covise)" },
    };
    if (!Request().httpRequest(getRequest, readBuffer, options))
        readBuffer = "[ERROR] Failed to fetch data from Nominatim. With request: " + url;
    curl_easy_cleanup(curl);
    return readBuffer;
}

std::optional<GeoDataLoader::geoLocation> GeoDataLoader::parseCoordinates(const std::string &jsonData)
{
    rapidjson::Document document;

    if (document.Parse(jsonData.c_str()).HasParseError())
        return std::nullopt;

    if (!document.IsArray() || document.Empty())
        return std::nullopt;

    const rapidjson::Value &location = document[0];
    if (!location.HasMember("lat") || !location.HasMember("lon"))
        return std::nullopt;

    GeoDataLoader::geoLocation result;
    result.latitude = std::stod(location["lat"].GetString());
    result.longitude = std::stod(location["lon"].GetString());

    PJ_COORD input;
    input.lp.phi = result.longitude;
    input.lp.lam = result.latitude;

    // Perform the transformation
    PJ_COORD output;
    output = proj_trans(ProjInstance, PJ_FWD, input); // Forward transformation (WGS84 -> UTM)

    result.easting = output.enu.e;
    result.northing = output.enu.n;
    result.altitude = 500; // Default altitude value

    if (location.HasMember("display_name"))
        result.displayName = location["display_name"].GetString();

    return result;
}

void GeoDataLoader::jumpToLocation(const osg::Vec3d &worldPos)
{
    double easting = worldPos.x();
    double northing = worldPos.y();
    double altitude = worldPos.z();
    double scale = cover->getScale();

    // --- Intersection-Test for zVal ---
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector = new osgUtil::LineSegmentIntersector(
        osg::Vec3d(easting, northing, 10000.0),
        osg::Vec3d(easting, northing, -10000.0));
    osgUtil::IntersectionVisitor iv(intersector);
    if (terrainNode)
        terrainNode->accept(iv);

    if (intersector->containsIntersections())
    {
        altitude = intersector->getFirstIntersection().getLocalIntersectPoint().z();
        altitude += 100;
        std::cout << "Intersection found, Z = " << (altitude - 100) << std::endl;
    }
    else
    {
        std::cout << "No intersection found, using default Z = " << altitude << std::endl;
    }

    // set the viewer position
    osg::Matrix mat;
    osg::Vec3 targetPos(easting, northing, altitude);
    std::cout << "Target Position in Meters (UTM): " << targetPos.x() << ", " << targetPos.y() << ", " << targetPos.z() << std::endl;
    targetPos += rootOffset;

    float trueNorthRadian = osg::DegreesToRadians(trueNorthDegree);
    osg::Matrix rot = osg::Matrix::rotate(-trueNorthRadian, osg::Vec3(0, 0, 1));

    targetPos = targetPos * rot;
    targetPos = targetPos * scale;
    mat.setTrans(-targetPos);
    cover->setXformMat(mat);

    // set the closest sky //TODO set sky based on viewer pos not only after location jump
    double minDistance = 1e30;
    int closestSky = -1;
    for (size_t i = 0; i < skyEntries.size(); ++i)
    {
        const skyEntry &se = skyEntries[i];
        PJ_COORD input;
        input.lp.phi = se.skyLongitude;
        input.lp.lam = se.skyLatitude;

        // Perform the transformation
        PJ_COORD output;
        output = proj_trans(ProjInstance, PJ_FWD, input); // Forward transformation (WGS84 -> UTM)

        double skyEasting = output.enu.e;
        double skyNorthing = output.enu.n;
        double dx = skyEasting - easting;
        double dy = skyNorthing - northing;
        double distance = sqrt(dx * dx + dy * dy);
        if (distance < minDistance)
        {
            minDistance = distance;
            closestSky = static_cast<int>(i) + 1; // +1 because sky selection list has "None" at index 0
        }
    }
    if (closestSky >= 0)
    {
        setSky(closestSky);
        skys->select(closestSky);
    }
}

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
    ProjContext = proj_context_create();
    // Define the transformation from WGS84 to UTM Zone 32N (EPSG:32632)
    ProjInstance = proj_create_crs_to_crs(ProjContext, "EPSG:4326", "EPSG:32632", NULL); // EPSG:4326 is WGS84, EPSG:32632 is UTM Zone 32N

    geoDataMenu = new ui::Menu("GeoData", this);
    geoDataMenu->setText("GeoData");
    geoDataMenu->allowRelayout(true);

    rootNode = new osg::MatrixTransform();
    rootNode->setName("geodata");
    skyRootNode = new osg::MatrixTransform();
    skyRootNode->setName("sky");
    cover->getObjectsRoot()->addChild(rootNode);
    cover->getScene()->addChild(skyRootNode);

    auto configFile = config();

    originGroup = new ui::Group(geoDataMenu, "originGroup");
    originGroup->setText("Origin");
    originGroup->allowRelayout(true);

    // selection list for offsets for different datasets
    datasetList = new ui::SelectionList(originGroup, "datasets");
    datasetList->setText("Choose Datasets");
    datasetList->append("None");
    datasets.clear();
    TexturedSphere = new osg::Geode;
    osg::Sphere *_Sphere = new osg::Sphere();
    _Sphere->setRadius(8000);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(1.0);
    hint->setCreateBackFace(true);
    hint->setCreateFrontFace(false);
    hint->setCreateTextureCoords(true);
    // hint->setCreateNormals(true);
    osg::ShapeDrawable *_sphereDrawable = new osg::ShapeDrawable(_Sphere, hint);
    _sphereDrawable->setColor(osg::Vec4(1, 1, 1, 1));
    _sphereDrawable->setUseDisplayList(false); // turn off display list so that we can change the pointer length
    TexturedSphere->addDrawable(_sphereDrawable);
    osg::StateSet *stateset = TexturedSphere->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::Material *spheremtl = new osg::Material;
    spheremtl->setColorMode(osg::Material::OFF);
    spheremtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1));
    spheremtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1));
    spheremtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    spheremtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
    spheremtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    stateset->setAttributeAndModes(spheremtl, osg::StateAttribute::ON);
    osg::CullFace *cF = new osg::CullFace();
    cF->setMode(osg::CullFace::BACK);
    stateset->setAttributeAndModes(cF, osg::StateAttribute::OFF);
    TexturedSphere->setStateSet(stateset);

    auto datasetEntries = configFile->array<opencover::config::Section>("", "datasets");

    for (size_t i = 0; i < datasetEntries->size(); i++)
    {
        opencover::config::Section entry = (*datasetEntries)[i];

        DatasetInfo dataset;
        dataset.name = entry.value<std::string>("", "name")->value();
        dataset.altitude = entry.value<double>("", "altitude")->value();
        dataset.trueNorth = entry.value<double>("", "trueNorth")->value();

        double latitude = entry.value<double>("", "latitude")->value();
        double longitude = entry.value<double>("", "longitude")->value();

        if (latitude || longitude)
        {
            PJ_COORD input;
            input.lp.phi = longitude;
            input.lp.lam = latitude;

            // transform
            PJ_COORD output = proj_trans(ProjInstance, PJ_FWD, input);
            dataset.easting = output.enu.e;
            dataset.northing = output.enu.n;
        }
        else
        {
            dataset.easting = entry.value<double>("", "easting")->value();
            dataset.northing = entry.value<double>("", "northing")->value();
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
        } });
    tempEastingText = "0.0";
    tempNorthingText = "0.0";
    tempAltitudeText = "0.0";
    tempTrueNorthText = "0.0";

    easting = new ui::EditField(originGroup, "easting");
    easting->setText("Easting (m):");
    easting->setCallback([this](std::string val)
        { this->tempEastingText = val; });

    northing = new ui::EditField(originGroup, "northing");
    northing->setText("Northing (m):");
    northing->setCallback([this](std::string val)
        { this->tempNorthingText = val; });

    altitude = new ui::EditField(originGroup, "altitude");
    altitude->setText("Altitude (m):");
    altitude->setCallback([this](std::string val)
        { this->tempAltitudeText = val; });

    trueNorth = new ui::EditField(originGroup, "trueNorth");
    trueNorth->setText("True North (°):");
    trueNorth->setCallback([this](std::string val)
        { this->tempTrueNorthText = val; });

    applyOffset = new ui::Button(originGroup, "applyOffset");
    applyOffset->setText("apply");
    applyOffset->setCallback([this](bool state)
        {
            double originEasting = 0.0;
            double originNorthing = 0.0;
            double originAltitude = 0.0; 
            double trueNorth = 0.0;
            if (tempEastingText != "" )
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
            setRootTransform(-origin, trueNorth);
            applyOffset->setState(false); });
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
    terrainVisibilityButton->setState(true);
    terrainVisibilityButton->setCallback([this](bool state)
        {
        for (const auto& pair : loadedTerrains)
        {
            if (pair.second)
                pair.second->setNodeMask(state ? 0xffffffff : 0x0);
        }
        showTerrain = state; });

    buildingVisibilityButton = new ui::Button(visibilityGroup, "buildingVisibility");
    buildingVisibilityButton->setText("Buildings");
    buildingVisibilityButton->setState(true);
    buildingVisibilityButton->setCallback([this](bool state)
        {
        for (const auto& pair : loadedBuildings)
        {
            if (pair.second)
                pair.second->setNodeMask(state ? 0xffffffff : 0x0);
        }
        showBuildings = state; });

    // create Button for each region in config
    geoObjectGroup = new ui::Group(geoDataMenu, "GeoObjects");
    geoObjectGroup->setText("Geo-Objects");
    geoObjectGroup->allowRelayout(true);

    std::map<std::string, ui::Button *> regionButtons;

    auto terrainEntries = configFile->array<opencover::config::Section>("", "regions");

    for (size_t i = 0; i < terrainEntries->size(); i++)
    {

        opencover::config::Section terrainEntry = (*terrainEntries)[i];

        std::string region_name = terrainEntry.value<std::string>("", "name")->value();
        std::string terrain_path = terrainEntry.value<std::string>("", "terrainPath")->value();
        std::string lod_path = terrainEntry.value<std::string>("", "lodPath")->value();

        if (terrain_path != "" || lod_path != "")
        {
            ui::Button *regionButton = new ui::Button(geoObjectGroup, region_name);
            regionButton->setText(region_name);
            regionButton->setState(false);
            regionButton->setCallback([this, region_name, terrain_path, lod_path](bool state)
                {
                if (state)
                {
                    if (loadedTerrains.find(region_name) == loadedTerrains.end())
                    {
                        terrainNode = loadTerrain(terrain_path,osg::Vec3d(0,0,0));
                        if (terrainNode)
                        {
                            loadedTerrains[region_name] = terrainNode;
                            rootNode->addChild(terrainNode);
                            terrainNode->setName(region_name);
                            terrainNode->setNodeMask(showTerrain ? 0xffffffff : 0x0);
                        }
                    }

                    if(loadedBuildings.find(region_name) == loadedBuildings.end())
                    {
                        buildingNode = loadTerrain(lod_path,osg::Vec3d(0,0,0));
                        if (buildingNode)
                        {
                            loadedBuildings[region_name] = buildingNode;
                            rootNode->addChild(buildingNode);
                            buildingNode->setName(region_name);
                            buildingNode->setNodeMask(showBuildings ? 0xffffffff : 0x0);
                        }
                    }                    
                }
                else
                {
                    if (loadedTerrains.find(region_name) != loadedTerrains.end())
                    {
                        rootNode->removeChild(loadedTerrains[region_name]);
                        loadedTerrains.erase(region_name);
                    }

                    if (loadedBuildings.find(region_name) != loadedBuildings.end())
                    {
                        rootNode->removeChild(loadedBuildings[region_name]);
                        loadedBuildings.erase(region_name);
                    }
                } });
            regionButtons[region_name] = regionButton;
        }
    }

    skyGroup = new ui::Group(geoDataMenu, "sky");
    skyGroup->setText("Sky");
    skyGroup->allowRelayout(true);

    skys = new ui::SelectionList(skyGroup, "Skys");
    skys->append("None");
    skyPath = configString("sky", "skyDir", "/data/Geodata/sky")->value();
    defaultSky = configInt("sky", "defaultSky", 6)->value();
    int skyNumber = 1;
    try
    {
        for (const auto &entry : fs::directory_iterator(skyPath))
        {
            if (entry.is_regular_file() && ((entry.path().extension() == ".wrl") || (entry.path().extension() == ".WRL")))
            {
                std::string name = entry.path().filename().string();
                skyEntry se(name.substr(0, name.length() - 4), entry.path().string(), 0.0, 0.0);
                se.fileName = name;
                skyEntries.push_back(se);
                skys->append(se.name);
                if (skyNumber == defaultSky)
                {
                    setSky(skyNumber);
                    skys->select(skyNumber, false);
                }
                skyNumber++;
            }
            if (entry.is_regular_file() && ((entry.path().extension() == ".jpg") || (entry.path().extension() == ".JPG")))
            {
                std::string name = entry.path().filename().string();
                skyEntry se(name.substr(0, name.length() - 4), entry.path().string(), 0.0, 0.0);
                se.fileName = name;
                se.type = skyEntry::texture;
                FILE *fp = fopen(entry.path().string().c_str(), "rb");
                if (fp == nullptr)
                {
                    printf("Can't open file %s.\n", entry.path().string().c_str());
                    return -1;
                }
                else
                {

                    int toread = 400000;
                    // fseek(fp, -toread, SEEK_END);
                    // unsigned long fsize = ftell(fp);
                    int bsize;
                    unsigned char *buf = new unsigned char[toread];
                    bsize = fread(buf, 1, toread, fp);
                    fclose(fp);

                    // Parse EXIF
                    float gimbalYaw = 0.0f;
                    float flightYaw = 0.0f;

                    auto image = Exiv2::ImageFactory::open(entry.path().string());
                    image->readMetadata();
                    auto &exif = image->exifData();
                    if (exif.empty())
                    {
                        std::cerr << "No EXIF data found in " << entry.path().string() << std::endl;
                        se.skyLongitude = 0.0;
                        se.skyLatitude = 0.0;
                    }
                    else
                    {
                        auto latIt = exif.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLatitude"));
                        if (latIt == exif.end())
                        {
                            std::cerr << "No GPS Latitude found in EXIF data of " << entry.path().string() << std::endl;
                            se.skyLatitude = 0.0;
                        }
                        else
                        {
                            Exiv2::URational latDeg = exif["Exif.GPSInfo.GPSLatitude"].toRational(0);
                            Exiv2::URational latMin = exif["Exif.GPSInfo.GPSLatitude"].toRational(1);
                            Exiv2::URational latSec = exif["Exif.GPSInfo.GPSLatitude"].toRational(2);
                            se.skyLatitude = latDeg.first + latMin.first / (60.0 * latMin.second) + latSec.first / (3600.0 * latSec.second);
                        }
                        auto lonIt = exif.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLongitude"));
                        if (lonIt == exif.end())
                        {
                            std::cerr << "No GPS Longitude found in EXIF data of " << entry.path().string() << std::endl;
                            se.skyLongitude = 0.0;
                        }
                        else
                        {
                            Exiv2::URational lonDeg = exif["Exif.GPSInfo.GPSLongitude"].toRational(0);
                            Exiv2::URational lonMin = exif["Exif.GPSInfo.GPSLongitude"].toRational(1);
                            Exiv2::URational lonSec = exif["Exif.GPSInfo.GPSLongitude"].toRational(2);
                            se.skyLongitude = lonDeg.first + lonMin.first / (60.0 * lonMin.second) + lonSec.first / (3600.0 * lonSec.second);
                        }
                    }

                    Exiv2::XmpData &xmpData = image->xmpData();
                    if (xmpData.empty())
                    {
                        std::cerr << "No XMP data found in " << entry.path().string() << std::endl;
                        se.skyTrueNorth = 0.0;
                    }
                    else
                    {
                        auto gimbalYawIt = xmpData.findKey(Exiv2::XmpKey("Xmp.drone-dji.GimbalYawDegree"));
                        if (gimbalYawIt == xmpData.end())
                        {
                            std::cerr << "No Gimbal Yaw found in XMP data of " << entry.path().string() << std::endl;
                            se.skyTrueNorth = 0.0;
                        }
                        else
                        {
                            gimbalYaw = xmpData["Xmp.drone-dji.GimbalYawDegree"].toFloat();
                            se.skyTrueNorth = gimbalYaw;
                            if (se.skyTrueNorth < -180.0)
                                se.skyTrueNorth += 360.0;
                            else if (se.skyTrueNorth > 180.0)
                                se.skyTrueNorth -= 360.0;
                        }
                        auto FlightYawIt = xmpData.findKey(Exiv2::XmpKey("Xmp.drone-dji.FlightYawDegree"));
                        if (FlightYawIt == xmpData.end())
                        {
                            std::cerr << "No Flight Yaw found in XMP data of " << entry.path().string() << std::endl;
                        }
                        else
                        {
                            flightYaw = xmpData["Xmp.drone-dji.FlightYawDegree"].toFloat();
                            se.skyTrueNorth = flightYaw;
                            if (se.skyTrueNorth < -180.0)
                                se.skyTrueNorth += 360.0;
                            else if (se.skyTrueNorth > 180.0)
                                se.skyTrueNorth -= 360.0;
                        }
                    }
                    cout << "Exiv2: Parsed EXIF for " << entry.path().string() << ": Longitude=" << se.skyLongitude << ", Latitude=" << se.skyLatitude << endl;
                    cout << "XMP: Parsed Gimbal Yaw for " << entry.path().string() << ": GimbalYawDegree=" << gimbalYaw << endl;
                    cout << "XMP: Parsed Flight Yaw for " << entry.path().string() << ": FlightYawDegree=" << flightYaw << endl;
                }
                skyEntries.push_back(se);
                skys->append(se.name);
                if (skyNumber == defaultSky)
                {
                    setSky(skyNumber);
                    skys->select(skyNumber, false);
                }
                skyNumber++;
            }
        }
    }
    catch (const fs::filesystem_error &err)
    {
        std::cerr << "Filesystem error: " << err.what() << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "General error: " << ex.what() << std::endl;
    }
    skys->setCallback([this](int selection)
        { setSky(selection); });
    northAngle = 0;

    float farValue = coVRConfig::instance()->farClip();
    float scale = (farValue * 2) / 20000;
    skyRootNode->setMatrix(osg::Matrix::scale(scale, scale, scale) * osg::Matrix::rotate(northAngle + osg::DegreesToRadians(90.0), osg::Vec3(0, 0, -1)));

    skyNorthSlider = new ui::Slider(skyGroup, "skyNorth");
    skyNorthSlider->setText("Sky True North (°)");
    skyNorthSlider->setBounds(-180.0, 180.0);
    skyNorthSlider->setValue(osg::RadiansToDegrees(northAngle));
    skyNorthSlider->setCallback([this](double value, bool released)
        {
            northAngle = osg::DegreesToRadians(value);
            float farValue = coVRConfig::instance()->farClip();
            float scale = (farValue * 2) / 20000;
            skyRootNode->setMatrix(osg::Matrix::scale(scale, scale, scale)*osg::Matrix::rotate(northAngle + osg::DegreesToRadians(90.0), osg::Vec3(0,0,-1))); });

    locationGroup = new ui::Group(geoDataMenu, "locationGroup");
    locationGroup->setText("Location");
    locationGroup->allowRelayout(true);

    location = new ui::EditField(locationGroup, "location");
    location->setText("");
    location->setCallback([this](const std::string &val)
        {
            std::string json = getCoordinates(val);

            auto geo = parseCoordinates(json);
            if (!geo)
                return;

            osg::Vec3d targetPos = osg::Vec3d(geo->easting, geo->northing, geo->altitude);
            jumpToLocation(targetPos);

            if (!geo->displayName.empty())
                location->setText(geo->displayName); });

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
    proj_destroy(ProjInstance);
    proj_context_destroy(ProjContext);
}

void GeoDataLoader::message(int toWhom, int type, int length, const void *data)
{
    const char *messageData = (const char *)data;
    if (type == PluginMessageTypes::LoadTerrain)
        loadTerrain(messageData + 20, osg::Vec3d(0, 0, 0)); // 20= opencover://terrain/
    else if (type == PluginMessageTypes::setSky)
    {
        int offset = 0;
        if (messageData[0] == '/' || messageData[0] == '\\')
            offset = 1;
        int num = atoi(messageData + offset);
        setSky(num);
    }
}
void GeoDataLoader::setSky(int selection)
{
    while (skyRootNode->getNumChildren())
        skyRootNode->removeChild(skyRootNode->getChild(0));

    currentSkyNode = nullptr;
    if (selection > 0 && selection <= static_cast<int>(skyEntries.size()))
    {
        skyEntry &sky = skyEntries[selection - 1];
        if (sky.type == skyEntry::geometry)
        {
            if (sky.skyNode != nullptr)
            {
                skyRootNode->addChild(sky.skyNode);
            }
            else
            {
                sky.skyNode = coVRFileManager::instance()->loadFile((skyPath + "/" + sky.fileName).c_str(), nullptr, skyRootNode);
                skyRootNode->addChild(sky.skyNode);
            }
        }
        else
        {

            sky.skyNode = TexturedSphere;
            if (sky.skyTexture == nullptr)
            {
                sky.skyTexture = coVRFileManager::instance()->loadTexture((skyPath + "/" + sky.fileName).c_str());
                sky.skyTexture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
                sky.skyTexture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
                sky.skyTexture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
                sky.skyTexture->setResizeNonPowerOfTwoHint(false);
            }
            osg::StateSet *stateset = TexturedSphere->getOrCreateStateSet();
            stateset->setTextureAttributeAndModes(0, sky.skyTexture, osg::StateAttribute::ON);

            shader = coVRShaderList::instance()->get("skySphere");
            shader->apply(stateset);
            topUniform = shader->getcoVRUniform("top");
            bottomUniform = shader->getcoVRUniform("bottom");
            floorColorUniform = shader->getcoVRUniform("floorColor");

            skyRootNode->addChild(sky.skyNode);
        }
        currentSkyNode = sky.skyNode;
        northAngle = osg::DegreesToRadians(sky.skyTrueNorth);
        if (skyNorthSlider != nullptr)
            skyNorthSlider->setValue(sky.skyTrueNorth);
    }
}

void GeoDataLoader::setSky(std::string fileName)
{
    int n = 0;
    for (const auto &sky : skyEntries)
    {
        if (sky.fileName == fileName) // already have this file in the list
        {
            setSky(n + 1);
            return;
        }
        n++;
    }
    std::filesystem::path path(fileName);
    std::string fn = path.filename().string();
    skyEntry se(fn.substr(0, fn.length() - 4), fileName, 0.0, 0.0);
    skyEntries.push_back(se);
    skys->append(se.name);
}

void GeoDataLoader::setTop(float t)
{
    if (topUniform != nullptr)
    {
        topUniform->setValue(t);
    }
}

void GeoDataLoader::setBottom(float b)
{
    if (bottomUniform != nullptr)
    {
        bottomUniform->setValue(b);
    }
}

void GeoDataLoader::setFloorColor(osg::Vec4 fc)
{
    if (floorColorUniform != nullptr)
    {
        floorColorUniform->setValue(fc);
    }
}

void GeoDataLoader::setRootTransform(const osg::Vec3 &off, float trueNorthDeg)
{
    rootOffset = off;
    trueNorthDegree = trueNorthDeg;

    float trueNorthRad = osg::DegreesToRadians(trueNorthDegree);
    osg::Matrix trans = osg::Matrix::translate(rootOffset);
    osg::Matrix rot = osg::Matrix::rotate(-trueNorthRad, osg::Vec3(0, 0, 1));
    rootNode->setMatrix(trans * rot);
}

bool GeoDataLoader::update()
{
    const osg::Matrix &m = cover->getObjectsXform()->getMatrix();
    // double x = m(0, 0);
    // double y = m(1, 0);
    // northAngle = -atan2(y, x);
    // skyRootNode->setMatrix(osg::Matrix::scale(1000, 1000, 1000) * osg::Matrix::rotate(northAngle, osg::Vec3(0, 0, 1)));
    float farValue = coVRConfig::instance()->farClip();
    float scale = ((farValue) / 20000) + 500; // scale the sphere so that it stays a bit closer than the far clipping plane.
    skyRootNode->setMatrix(osg::Matrix::scale(scale, scale, scale) * osg::Matrix::rotate(northAngle + osg::DegreesToRadians(90.0), osg::Vec3(0, 0, -1)) * osg::Matrix::rotate(cover->getObjectsXform()->getMatrix().getRotate()));
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
