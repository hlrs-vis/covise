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
#include <cover/coVRShader.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>

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
#include <../../../../3rdparty/easyexif/exif.h>
#include <stdio.h>
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

skyEntry::skyEntry(const std::string& n, const std::string& fn)
{
    name = n;
    fileName = fn;
}
skyEntry::~skyEntry()
{

}
skyEntry::skyEntry(const skyEntry& se)
{
    name = se.name;
    fileName = se.fileName;
    skyNode = se.skyNode;
    skyTexture = se.skyTexture;
    type = se.type;
}

std::string name;
std::string fileName;
osg::ref_ptr<osg::Node> skyNode;
osg::ref_ptr<osg::Texture2D> skyTexture;

std::string getCoordinates(const std::string& address) {
    using namespace opencover::httpclient::curl;
    std::string readBuffer;

    CURL *curl = curl_easy_init();
    if (!curl) {
        return "[ERROR] Failed to initialize CURL.";
    }

    char *encoded = curl_easy_escape(curl, address.c_str(), address.length());
    if (!encoded) {
        curl_easy_cleanup(curl);
        return "[ERROR] Failed to encode address.";
    }

    // Nominatim request URL
    std::string url = "https://nominatim.openstreetmap.org/search?q=" + std::string(encoded) + "&format=json&limit=1";
    curl_free(encoded);

    GET getRequest(url);
    Request::Options options = {
        {CURLOPT_USERAGENT, "Covise Plugin GeoDataLoader (https://github.com/hlrs-vis/covise)"},
    };
    if (!Request().httpRequest(getRequest, readBuffer, options))
        readBuffer = "[ERROR] Failed to fetch data from Nominatim. With request: " + url;
    curl_easy_cleanup(curl);
    return readBuffer;
}

std::optional<GeoDataLoader::geoLocation> GeoDataLoader::parseCoordinates(const std::string& jsonData)
{
    rapidjson::Document document;

    if (document.Parse(jsonData.c_str()).HasParseError())
        return std::nullopt;

    if (!document.IsArray() || document.Empty())
        return std::nullopt;

    const rapidjson::Value& location = document[0];
    if (!location.HasMember("lat") || !location.HasMember("lon"))
        return std::nullopt;

    GeoDataLoader::geoLocation result;
    result.latitude  = std::stod(location["lat"].GetString());
    result.longitude = std::stod(location["lon"].GetString());

    PJ_COORD input;
    input.lp.phi = result.longitude;
    input.lp.lam = result.latitude;

    // Perform the transformation
    PJ_COORD output;
    output = proj_trans(ProjInstance, PJ_FWD, input); // Forward transformation (WGS84 -> UTM)

    result.easting = output.enu.e;
    result.northing = output.enu.n;
    result.height = 500; // Default height value

    if (location.HasMember("display_name"))
        result.displayName = location["display_name"].GetString();

    return result;
}

void GeoDataLoader::jumpToLocation(const osg::Vec3d &worldPos)
{
    double easting = worldPos.x();
    double northing = worldPos.y();
    double height = worldPos.z();
    double scale = cover->getScale();

    // --- Intersection-Test for zVal ---
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector =
        new osgUtil::LineSegmentIntersector(
            osg::Vec3d(easting, northing, 10000.0),
            osg::Vec3d(easting, northing, -10000.0));
    osgUtil::IntersectionVisitor iv(intersector);
        if (terrainNode)
            terrainNode->accept(iv);

        if (intersector->containsIntersections()) {
            height = intersector->getFirstIntersection().getLocalIntersectPoint().z();
            height += 100;
            std::cout << "Intersection found, Z = " << (height-100) << std::endl;
        } else {
            std::cout << "No intersection found, using default Z = " << height << std::endl;
        }

        // set the viewer position
        osg::Matrix mat;
        osg::Vec3 targetPos(easting, northing, height);
        std::cout << "Target Position in Meters (UTM): " << targetPos.x() << ", " << targetPos.y() << ", " << targetPos.z() << std::endl;
        targetPos += rootOffset;

        float trueNorthRadian = osg::DegreesToRadians(trueNorthDegree);
        osg::Matrix rot = osg::Matrix::rotate(-trueNorthRadian, osg::Vec3(0,0,1));

        targetPos = targetPos * rot;
        targetPos = targetPos * scale;
        mat.setTrans(-targetPos);
        cover->setXformMat(mat);
}

GeoDataLoader* GeoDataLoader::s_instance = nullptr;

GeoDataLoader::GeoDataLoader(): coVRPlugin(COVER_PLUGIN_NAME), ui::Owner("GeoData", cover->ui)
{
    assert(s_instance == nullptr);
    s_instance = this;

}
bool GeoDataLoader::init()
{
    ProjContext = proj_context_create();
    // Define the transformation from WGS84 to UTM Zone 32N (EPSG:32632)
    ProjInstance = proj_create_crs_to_crs(ProjContext, "EPSG:4326", "EPSG:32632", NULL); // EPSG:4326 is WGS84, EPSG:32632 is UTM Zone 32N
    
    geoDataMenu = new ui::Menu("GeoData", this);
    geoDataMenu->setText("GeoData");
    geoDataMenu->allowRelayout(true);

    terrainMenu = new ui::Menu(geoDataMenu, "Terrain");
    terrainMenu->setText("Terrains");
    terrainMenu->allowRelayout(true);
    
    buildingMenu = new ui::Menu(geoDataMenu, "Buildings");
    buildingMenu->setText("Buildings");
    buildingMenu->allowRelayout(true);

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
    osg::TessellationHints* hint = new osg::TessellationHints();
    hint->setDetailRatio(1.0);
    hint->setCreateBackFace(true);
    hint->setCreateFrontFace(false);
    hint->setCreateTextureCoords(true);
    //hint->setCreateNormals(true);
    osg::ShapeDrawable *_sphereDrawable = new osg:: ShapeDrawable(_Sphere, hint);
    _sphereDrawable->setColor(osg::Vec4(1,1,1,1));
    _sphereDrawable->setUseDisplayList(false); // turn off display list so that we can change the pointer length
    TexturedSphere->addDrawable(_sphereDrawable);
    osg::StateSet * stateset = TexturedSphere->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::Material* spheremtl = new osg::Material;
    spheremtl->setColorMode(osg::Material::OFF);
    spheremtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1,1,1,1));
    spheremtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1));
    spheremtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    spheremtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
    spheremtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    stateset->setAttributeAndModes(spheremtl, osg::StateAttribute::ON);
    texMat = new osg::TexMat();
    osg::Matrixd tMat;
    tMat.makeIdentity();
    tMat(0, 0) = -1;
    tMat(1, 1) = 1;
    tMat(2, 2) = 1;
    texMat->setMatrix(tMat);
    stateset->setTextureAttributeAndModes(0,texMat, osg::StateAttribute::ON);
    osg::CullFace* cF = new osg::CullFace();
    cF->setMode(osg::CullFace::BACK);
    stateset->setAttributeAndModes(cF, osg::StateAttribute::OFF);
        TexturedSphere->setStateSet(stateset);

    auto datasetEntries = configFile->entries("datasets");
    
    for (const auto& entry : datasetEntries)
    {
        std::vector<std::string> defaultArray = {"", "", "", "", "", ""};
        std::vector<std::string> datasetArray = configStringArray("datasets", entry, defaultArray)->value();
        if (datasetArray.size() < 7)
        {
            std::cerr << "GeoDataLoader: dataset entry " << entry << " has too few elements, skipping\n";
            continue;
        }

        DatasetInfo dataset;
        dataset.name = datasetArray[0];
        if (datasetArray[1] != "" && datasetArray[2] != "") // Origin given in lat/lon
        {
            double latitude = std::stod(datasetArray[1]);
            double longitude = std::stod(datasetArray[2]);
            PJ_COORD input;
                input.lp.phi = longitude;
                input.lp.lam = latitude;

                // transform
                PJ_COORD output = proj_trans(ProjInstance, PJ_FWD, input);
                dataset.easting = output.enu.e;
                dataset.northing = output.enu.n;
        }
        else if (datasetArray[3] != "" && datasetArray[4] != "") // Origin given in easting/northing
        {
            dataset.easting = std::stod(datasetArray[3]);
            dataset.northing = std::stod(datasetArray[4]);
        }
        else
        {
            std::cerr << "GeoDataLoader: Invalid dataset entry for '" << dataset.name
                  << "' (provide lat/lon or easting/northing). Using (0,0) as fallback.\n";
            dataset.easting = 0.0;
            dataset.northing = 0.0;
        }
        if (datasetArray[5].empty())
        {
            std::cerr << "GeoDataLoader: No height given for dataset '" << dataset.name << "'. Using 0.0 as fallback.\n";
            dataset.height = 0.0;
        }
        else 
        {
            dataset.height = std::stod(datasetArray[5]);
        }
        if (datasetArray[6].empty())
        {
            std::cerr << "GeoDataLoader: No trueNorth given for dataset '" << dataset.name << "'. Using 0.0 as fallback.\n";
            dataset.trueNorth = 0.0;
        }
        else
        {
            dataset.trueNorth = std::stod(datasetArray[6]);
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
            height->setValue(0.0);
            trueNorth->setValue(0.0);

            tempEastingText = "0.0";
            tempNorthingText = "0.0";
            tempHeightText = "0.0";
            tempTrueNorthText = "0.0";
        } 
        else
        {
            int index = selection - 1;
            if (index < 0 || index >= datasets.size())
            {
                easting->setValue(0.0);
                northing->setValue(0.0);
                height ->setValue(0.0);
                trueNorth->setValue(0.0);

                tempEastingText = "0.0";
                tempNorthingText = "0.0";
                tempHeightText = "0.0";
                tempTrueNorthText = "0.0";
            }
            else
            {
                const DatasetInfo& dataset = datasets[index];
                easting->setValue(dataset.easting);
                northing->setValue(dataset.northing);
                height->setValue(dataset.height);
                trueNorth->setValue(dataset.trueNorth);
                tempEastingText = std::to_string(dataset.easting);
                tempNorthingText = std::to_string(dataset.northing);
                tempHeightText = std::to_string(dataset.height);
                tempTrueNorthText = std::to_string(dataset.trueNorth);
            }
        }
    });
    tempEastingText = "0.0";
    tempNorthingText = "0.0";
    tempHeightText = "0.0";
    tempTrueNorthText = "0.0";

    easting = new ui::EditField(originGroup, "easting");
    easting->setText("Easting (m):");
    easting->setCallback([this](std::string val) 
        {
            this->tempEastingText = val;
        });
    
    northing = new ui::EditField(originGroup, "northing");
    northing->setText("Northing (m):");
    northing->setCallback([this](std::string val)
        {
            this->tempNorthingText = val;
        });
    
    height = new ui::EditField(originGroup, "height");
    height->setText("Height (m):");
    height->setCallback([this](std::string val) 
        {
            this->tempHeightText = val;
        });
    
    trueNorth = new ui::EditField(originGroup, "trueNorth");
    trueNorth->setText("True North (°):");
    trueNorth->setCallback([this](std::string val) 
        {
            this->tempTrueNorthText = val;
        });
    
    applyOffset = new ui::Button(originGroup, "applyOffset");
    applyOffset->setText("apply");
    applyOffset->setCallback([this](bool state) 
        {
            double originEasting = 0.0;
            double originNorthing = 0.0;
            double originHeight = 0.0; 
            double trueNorth = 0.0;
            if (tempEastingText != "" )
            {
                originEasting = std::stod(tempEastingText);
            }
            if (tempNorthingText != "")
            {
                originNorthing = std::stod(tempNorthingText);
            }
            if (tempHeightText != "")
            {
                originHeight = std::stod(tempHeightText);
            }
            if (tempTrueNorthText != "")
            {
                trueNorth = std::stod(tempTrueNorthText);
            }

            osg::Vec3 origin = osg::Vec3(originEasting, originNorthing, originHeight);
            setRootTransform(-origin, trueNorth);
            applyOffset->setState(false);
        });

    // create Button for each terrain file in config
    std::map<std::string, ui::Button*> terrainButtons;
    std::map<std::string, ui::Button*> buildingButtons;

    auto terrainFiles = configFile->entries("terrain");

    for (const auto& terrainFile : terrainFiles)
    {   
        std::vector<std::string> defaultArray = {"", "", ""};
        std::vector<std::string> terrainArray = configStringArray("terrain", terrainFile, defaultArray)->value();
        
        std::string terrain_name = terrainArray[0];
        std::string terrain_path = terrainArray[1];
        std::string lod_path = terrainArray[2];

        if (terrain_path != "")
        {
            ui::Button* terrainButton = new ui::Button(terrainMenu, terrain_name);
            terrainButton->setText(terrain_name);
            terrainButton->setState(false);
            terrainButton->setCallback([this, terrain_path, terrain_name](bool state)
            {
                if (state)
                {
                    if (loadedTerrains.find(terrain_name) == loadedTerrains.end())
                    {
                        terrainNode = loadTerrain(terrain_path,osg::Vec3d(0,0,0));
                        if (terrainNode)
                        {
                            loadedTerrains[terrain_name] = terrainNode;
                            rootNode->addChild(terrainNode);
                            terrainNode->setName(terrain_name);
                        }
                    }
                }
                else
                {
                    if (loadedTerrains.find(terrain_name) != loadedTerrains.end())
                    {
                        rootNode->removeChild(loadedTerrains[terrain_name]);
                        loadedTerrains.erase(terrain_name);
                    }
                }
            });
            terrainButtons[terrain_name] = terrainButton;
        }
        
        if (lod_path != "")
        {
            ui::Button* buildingButton = new ui::Button(buildingMenu, terrain_name);
            buildingButton->setText(terrain_name);
            buildingButton->setState(false);
            buildingButton->setCallback([this, lod_path, terrain_name](bool state)
            {
                if (state)
                {
                    if(loadedBuildings.find(terrain_name) == loadedBuildings.end())
                    {
                        osg::ref_ptr<osg::Node> buildingNode = loadTerrain(lod_path,osg::Vec3d(0,0,0));
                        if (buildingNode)
                        {
                            loadedBuildings[terrain_name] = buildingNode;
                            rootNode->addChild(buildingNode);
                            buildingNode->setName(terrain_name);
                        }
                    }
                }
                else
                {
                    if (loadedBuildings.find(terrain_name) != loadedBuildings.end())
                    {
                        rootNode->removeChild(loadedBuildings[terrain_name]);
                        loadedBuildings.erase(terrain_name);
                    }
                }
            });
            buildingButtons[terrain_name] = buildingButton;
        }
    }

    skyGroup = new ui::Group(geoDataMenu, "sky");
    skyGroup->setText("Sky");
    skyGroup->allowRelayout(true);

    //Restart Button
    skyButton = new ui::Button(skyGroup, "Sky");
    skyButton->setText("Sky");
    skyButton->setState(true);
    skyButton->setCallback([this](bool state) 
        {
            if (state && currentSkyNode.get()!=nullptr && currentSkyNode->getNumParents()==0)
                skyRootNode->addChild(currentSkyNode.get());
            else if(!state && currentSkyNode.get()!=nullptr)
                skyRootNode->removeChild(currentSkyNode.get());
        });
    skys = new ui::SelectionList(skyGroup, "Skys");
    skys->append("None");
    skyPath = configString("sky", "skyDir" ,"/data/Geodata/sky")->value();
    defaultSky = configInt("sky", "defaultSky", 6)->value();
    int skyNumber = 1;
    try {
        for (const auto& entry : fs::directory_iterator(skyPath))
        {
            if (entry.is_regular_file() && ((entry.path().extension() == ".wrl") || (entry.path().extension() == ".WRL"))) {
                std::string name = entry.path().filename().string();
                skyEntry se(name.substr(0, name.length() - 4),entry.path().string());
                se.fileName = name;
                skyEntries.push_back(se);
                skys->append(se.name);
                if (skyNumber == defaultSky)
                {
                    setSky(skyNumber);
                }
                skyNumber++;
            }
            if (entry.is_regular_file() && ((entry.path().extension() == ".jpg") || (entry.path().extension() == ".JPG"))) {
                std::string name = entry.path().filename().string();
                skyEntry se(name.substr(0, name.length() - 4), entry.path().string());
                se.fileName = name;
                se.type = skyEntry::texture;
                FILE* fp = fopen(entry.path().string().c_str(), "rb");
                if(fp == nullptr) {
                    printf("Can't open file %s.\n",entry.path().string().c_str());
                    return -1;
                }
                else
                {

                    int toread = 400000;
                    //fseek(fp, -toread, SEEK_END);
                    //unsigned long fsize = ftell(fp);
                    int bsize;
                    unsigned char* buf = new unsigned char[toread];
                    bsize = fread(buf, 1, toread, fp);
                    fclose(fp);

                    // Parse EXIF
                    easyexif::EXIFInfo result;
                    int code = result.parseFrom(buf, bsize);
                    delete[] buf;
                    if (code) {
                        printf("Error parsing EXIF: code %d\n", code);
                    }
                }
                skyEntries.push_back(se);
                skys->append(se.name);
                if (skyNumber == defaultSky)
                {
                    setSky(skyNumber);
                }
                skyNumber++;
            }
        }
    }
    catch (const fs::filesystem_error& err) {
        std::cerr << "Filesystem error: " << err.what() << std::endl;
    }
    catch (const std::exception& ex) {
        std::cerr << "General error: " << ex.what() << std::endl;
    }
    skys->setCallback([this](int selection)
        {
            setSky(selection);
        });
    float northAngle = 0;

    float farValue = coVRConfig::instance()->farClip();
    float scale = (farValue * 2) / 20000;
    skyRootNode->setMatrix(osg::Matrix::scale(scale, scale, scale)*osg::Matrix::rotate(northAngle,osg::Vec3(0,0,1)));
    
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

            osg::Vec3d targetPos = osg::Vec3d(geo->easting, geo->northing, geo->height);
            jumpToLocation(targetPos);

            if (!geo->displayName.empty())
                location->setText(geo->displayName);
        });
    return true;
}

// this is called if the plugin is removed at runtime
GeoDataLoader::~GeoDataLoader()
{
    s_instance = nullptr;
    proj_destroy(ProjInstance);
    proj_context_destroy(ProjContext);
}

void GeoDataLoader::message(int toWhom, int type, int length, const void* data)
{
    const char* messageData = (const char*)data;
    if(type == PluginMessageTypes::LoadTerrain)
        loadTerrain(messageData+20, osg::Vec3d(0, 0, 0)); // 20= opencover://terrain/
    else if (type == PluginMessageTypes::setSky)
    {
        int offset = 0;
        if(messageData[0]=='/' || messageData[0] == '\\')
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
    if (selection != 0)
    {


        int n = 1;
        for( auto& sky : skyEntries)
        {
            if (n == selection)
            {
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
                    osg::StateSet* stateset = TexturedSphere->getOrCreateStateSet();
                    stateset->setTextureAttributeAndModes(0, sky.skyTexture, osg::StateAttribute::ON);

                    osg::Matrixd tMat;
                    tMat.makeIdentity();
                    tMat(0, 0) = -1.0;
                    tMat(1, 1) = 1.0;
                    tMat(2, 2) = 1.0;
                    texMat->setMatrix(tMat);
                    stateset->setTextureAttributeAndModes(0,texMat, osg::StateAttribute::ON);
                    skyRootNode->addChild(sky.skyNode);

                }
                currentSkyNode = sky.skyNode;
            }
            n++;
        }

    }
}

void GeoDataLoader::setSky(std::string fileName)
{
    int n = 0;
    for (const auto& sky : skyEntries)
    {
        if (sky.fileName == fileName)// already have this file in the list
        {
            setSky(n+1); 
        }
        n++;
    }
    std::filesystem::path path(fileName);
    std::string fn = path.filename().string();
    skyEntry se(fn.substr(0, fn.length() - 4), fileName);
    skyEntries.push_back(se);
    skys->append(se.name);
}

void GeoDataLoader::setRootTransform(const osg::Vec3& off, float trueNorthDeg)
{
    rootOffset = off;
    trueNorthDegree = trueNorthDeg;

    float trueNorthRad = osg::DegreesToRadians(trueNorthDegree);
    osg::Matrix trans = osg::Matrix::translate(rootOffset);
    osg::Matrix rot = osg::Matrix::rotate(-trueNorthRad, osg::Vec3(0,0,1));
    rootNode->setMatrix(trans * rot);
}

bool GeoDataLoader::update()
{
    const osg::Matrix &m = cover->getObjectsXform()->getMatrix();
    //double x = m(0, 0);
    //double y = m(1, 0);
    //northAngle = -atan2(y, x);
    //skyRootNode->setMatrix(osg::Matrix::scale(1000, 1000, 1000) * osg::Matrix::rotate(northAngle, osg::Vec3(0, 0, 1)));
    float farValue = coVRConfig::instance()->farClip();
    float scale = ((farValue) /20000)+500; // scale the sphere so that it stays a bit closer than the far clipping plane.
    skyRootNode->setMatrix(osg::Matrix::scale(scale, scale, scale) * osg::Matrix::rotate(cover->getObjectsXform()->getMatrix().getRotate()));
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
    //pager->setIncrementalCompileOperation(new osgUtil::IncrementalCompileOperation());
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
        /*osgTerrain::Terrain* terrainNode = dynamic_cast<osgTerrain::Terrain*>(terrain);
      if(terrainNode) {
         std::cout << "Casted to terrainNode!" << std::endl;
         std::cout << "Num children: " << terrainNode->getNumChildren() << std::endl;
         if(terrainNode->getNumChildren()>0) {
            terrain = terrainNode->getChild(0);
         }
      }*/
        terrain->setDataVariance(osg::Object::DYNAMIC);

        osg::StateSet *terrainStateSet = terrain->getOrCreateStateSet();

        terrainStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
        terrainStateSet->setMode(GL_LIGHT0, osg::StateAttribute::ON);
        //terrainStateSet->setMode ( GL_LIGHT1, osg::StateAttribute::ON);

        //osgUtil::Optimizer optimizer;
        //optimizer.optimize(terrain);

        terrain->setNodeMask(terrain->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

        //std::vector<BoundingArea> voidBoundingAreaVector;
        //voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(506426.839,5398055.357),osg::Vec2(508461.865,5399852.0)));
        osgTerrain::TerrainTile::setTileLoadedCallback(new CutGeometry(this));

        osg::PositionAttitudeTransform *terrainTransform = new osg::PositionAttitudeTransform();
        terrainTransform->setName("Terrain");
        terrainTransform->addChild(terrain);
        terrainTransform->setPosition(localOffset);

        const osg::BoundingSphere &terrainBS = terrain->getBound();
        std::cout << "Terrain BB: center: (" << terrainBS.center()[0] << ", " << terrainBS.center()[1] << ", " << terrainBS.center()[2] << "), radius: " << terrainBS.radius() << std::endl;

        return terrainTransform;
    }
    else
    {
        return nullptr;
    }
}

bool GeoDataLoader::addLayer(std::string filename)
{
#if GDAL_VERSION_MAJOR<2

#else
        if (GDALGetDriverCount() == 0)
            GDALAllRegister();

        // Try to open data source
        GDALDataset* poDS  = (GDALDataset*) GDALOpenEx( filename.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL );
#endif
        /*
    if (poDS == NULL)
    {
        std::cout << "GeoDataLoader: Could not open layer " << filename << "!" << std::endl;
        return false;
    }

    for (int layerIt = 0; layerIt < poDS->GetLayerCount(); ++layerIt)
    {
    }*/

    return true;
}

COVERPLUGIN(GeoDataLoader)
