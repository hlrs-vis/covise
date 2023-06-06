/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2023 HLRS  **
 **                                                                          **
 ** Description: Urban Tempo OpenCOVER Plugin                                **
 **                                                                          **
 **                                                                          **
 ** Author: Kilian Türk                                                      **
 **                                                                          **
 ** History:  								                                 **
 ** Feb 2023  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "UrbanTempo.h"
#include <cover/coVRPluginSupport.h>
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h> 
#include <rapidjson/rapidjson.h>
#include <iostream>
#include <osgDB/ReadFile>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>
#include <osg/Referenced>
#include <fstream>
#include <sstream>
#include <osg/ComputeBoundsVisitor>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3d>
#include <unordered_map>

using namespace opencover;


TreeModel::TreeModel(std::string configName)
{


    std::vector<double> defaultVec{ 0.0, 0.0, 0.0 };
    auto configSpeciesName = UrbanTempo::instance()->configString(configName, "species_name", "default");
    auto configTreeModel = UrbanTempo::instance()->configString(configName, "model_path", "default");
    auto configSeason = UrbanTempo::instance()->configString(configName, "season", "default");
    auto configLOD = UrbanTempo::instance()->configInt(configName, "LOD", 0);
    auto configRotation = UrbanTempo::instance()->configFloatArray(configName, "rotation", defaultVec);
    auto configTranslation = UrbanTempo::instance()->configFloatArray(configName, "translation", defaultVec);
    osg::Vec3 translation(configTranslation->value()[0], configTranslation->value()[1], configTranslation->value()[2]);

    speciesName = *configSpeciesName;
    modelPath = *configTreeModel;
    model = osgDB::readNodeFile(modelPath);
    season = UrbanTempo::stringToSeason(*configSeason);
    transform = osg::Matrixd::rotate(osg::DegreesToRadians((*configRotation)[0]), osg::Vec3(1.0f, 0.0f, 0.0f),
        osg::DegreesToRadians((*configRotation)[1]), osg::Vec3(0.0f, 1.0f, 0.0f),
        osg::DegreesToRadians((*configRotation)[2]), osg::Vec3(0.0f, 0.0f, 1.0f));
    transform.postMultTranslate(translation);
    if (model)
    {
        osg::ComputeBoundsVisitor cbv;
        model->accept(cbv);
        osg::BoundingBox bb = cbv.getBoundingBox();
        osg::Vec3 size(bb.xMax() - bb.xMin(), bb.yMax() - bb.yMin(), bb.zMax() - bb.zMin());
        if((*configRotation)[0]!=0.0)
            height = size[1]; // hack, if y is up, use y as height, otherwise z
        else
            height = size[2];
    }
}

TreeModel::~TreeModel()
{
}

UrbanTempo *UrbanTempo::plugin = nullptr;

UrbanTempo::UrbanTempo()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    plugin = this;
}

bool UrbanTempo::init()
{
    auto url_ptr = configString("general", "url_api", "default");
    url = *url_ptr;

    auto api = configBool("general", "use_api", false);
    if (*api)
    {
        request();
        simplifyResponse();
        // saveStringToFile(simpleResponse);
        printResponseToConfig();
    }
    // printInformation();
    setupPluginNode();
    setTrees();

    return true;
}

bool UrbanTempo::destroy()
{
    cover->getObjectsRoot()->removeChild(pluginNode);
    return true;
}

void UrbanTempo::setupPluginNode()
{
    pluginNode = new osg::Group;
    cover->getObjectsRoot()->addChild(pluginNode);
    pluginNode->setName(getName());
}

static size_t writeCallback(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    std::string *response = static_cast<std::string *>(userdata);
    response->append(ptr, size * nmemb);
    return size * nmemb;
}

void UrbanTempo::request()
{
    CURL *curl = curl_easy_init();
    if (!curl)
    {
        std::cerr << "Failed to create cURL handle" << std::endl;
    }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "GET");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK)
    {
        std::cerr << "Failed to execute HTTP request: " << curl_easy_strerror(res) << std::endl;
    }

    curl_easy_cleanup(curl);
}

void UrbanTempo::simplifyResponse()
{
    rapidjson::Document document;
    document.Parse(response.c_str());
    rapidjson::Document simpleDocument;
    simpleDocument.SetArray();
    rapidjson::Document::AllocatorType &allocator = simpleDocument.GetAllocator();

    for (rapidjson::Value::ValueIterator itr = document["features"].Begin(); itr != document["features"].End(); ++itr)
    {
        if ((*itr)["attributes"]["species_name"].IsString() && (*itr)["geometry"]["x"].IsDouble() && (*itr)["geometry"]["y"].IsDouble())
        {
            rapidjson::Value entry(rapidjson::kObjectType);
            rapidjson::Value &value = (*itr)["attributes"]["species_name"];
            entry.AddMember("species_name", value, allocator);

            if ((*itr)["attributes"]["height"].IsInt())
            {
                value = (*itr)["attributes"]["height"].GetInt();
                entry.AddMember("height", value, allocator);
            }
            else
            {
                entry.AddMember("height", rapidjson::Value(rapidjson::kNullType), allocator);
            }

            value = (*itr)["geometry"]["x"].GetDouble();
            entry.AddMember("x", value, allocator);
            value = (*itr)["geometry"]["y"].GetDouble();
            entry.AddMember("y", value, allocator);
            simpleDocument.PushBack(entry, allocator);
        }
    }

    simpleResponse = documentToString(simpleDocument);
}

void UrbanTempo::saveStringToFile(const std::string &string)
{
    std::ofstream file("trees.json");
    if (file.is_open())
    {
        file << string;
        file.close();
    }
}

std::string UrbanTempo::readJSONFromFile(const std::string &path)
{
    std::ifstream file(path);
    std::stringstream filecontent;
    if (file.is_open())
    {
        filecontent << file.rdbuf();
    }
    return filecontent.str();
}

void UrbanTempo::setTrees()
{
    // read in default tree model
    //osg::ref_ptr<osg::Node> model;
    

    int i = 1;
    while (true)
    {
        auto configSpeciesName = configString("treeModel" + std::to_string(i), "species_name", "default");
        std::string speciesName = *configSpeciesName; if (speciesName == "default")
        {
            treeModels.emplace_back(std::make_unique<TreeModel>("treeModelDefault"));

            defaultTreeIterator = treeModels.end();
            defaultTreeIterator--;
            break;
        }
        treeModels.emplace_back(std::make_unique<TreeModel>("treeModel" + std::to_string(i)));
        
        i++;
    }

    // create transform node to translate all placed trees
    osg::ref_ptr<osg::MatrixTransform> rootTransform(new osg::MatrixTransform);
    rootTransform->setName("rootTransform");
    pluginNode->addChild(rootTransform);
    osg::Matrix rootMatrix;
    auto configOffsetX = configFloat("offset", "x", 0.0);
    auto configOffsetY = configFloat("offset", "y", 0.0);
    rootMatrix.makeTranslate(osg::Vec3d(*configOffsetX, *configOffsetY, 0.0));
    rootTransform->setMatrix(rootMatrix);

    auto configDefaultHeight = configFloat("general", "default_height", 10);
    double defaultHeight = *configDefaultHeight;

    // add trees one after another with information from config file
    GDALAllRegister();
    auto configGeotiffpath = configString("general", "geotiff_path", "");
    std::string geotiffpath = *configGeotiffpath;
    openImage(geotiffpath);

    i = 1;
    while (true)
    {
        auto configSpeciesName = configString("tree" + std::to_string(i), "species_name", "default");
        std::string speciesName = *configSpeciesName;
        int condition = speciesName.compare("default");
        if (!condition)
            break;
        osg::ref_ptr<osg::MatrixTransform> treeTransform(new osg::MatrixTransform);
        treeTransform->setName("treeTransform"+std::to_string(i));
        rootTransform->addChild(treeTransform);

        // choose correct tree model and add it to the scenegraph
        auto itr = std::find_if(treeModels.begin(), treeModels.end(), [&](const std::unique_ptr<TreeModel>& treeModel){
            return treeModel->speciesName == speciesName;
        });
        if (itr == treeModels.end())
        {
            itr = defaultTreeIterator;
        }
        if (((*itr)->model) == nullptr)
        {
            itr = defaultTreeIterator;
        }
        /*if ((*itr)->season != stringToSeason(configSeason->value()))
        {
            model = defaultTreeModel;
            rotation = osg::Vec3d(configDefaultRotation->value()[0], configDefaultRotation->value()[1], configDefaultRotation->value()[2]);
            translate = osg::Vec3d(configDefaultTranslation->value()[0], configDefaultTranslation->value()[1], configDefaultTranslation->value()[2]);
            // std::cout << "default tree model" << std::endl;
        }
        else
        {
            model = treeModels[index]->model;
            rotation = treeModels[index]->rotation;
            translate = treeModels[index].translation;
            // std::cout << "index: " << index << "; Model: " << treeModels[index].speciesName << std::endl;
        }*/
        if((*itr)->model)
        {
            treeTransform->addChild((*itr)->model);

            // read tree position and get altitude from geotiff
            auto configTreeX = configFloat("tree" + std::to_string(i), "x", 0);
            auto configTreeY = configFloat("tree" + std::to_string(i), "y", 0);
            float treeAltitude = getAlt(*configTreeX, *configTreeY);

            // set height for specific trees or get default height, if no height is specified and scale models to height
            double height;
            // auto configHeight = configFloat("tree" + std::to_string(i), "height", defaultHeight); // this will lead to an error
            // height = *configHeight;
            auto configHeight = configFloat("tree" + std::to_string(i), "height", 0.0);
            if (*configHeight == 0.0)
                height = defaultHeight;
            else
                height = *configHeight;

            // scale tree model to height specifies in config
            // osg::Vec3 center(bb.xMin() + size.x() / 2.0f, bb.yMin() + size.y() / 2.0f, bb.zMin() + size.z() / 2.0f);
            ;
            double scaleFactor = height / (*itr)->height;

            treeTransform->setMatrix(osg::Matrixd::scale(osg::Vec3d(scaleFactor, scaleFactor, scaleFactor)) * (*itr)->transform * osg::Matrix::translate(osg::Vec3d(*configTreeX, *configTreeY, treeAltitude)));
        }

        i++;
    }
    closeImage();
}

std::string UrbanTempo::documentToString(const rapidjson::Document &document)
{
    rapidjson::StringBuffer stringbuffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(stringbuffer);
    document.Accept(writer);
    return stringbuffer.GetString();
}

void UrbanTempo::printResponseToConfig()
{
    rapidjson::Document document;
    document.Parse(simpleResponse.c_str());

    int i = 1;
    for (rapidjson::Value::ValueIterator itr = document.Begin(); itr != document.End(); ++itr)
    {
        if ((*itr)["species_name"].IsString() && (*itr)["x"].IsDouble() && (*itr)["y"].IsDouble())
        {

            if ((*itr)["height"].IsInt())
            {
                int heightResponse = (*itr)["height"].GetInt();
                auto heightConfig = configFloat("tree" + std::to_string(i), "height", 0.0);
                *heightConfig = heightResponse;
            }

            std::string speciesNameResponse = (*itr)["species_name"].GetString();
            auto speciesNameConfig = configString("tree" + std::to_string(i), "species_name", "default");
            *speciesNameConfig = speciesNameResponse;

            double xResponse = (*itr)["x"].GetDouble();
            auto xConfig = configFloat("tree" + std::to_string(i), "x", 0);
            *xConfig = xResponse;

            double yResponse = (*itr)["y"].GetDouble();
            auto yConfig = configFloat("tree" + std::to_string(i), "y", 0);
            *yConfig = yResponse;

            config()->save();
            i++;
        }
    }
}

float UrbanTempo::getAlt(double x, double y)
{
    int col = int((x - xOrigin) / pixelWidth);
    int row = int((yOrigin - y) / pixelHeight);
    if (col < 0)
        col = 0;
    if (col >= cols)
        col = cols - 1;
    if (row < 0)
        row = 0;
    if (row >= rows)
        row = rows - 1;
    return rasterData[col + (row * cols)];
    float *pafScanline;
    int nXSize = heightBand->GetXSize();

    delete[] pafScanline;
    pafScanline = new float[nXSize];
    auto err = heightBand->RasterIO(GF_Read, (int)x, (int)y, 1, 1,
                                    pafScanline, nXSize, 1, GDT_Float32,
                                    0, 0);
    float height = pafScanline[0];
    delete[] pafScanline;

    if (err != CE_None)
    {
        std::cerr << "MapDrape::getAlt: error" << std::endl;
        return 0.f;
    }

    return height;
}

void UrbanTempo::openImage(std::string &name)
{
    heightDataset = (GDALDataset *)GDALOpen(name.c_str(), GA_ReadOnly);
    if (heightDataset != NULL)
    {
        int nBlockXSize, nBlockYSize;
        int bGotMin, bGotMax;
        double adfMinMax[2];
        double adfGeoTransform[6];

        printf("Size is %dx%dx%d\n", heightDataset->GetRasterXSize(), heightDataset->GetRasterYSize(), heightDataset->GetRasterCount());
        if (heightDataset->GetGeoTransform(adfGeoTransform) == CE_None)
        {
            printf("Origin = (%.6f,%.6f)\n",
                   adfGeoTransform[0], adfGeoTransform[3]);
            printf("Pixel Size = (%.6f,%.6f)\n",
                   adfGeoTransform[1], adfGeoTransform[5]);
        }
        int numRasterBands = heightDataset->GetRasterCount();

        heightBand = heightDataset->GetRasterBand(1);
        heightBand->GetBlockSize(&nBlockXSize, &nBlockYSize);
        cols = heightDataset->GetRasterXSize();
        rows = heightDataset->GetRasterYSize();
        double transform[100];
        heightDataset->GetGeoTransform(transform);

        xOrigin = transform[0];
        yOrigin = transform[3];
        pixelWidth = transform[1];
        pixelHeight = -transform[5];
        delete[] rasterData;
        rasterData = new float[cols * rows];
        float *pafScanline;
        int nXSize = heightBand->GetXSize();
        pafScanline = (float *)CPLMalloc(sizeof(float) * nXSize);
        for (int i = 0; i < rows; i++)
        {
            if (heightBand->RasterIO(GF_Read, 0, i, nXSize, 1,
                                     pafScanline, nXSize, 1, GDT_Float32,
                                     0, 0) == CE_Failure)
            {
                std::cerr << "MapDrape::openImage: GDALRasterBand::RasterIO failed" << std::endl;
                break;
            }
            memcpy(&(rasterData[(i * cols)]), pafScanline, nXSize * sizeof(float));
        }

        if (heightBand->ReadBlock(0, 0, rasterData) == CE_Failure)
        {
            std::cerr << "MapDrape::openImage: GDALRasterBand::ReadBlock failed" << std::endl;
            return;
        }

        adfMinMax[0] = heightBand->GetMinimum(&bGotMin);
        adfMinMax[1] = heightBand->GetMaximum(&bGotMax);
        if (!(bGotMin && bGotMax))
            GDALComputeRasterMinMax((GDALRasterBandH)heightBand, TRUE, adfMinMax);

        printf("Min=%.3fd, Max=%.3f\n", adfMinMax[0], adfMinMax[1]);
    }
}

void UrbanTempo::closeImage()
{
    if (heightDataset)
        GDALClose(heightDataset);
}

void UrbanTempo::printInformation()
{
    std::vector<std::string> speciesNames;

    int i = 1;
    while (true)
    {
        auto configSpeciesName = configString("tree" + std::to_string(i), "species_name", "default");
        std::string speciesName = *configSpeciesName;
        int condition = speciesName.compare("default");
        if (!condition)
            break;
        speciesNames.push_back(speciesName);
        i++;
    }

    std::unordered_map<std::string, int> stringCounts;

    for (const auto& str : speciesNames) {
        stringCounts[str]++;
    }

    std::vector<std::pair<std::string, int>> sortedResults(stringCounts.begin(), stringCounts.end());
    std::sort(sortedResults.begin(), sortedResults.end(),
              [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                  return a.second > b.second;
              });

    // print the sorted results
    for (const auto& pair : sortedResults) {
        std::cout << "String: " << pair.first << ", Count: " << pair.second << std::endl;
    }

}

Season UrbanTempo::stringToSeason(const std::string& string)
{
    if (string == "winter")
        return Season::Winter;
    else if (string == "spring")
        return Season::Spring;
    else if (string == "summer")
        return Season::Summer;
    else if (string == "fall")
        return Season::Fall;
    else
        return Season::Summer;
}

COVERPLUGIN(UrbanTempo)
