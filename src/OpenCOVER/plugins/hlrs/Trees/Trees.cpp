/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2023 HLRS  **
 **                                                                          **
 ** Description: Trees OpenCOVER Plugin                                      **
 **                                                                          **
 **                                                                          **
 ** Author: Kilian Türk                                                      **
 **                                                                          **
 ** History:  								                                 **
 ** Feb 2023  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Trees.h"
#include <cover/coVRPluginSupport.h>
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>  //can remove one writer later probably
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

using namespace opencover;

bool Trees::init()
{
    auto url_ptr = configString("general", "url_api", "default");
    url = *url_ptr;
    
    auto api = configBool("general", "use_api", false);
    if(*api)
    {
        request();
        simplifyResponse();
        saveStringToFile(simpleResponse);
        printResponseToConfig();
    }
    // testFunc();
    setupPluginNode();
    setTrees();

    return true;
}

void Trees::setupPluginNode()
{
    pluginNode = new osg::Group;
    cover->getObjectsRoot()->addChild(pluginNode);
    pluginNode->setName(getName());
}

static size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    std::string* response = static_cast<std::string*>(userdata);
    response->append(ptr, size * nmemb);
    return size * nmemb;
}

void Trees::request()
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

void Trees::simplifyResponse()
{
    rapidjson::Document document;
    document.Parse(response.c_str());
    rapidjson::Document simpleDocument;
    simpleDocument.SetArray();
    rapidjson::Document::AllocatorType& allocator = simpleDocument.GetAllocator();

    for (rapidjson::Value::ValueIterator itr = document["features"].Begin(); itr != document["features"].End(); ++itr)
    {
        if((*itr)["attributes"]["species_name"].IsString()
            && (*itr)["geometry"]["x"].IsDouble()
            && (*itr)["geometry"]["y"].IsDouble()) 
        {
            rapidjson::Value entry(rapidjson::kObjectType);
            rapidjson::Value& value = (*itr)["attributes"]["species_name"];
            entry.AddMember("species_name", value, allocator);

            if((*itr)["attributes"]["height"].IsInt())
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

void Trees::saveStringToFile(const std::string& string)
{
    std::ofstream file("trees.json");
    if(file.is_open())
    {
        file << string;
        file.close();
    }
}

std::string Trees::readJSONFromFile(const std::string& path)
{
    std::ifstream file(path);
    std::stringstream filecontent;
    if(file.is_open())
    {
        filecontent << file.rdbuf();
    }
    return filecontent.str();
}

void Trees::setTrees()
{
    // read in default tree model
    osg::ref_ptr<osg::Node> model;
    auto defaultTreeModelPtr = configString("treeModelDefault", "model_path", "default");
    osg::ref_ptr<osg::Node> defaultTreeModel = osgDB::readNodeFile(*defaultTreeModelPtr);

    // get list with all species_name entries in treemodels and load all models
    std::vector<std::string> speciesNames;
    std::vector<osg::ref_ptr<osg::Node>> treeModels;
    int i = 1;
    while (true)
    {
        auto value = configString("treeModel" + std::to_string(i), "species_name", "default");
        auto treeModelPtr = configString("treeModel" + std::to_string(i), "model_path", "default");
        std::string speciesName = *value;
        int condition = speciesName.compare("default");
        if(!condition) 
        {
            break;
        }
        speciesNames.push_back(speciesName);
        osg::ref_ptr<osg::Node> treeModel = osgDB::readNodeFile(*treeModelPtr);
        treeModels.push_back(treeModel);
        i++;
    }

    // create transform node to translate all placed trees
    osg::ref_ptr<osg::MatrixTransform> transform(new osg::MatrixTransform);
    transform->setName("rootTransform");
    pluginNode->addChild(transform);
    osg::Matrix rootMatrix;
    auto configOffsetX = configFloat("offset", "x", 0.0);
    auto configOffsetY = configFloat("offset", "y", 0.0);
    rootMatrix.makeTranslate(osg::Vec3d(*configOffsetX, *configOffsetY, 0.0));
    transform->setMatrix(rootMatrix);

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
        if(!condition) 
        {
            break;
        }
        osg::ref_ptr<osg::PositionAttitudeTransform> treeTransform(new osg::PositionAttitudeTransform);
        treeTransform->setName("treeTransform");
        transform->addChild(treeTransform);

        // choose correct tree model
        auto it = std::find(speciesNames.begin(), speciesNames.end(), speciesName);
        if (it == speciesNames.end())
        {
            model = defaultTreeModel;
            std::cout << "default tree model" << std::endl;
        } else
        {
            auto index = std::distance(speciesNames.begin(), it);
            model = treeModels[index];
            std::cout << "index: " << index << "; Model: " << speciesNames[index] << std::endl;
        }
        treeTransform->addChild(model);
        auto x = configFloat("tree" + std::to_string(i), "x", 0);
        auto y = configFloat("tree" + std::to_string(i), "y", 0);
        float altitude = getAlt(*x, *y);
        treeTransform->setPosition(osg::Vec3d(*x, *y, altitude));
        std::cout << "added " << speciesName << " at position " << *x << ", " << *y << std::endl;

        // scale tree model to height specifies in config 
        osg::ComputeBoundsVisitor cbv;
        model->accept(cbv);
        osg::BoundingBox bb = cbv.getBoundingBox();

        osg::Vec3 size(bb.xMax() - bb.xMin(), bb.yMax() - bb.yMin(), bb.zMax() - bb.zMin());
        osg::Vec3 center(bb.xMin() + size.x()/2.0f, bb.yMin() + size.y()/2.0f, bb.zMin() + size.z()/2.0f);

        std::cout << "bounding box: " << size[0] << ", " << size[1] << ", " << size[2] << std::endl;

        // set height for specific trees or get default height, if no height is specified and scale models to height
        double height;
        auto configHeight = configFloat("tree" + std::to_string(i), "height", 0.0); //todo: set better default value
        if (*configHeight == 0.0)
        {
            height = defaultHeight;
        } else
        {
            height = *configHeight;
        }

        double scaleFactor = height / size[2];
        std::cout << "scaled size: " << scaleFactor * size[0] << ", " << scaleFactor * size[1] << ", " << scaleFactor * size[2] << std::endl;
        
        treeTransform->setScale(osg::Vec3d(scaleFactor, scaleFactor, scaleFactor));

        // std::cout << "used Model: " << speciesNames[i] << std::endl;
        i++;
    }
    closeImage();
}

std::string Trees::documentToString(const rapidjson::Document& document)
{
    rapidjson::StringBuffer stringbuffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(stringbuffer);
    document.Accept(writer);
    return stringbuffer.GetString();
}

void Trees::addSeason(std::string& season)
{
    rapidjson::Document document;
    document.Parse(simpleResponse.c_str());
    rapidjson::Document::AllocatorType& allocator = document.GetAllocator();

    for (rapidjson::Value::ValueIterator itr = document.Begin(); itr != document.End(); ++itr)
    {
        rapidjson::Value value(rapidjson::kStringType);
        value.SetString(season.c_str(), allocator);
        (*itr).AddMember("season_style", value, allocator);
    }

    simpleResponse = documentToString(document);
}

void Trees::printResponseToConfig()
{
    rapidjson::Document document;
    document.Parse(simpleResponse.c_str());

    int idx = 1;
    for (rapidjson::Value::ValueIterator itr = document.Begin(); itr != document.End(); ++itr)
    {
        if((*itr)["species_name"].IsString()
            && (*itr)["x"].IsDouble()
            && (*itr)["y"].IsDouble()) 
        {

            if((*itr)["height"].IsInt())
            {
                int heightResponse = (*itr)["height"].GetInt();
                auto heightConfig = configFloat("tree" + std::to_string(idx), "height", 0.0); // todo: set better default value 
                *heightConfig = heightResponse;
            }
            else
            {
                // entweder hier default reinschreiben oder aktuelles verhalten mit defaulheight in general benutzen
            }

            std::string speciesNameResponse = (*itr)["species_name"].GetString();
            auto speciesNameConfig = configString("tree" + std::to_string(idx), "species_name", "default");
            *speciesNameConfig = speciesNameResponse;

            double xResponse = (*itr)["x"].GetDouble();
            auto xConfig = configFloat("tree" + std::to_string(idx), "x", 0);
            *xConfig = xResponse;
            
            double yResponse = (*itr)["y"].GetDouble();
            auto yConfig = configFloat("tree" + std::to_string(idx), "y", 0);
            *yConfig = yResponse;

            config()->save();
            idx++;
        }
    }
}

float Trees::getAlt(double x, double y)
{
    int col = int((x - xOrigin) / pixelWidth);
    int row = int((yOrigin - y) / pixelHeight);
    if (col < 0)
        col = 0;
    if (col >= cols)
        col = cols-1;
    if (row < 0)
        row = 0;
    if (row >= rows)
        row = rows - 1;
    return rasterData[col + (row*cols)];
	float *pafScanline;
	int   nXSize = heightBand->GetXSize();

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

void Trees::openImage(std::string &name)
{
    heightDataset = (GDALDataset *)GDALOpen(name.c_str(), GA_ReadOnly);
    if (heightDataset != NULL)
    {
        int             nBlockXSize, nBlockYSize;
        int             bGotMin, bGotMax;
        double          adfMinMax[2];
        double        adfGeoTransform[6];

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
        rasterData = new float[cols*rows];
        float *pafScanline;
        int   nXSize = heightBand->GetXSize();
        pafScanline = (float *)CPLMalloc(sizeof(float)*nXSize);
        for (int i = 0; i < rows; i++)
        {
            if (heightBand->RasterIO(GF_Read, 0, i, nXSize, 1,
                pafScanline, nXSize, 1, GDT_Float32,
                0, 0) == CE_Failure)
            {
                std::cerr << "MapDrape::openImage: GDALRasterBand::RasterIO failed" << std::endl;
                break;
            }
            memcpy(&(rasterData[(i*cols)]), pafScanline, nXSize * sizeof(float));
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

void Trees::closeImage()
{
    if (heightDataset)
        GDALClose(heightDataset);
}

COVERPLUGIN(Trees)