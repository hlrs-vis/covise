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
#include "Tree.h"
#include "TreeModel.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRTui.h>
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
#include <HTTPClient/CURL/request.h>
#include <HTTPClient/CURL/methods.h>

#include <osg/io_utils>

using namespace opencover;

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
    getGeneralConfigData();
    coVRConfig::instance()->setLODScale(100);
    setupScenegraph();
    setTreeModels();
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

void UrbanTempo::request()
{
    httpclient::curl::GET get(url);
    if (!httpclient::curl::Request().httpRequest(get, response))
    {
        std::cerr << "Failed to fetch data from UrbanTempo" << std::endl;
    }
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

void UrbanTempo::getGeneralConfigData()
{
    auto configDefaultHeight = configFloat("general", "default_height", 10);
    defaultHeight = *configDefaultHeight;

    auto configGeotiffpath = configString("general", "geotiff_path", "");
    geotiffpath = *configGeotiffpath;
}

void UrbanTempo::setTreeModels()
{
    int i = 1;
    while (true)
    {
        auto configSpeciesName = configString("treeModel" + std::to_string(i), "species_name", "default");
        std::string speciesName = *configSpeciesName; if (speciesName == "default")
        {
            treeModels.emplace_back(std::make_shared<TreeModel>("treeModelDefault"));

            defaultTreeIterator = treeModels.end();
            defaultTreeIterator--;
            break;
        }
        treeModels.emplace_back(std::make_shared<TreeModel>("treeModel" + std::to_string(i)));
        
        i++;
    }
}

void UrbanTempo::setupScenegraph()
{
    // osg::ref_ptr<osg::MatrixTransform> rootTransform(new osg::MatrixTransform);
    rootTransform = new osg::MatrixTransform;
    rootTransform->setName("rootTransform");
    osg::Matrix rootMatrix;
    auto configOffsetX = configFloat("offset", "x", 0.0);
    auto configOffsetY = configFloat("offset", "y", 0.0);
    rootMatrix.makeTranslate(osg::Vec3d(*configOffsetX, *configOffsetY, 0.0));
    rootTransform->setMatrix(rootMatrix);
    pluginNode->addChild(rootTransform);
}

void UrbanTempo::setTrees()
{
    GDALAllRegister();
    openImage(geotiffpath);

    int i = 1;
    while (true)
    {
        auto configSpeciesName = configString("tree" + std::to_string(i), "species_name", "default");
        std::string speciesName = *configSpeciesName;
        if (speciesName == "default")
        {
            break;
        }
        trees.emplace_back(std::make_unique<Tree>("tree" + std::to_string(i)));
        
        osg::ref_ptr<osg::MatrixTransform> treeTransform(new osg::MatrixTransform);
        trees[i-1]->sceneGraphNode = treeTransform;
        treeTransform->setName("treeTransform"+std::to_string(i));
        rootTransform->addChild(treeTransform);

        // choose correct tree model and add it to the scenegraph
        auto itr = std::find_if(treeModels.begin(), treeModels.end(), [&](const std::shared_ptr<TreeModel>& treeModel){
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

        if((*itr)->model)
        {
            treeTransform->addChild((*itr)->model);
            trees[i-1]->treeModel = *itr;

            // read tree position and get altitude from geotiff
            auto treeXPos = trees[i-1]->xPos;
            auto treeYPos = trees[i-1]->yPos;
            float treeAltitude = getAlt(treeXPos, treeYPos);
            trees[i-1]->setAltitude(treeAltitude);
            // trees[i-1]->setAltitude(getAlt(trees[i-1]->xPos, trees[i-1]->yPos);

            // set height for specific trees or get default height, if no height is specified and scale models to height
            auto height = trees[i-1]->height;

            double scaleFactor = height / (*itr)->height;
            trees[i-1]->scale = scaleFactor;

            treeTransform->setMatrix(osg::Matrixd::scale(osg::Vec3d(scaleFactor, scaleFactor, scaleFactor)) * (*itr)->transform * osg::Matrix::translate(osg::Vec3d(treeXPos, treeYPos, treeAltitude)));
            trees[i-1]->setTransform();
        }

        trees[i-1]->setTreeModelLODs();
        i++;
    }
    closeImage();
}

// void UrbanTempo::setTrees()
// {
//     // read in default tree model
//     //osg::ref_ptr<osg::Node> model;
    
//     // int i = 1;
//     // while (true)
//     // {
//     //     auto configSpeciesName = configString("treeModel" + std::to_string(i), "species_name", "default");
//     //     std::string speciesName = *configSpeciesName; if (speciesName == "default")
//     //     {
//     //         treeModels.emplace_back(std::make_shared<TreeModel>("treeModelDefault"));

//     //         defaultTreeIterator = treeModels.end();
//     //         defaultTreeIterator--;
//     //         break;
//     //     }
//     //     treeModels.emplace_back(std::make_shared<TreeModel>("treeModel" + std::to_string(i)));
        
//     //     i++;
//     // }

//     setupScenegraph();
//     setTreeModels();

//     int i = 1;
//     while (true)
//     {
//         auto configSpeciesName = configString("tree" + std::to_string(i), "species_name", "default");
//         std::string speciesName = *configSpeciesName;
//         if (speciesName == "default")
//         {
//             // trees.emplace_back(std::make_unique<TreeModel>("treeModelDefault"));

//             // defaultTreeIterator = treeModels.end();
//             // defaultTreeIterator--;
//             break;
//         }
//         trees.emplace_back(std::make_unique<Tree>("tree" + std::to_string(i)));

//         // auto configTreeX = UrbanTempo::instance()->configFloat("tree" + std::to_string(i), "x", 0);
//         // double xPos = *configTreeX;
//         // auto configTreeY = UrbanTempo::instance()->configFloat("tree" + std::to_string(i), "y", 0);
//         // double yPos = *configTreeY;
//         // auto configHeight = UrbanTempo::instance()->configFloat("tree" + std::to_string(i), "height", 0.0);
//         // double height = *configHeight;
//         // if (height == 0.0)
//         //     height = UrbanTempo::instance()->defaultHeight;

//         // auto configTreeX = UrbanTempo::instance()->configFloat("tree" + std::to_string(i), "x", 0);
//         // double xPos = *configTreeX;
//         // treeBuilder.setXPos(xPos);
//         // auto configTreeY = UrbanTempo::instance()->configFloat("tree" + std::to_string(i), "y", 0);
//         // double yPos = *configTreeY;
//         // treeBuilder.setYPos(yPos);
//         // auto configHeight = UrbanTempo::instance()->configFloat("tree" + std::to_string(i), "height", 0.0);
//         // double height = *configHeight;
//         // if (height == 0.0)
//         //     height = UrbanTempo::instance()->defaultHeight;
//         // treeBuilder.setHeight(height);

//         // trees.emplace_back(std::make_unique<Tree>(speciesName, xPos, yPos, height));
//         i++;
//     }

//     // add trees one after another with information from config file
//     GDALAllRegister();
//     openImage(geotiffpath);

//     // i = 1;
//     // while (i >= trees.size())
//     for (int i = 0; i < trees.size(); ++i)
//     {
//         std::string speciesName = trees[i]->speciesName;
//         // auto configSpeciesName = configString("tree" + std::to_string(i), "species_name", "default");
//         // std::string speciesName = *configSpeciesName;
//         // int condition = speciesName.compare("default");
//         // if (!condition)
//         //     break;
//         osg::ref_ptr<osg::MatrixTransform> treeTransform(new osg::MatrixTransform);
//         trees[i]->sceneGraphNode = treeTransform;
//         treeTransform->setName("treeTransform"+std::to_string(i));
//         rootTransform->addChild(treeTransform);

//         // choose correct tree model and add it to the scenegraph
//         auto itr = std::find_if(treeModels.begin(), treeModels.end(), [&](const std::shared_ptr<TreeModel>& treeModel){
//             return treeModel->speciesName == speciesName;
//         });
//         if (itr == treeModels.end())
//         {
//             itr = defaultTreeIterator;
//         }
//         if (((*itr)->model) == nullptr)
//         {
//             itr = defaultTreeIterator;
//         }
//         /*if ((*itr)->season != stringToSeason(configSeason->value()))
//         {
//             model = defaultTreeModel;
//             rotation = osg::Vec3d(configDefaultRotation->value()[0], configDefaultRotation->value()[1], configDefaultRotation->value()[2]);
//             translate = osg::Vec3d(configDefaultTranslation->value()[0], configDefaultTranslation->value()[1], configDefaultTranslation->value()[2]);
//             // std::cout << "default tree model" << std::endl;
//         }
//         else
//         {
//             model = treeModels[index]->model;
//             rotation = treeModels[index]->rotation;
//             translate = treeModels[index].translation;
//             // std::cout << "index: " << index << "; Model: " << treeModels[index].speciesName << std::endl;
//         }*/
//         if((*itr)->model)
//         {
//             treeTransform->addChild((*itr)->model);
//             trees[i]->treeModel = *itr;

//             // read tree position and get altitude from geotiff
//             auto treeXPos = trees[i]->xPos;
//             auto treeYPos = trees[i]->yPos;
//             float treeAltitude = getAlt(treeXPos, treeYPos);
//             trees[i]->setAltitude(treeAltitude);

//             // set height for specific trees or get default height, if no height is specified and scale models to height
//             auto height = trees[i]->height;

//             double scaleFactor = height / (*itr)->height;
//             // trees[i]->setScale(scaleFactor);
//             trees[i]->scale = scaleFactor;

//             treeTransform->setMatrix(osg::Matrixd::scale(osg::Vec3d(scaleFactor, scaleFactor, scaleFactor)) * (*itr)->transform * osg::Matrix::translate(osg::Vec3d(treeXPos, treeYPos, treeAltitude)));
//             trees[i]->setTransform();
//         }

//         trees[i]->setTreeModelLODs();
//     }
//     closeImage();
// }

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

double UrbanTempo::getDistance(const osg::Vec3 vec1, const osg::Vec3 vec2)
{
    auto diff = vec2 - vec1;
    return sqrt(diff * diff);
}

void UrbanTempo::key(int type, int keysym, int mod)
{
    std::cout << "pressed key\n";
    this->updateTrees();

    // auto invBaseMat = cover->getInvBaseMat();
    // // auto baseMat = cover->getBaseMat();

    // auto viewerMat = cover->getViewerMat(); // -> Koordinatensystem in Cave Weltkoordinaten
    // auto posViewerWelt = viewerMat.getTrans();

    // auto mat1 = viewerMat * invBaseMat; // -> Scenekoordinaten
    // auto posViewerObjekt = mat1.getTrans();

    // osg::Group* firstChild = dynamic_cast<osg::Group*>(pluginNode->getChild(0));
    // auto numChildren = firstChild->getNumChildren();
    // for (int i = 0; i < numChildren; ++i)
    // {
    //     osg::Group* secondChild = dynamic_cast<osg::Group*>(firstChild->getChild(i));
    //     auto nodeName = secondChild->getName();
    //     auto nodeNameSec = secondChild->getChild(0)->getName();
    //     osg::MatrixTransform* matrixTransform = dynamic_cast<osg::MatrixTransform*>(secondChild);

    //     osg::Vec3 posTreeObjekt(0.0, 0.0, 0.0);
    //     // osg::Vec3 posTreeWelt(0.0, 0.0, 0.0);
    //     if (matrixTransform)
    //     {
    //         osg::Matrix matrix = matrixTransform->getMatrix();
    //         posTreeObjekt = matrix.getTrans(); // -> Objektkoordinaten
    //         // osg::Matrix matrix2 = matrix * baseMat;
    //         // posTreeWelt = matrix2.getTrans(); // -> Weltkoordinaten
    //     }

    //     // auto distanceWorld = getDistance(posViewerWelt, posTreeWelt);
    //     auto distanceObjekt = getDistance(posViewerObjekt, posTreeObjekt);

    //     std::cout << "nodeName: " << nodeName << "\n";
    //     std::cout << "nodeNameSec: " << nodeNameSec << "\n";
    //     // std::cout << "Abstand Welt: " << distanceWorld << "\n";
    //     std::cout << "Abstand Objekt: " << distanceObjekt << "\n";

    //     // std::string modelPath = "/home/kilian/projects/trees/Birch-1/Birch 2 N111212.3DS";
    //     // auto model = osgDB::readNodeFile(modelPath);
    //     // double height;
    //     // if (model)
    //     // {
    //     //     osg::ComputeBoundsVisitor cbv;
    //     //     model->accept(cbv);
    //     //     osg::BoundingBox bb = cbv.getBoundingBox();
    //     //     osg::Vec3 size(bb.xMax() - bb.xMin(), bb.yMax() - bb.yMin(), bb.zMax() - bb.zMin());
    //     //     // if((*configRotation)[0]!=0.0)
    //     //     //     height = size[1]; // hack, if y is up, use y as height, otherwise z
    //     //     // else
    //     //     //     height = size[2];
    //     //     height = size[2];
    //     // }

    //     double threshold = 75.0;

    //     if (distanceObjekt < threshold)
    //     {
            
    //         // secondChild->replaceChild(secondChild->getChild(0), model);

    //         // secondChild->getChild(0);

    //         // osg::MatrixTransform* transform = dynamic_cast<osg::MatrixTransform*>(secondChild);
    //         // osg::MatrixTransform* scaleOrig = dynamic_cast<osg::MatrixTransform*>(secondChild->getChild(1));

    //         // transform->setScale(osg::Vec3d(1.0, 1.0, 1.0));
    //         // // auto transformOrigMat = transform->getMatrix();
    //         // // auto scaleOrigMat = scaleOrig->getMatrix();

    //         // auto configDefaultHeight = configFloat("general", "default_height", 10);
    //         // double defaultHeight = *configDefaultHeight;

    //         // double scaleFactor = height / defaultHeight;

    //         // transform->setScale(osg::Matrixd::scale(osg::Vec3d(scaleFactor, scaleFactor, scaleFactor));

    //     }
    // }
    // // osg::Group* secondChild = dynamic_cast<osg::Group*>(firstChild->getChild(0));
    // // auto nodeName = secondChild->getName();
    // // osg::MatrixTransform* matrixTransform = dynamic_cast<osg::MatrixTransform*>(secondChild);

    // // osg::Vec3 posTreeObjekt(0.0, 0.0, 0.0);
    // // osg::Vec3 posTreeWelt(0.0, 0.0, 0.0);
    // // if (matrixTransform)
    // // {
    // //     osg::Matrix matrix = matrixTransform->getMatrix();
    // //     posTreeObjekt = matrix.getTrans(); // -> Objektkoordinaten
    // //     osg::Matrix matrix2 = matrix * baseMat;
    // //     posTreeWelt = matrix2.getTrans(); // -> Weltkoordinaten
    // // }

    // // auto distanceWorld = getDistance(posViewerWelt, posTreeWelt);
    // // auto distanceObjekt = getDistance(posViewerObjekt, posTreeObjekt);

    // // std::cout << "posViewerWelt: " << posViewerWelt << "\n";
    // // std::cout << "posViewerObjekt: " << posViewerObjekt << "\n";
    // // std::cout << nodeName << "_Welt: " << posTreeWelt << "\n";
    // // std::cout << nodeName << "_Objekt: " << posTreeObjekt << "\n";
    // // std::cout << "Abstand Welt: " << distanceWorld << "\n";
    // // std::cout << "Abstand Objekt: " << distanceObjekt << "\n";

    // // std::cout << "getInvBaseMat(): " << invBaseMat << "\n";
    // // std::cout << "getViewerMat(): " << viewerMat << "\n";
    // std::cout << "-------------------------------------" << "\n";  
}

void UrbanTempo::updateTrees()
{
    for (int i = 0; i < trees.size(); ++i)
    {
        trees[i]->updateTreeModel();
    }
}

void UrbanTempo::preFrame()
{
    this->updateTrees();
}

COVERPLUGIN(UrbanTempo)
