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


using namespace opencover;

bool Trees::init()
{
    url = "https://services3.arcgis.com/FwX2qF9JecNSRnwr/ArcGIS/rest/services/IndividualPlants_GT_Tallinn/FeatureServer/1/query?where=1%3D1&outFields=*&f=pjson";
    request();
    simplifyResponse();
    std::string season = "winter";
    addSeason(season);
    saveStringToFile(simpleResponse);
    setTrees();

    return true;
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
        const std::string section = "section";
        const std::string name = "name";
        const int number = 6;

        // configInt(section, name, 5);
        auto intConfig = configInt("section", "number", 2, config::Flag::PerModel);
        // auto intConfig = configInt("section", "number", 2);
        int intConfigValue = *intConfig;
        std::cout << intConfigValue << std::endl;
        // coVRPlugin::configInt(section, name, 5);

        // std::unique_ptr<config::Value<int>> test = config.value<int>(path, section, name, number);
        // auto test = config.value<int>(path, section, name, number);

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
    // parse json with information about trees
    rapidjson::Document document;
    document.Parse(simpleResponse.c_str());

    // get list with all species_name entries
    std::vector<std::string> speciesNames;
    for (rapidjson::Value::ValueIterator itr = document.Begin(); itr != document.End(); ++itr)
    {
        if((*itr)["species_name"].IsString()
            && (*itr)["x"].IsDouble()
            && (*itr)["y"].IsDouble()) 
        {
            std::string species_name = (*itr)["species_name"].GetString();
            speciesNames.push_back(species_name);
        }
    }
    
    // remove duplicates
    std::sort(speciesNames.begin(), speciesNames.end());
    auto end = std::unique(speciesNames.begin(), speciesNames.end());
    speciesNames.erase(end, speciesNames.end());

    // std::vector<osg::Node> treeModels;
    // for(auto itr = speciesNames.begin(); itr != speciesNames.end(); ++itr)
    // {
    //     // über species names iterieren und modelle laden
    //     // osg::ref_ptr<osg::Node> treeModel;
    //     // treeModels.push_back(*treeModel);
    // }

    // create transform node to translate placed trees
    osg::ref_ptr<osg::MatrixTransform> transform(new osg::MatrixTransform);
    cover->getObjectsRoot()->addChild(transform);
    osg::Matrix rootMatrix;
    auto configX = configFloat("offset", "x", 0.0);
    auto configY = configFloat("offset", "y", 0.0);
    rootMatrix.makeTranslate(osg::Vec3d(0.0, 0.0, 0.0));
    transform->setMatrix(rootMatrix);

    // add trees one after another with information from the json file
    for (rapidjson::Value::ValueIterator itr = document.Begin(); itr != document.End(); ++itr)
    {
        if((*itr)["species_name"].IsString()
            && (*itr)["x"].IsDouble()
            && (*itr)["y"].IsDouble()) 
        {
            osg::ref_ptr<osg::MatrixTransform> treeTransform(new osg::MatrixTransform);
            transform->addChild(treeTransform);

            std::string species_name = (*itr)["species_name"].GetString();
            std::cout << species_name << std::endl;
            const std::string path = "/home/kilian/Downloads/plants/uploads_files_1872992_plants.3ds";
            osg::ref_ptr<osg::Node> object = osgDB::readNodeFile(path);
            osg::ref_ptr<osg::Geode> geode = object->asGeode();
            treeTransform->addChild(object);

            // if((*itr)["height"].IsInt())
            // {
            //     // std::cout << (*itr)["attributes"]["height"].GetInt() << std::endl;
            //     value = (*itr)["attributes"]["height"].GetInt(); 
            //     entry.AddMember("height", value, allocator);
            // }
            // else
            // {
            //     // std::cout << "NULL" << std::endl;
            //     entry.AddMember("height", rapidjson::Value(rapidjson::kNullType), allocator);
            // }

            double x = (*itr)["x"].GetDouble();
            std::cout << x << std::endl;

            double y = (*itr)["y"].GetDouble();
            std::cout << y << std::endl;

            osg::Matrix matrix;
            matrix.makeTranslate(osg::Vec3d(x, y, 0.0));
            treeTransform->setMatrix(matrix);
        }
    }
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

COVERPLUGIN(Trees)