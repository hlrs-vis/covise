/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _URBANTEMPO_PLUGIN_H
#define _URBANTEMPO_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2023 HLRS  **
 **                                                                          **
 ** Description: Urban Tempo OpenCOVER Plugin                                **
 **                                                                          **
 **                                                                          **
 ** Author: Kilian Türk		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** Feb 2023  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <string>
#include <cover/coVRPlugin.h>
#include <cover/coVRFileManager.h>
#include <rapidjson/document.h>
#include <osg/Node>
#include <osg/ref_ptr>
#include <gdal_priv.h>

enum Season { 
    Winter,
    Spring,
    Summer,
    Fall
};


class  TreeModel
{
public:
    TreeModel(std::string configName);
    ~TreeModel();
    std::string speciesName;
    std::string modelPath;
    Season season;
    osg::ref_ptr<osg::Node> model;
    osg::Matrix transform;
    float height;
};

class UrbanTempo : public opencover::coVRPlugin
{
public:
    UrbanTempo();
    bool init() override;
    bool destroy() override;
    void request();
    void simplifyResponse();
    void saveStringToFile(const std::string&);
    std::string readJSONFromFile(const std::string&);
    void setTrees();
    std::string documentToString(const rapidjson::Document&);
    void printResponseToConfig();
    void setupPluginNode();
    void printInformation();

    float getAlt(double x, double y);
    static Season stringToSeason(const std::string&);

    std::vector<std::unique_ptr<TreeModel>> treeModels;
    std::vector<std::unique_ptr<TreeModel>>::iterator defaultTreeIterator;
    static UrbanTempo* instance() { return plugin; };

private:
    std::string url;
    std::string path;
    std::string response;
    std::string simpleResponse;
    osg::ref_ptr<osg::Group> pluginNode;

    float *rasterData=NULL;
    double xOrigin;
    double yOrigin;
    double pixelWidth;
    double pixelHeight;
    int cols;
    int rows;
    GDALDataset  *heightDataset;
    GDALRasterBand  *heightBand;
    void openImage(std::string& name);
    void closeImage();
    static UrbanTempo* plugin;
};
#endif

