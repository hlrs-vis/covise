/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TREES_PLUGIN_H
#define _TREES_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2023 HLRS  **
 **                                                                          **
** Description: Trees OpenCOVER Plugin                                       **
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
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <osg/Node>
#include <osg/ref_ptr>
#include <gdal_priv.h>


class Trees : public opencover::coVRPlugin
{
public:
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

    float getAlt(double x, double y);
    void openImage(std::string &name);
    void closeImage();

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
};
#endif

