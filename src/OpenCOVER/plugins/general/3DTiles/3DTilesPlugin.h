/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Google3DTiles_PLUGIN_H
#define _Google3DTiles_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2023 HLRS  **
 **                                                                          **
 ** Description: Google3DTiles Plugin (reads Google3DTiles and GLB files)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** JULY-23  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

class PLUGINEXPORT ListenerCover;
class PLUGINEXPORT ViewerOsg;
class VrmlScene;
class PLUGINEXPORT SystemCover;
#include <osg/Group>
#include <osg/Matrix>
#include <osg/Material>

#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h> 
#include <rapidjson/rapidjson.h>
#include <iostream>

#include <proj_api.h>

#include <osgDB/FileNameUtils>
#include <osgDB/Registry>


class PLUGINEXPORT Google3DTilesPlugin : public coVRPlugin
{
    friend class ListenerCover;
    friend class SystemCover;
    friend class ViewerOsg;

public:
    static Google3DTilesPlugin *plugin;

    std::string GoogleAPIKey;
    bool request(const std::string &url, std::string &response);
    bool parseResponse(const std::string& response);
    bool parseChildren(const rapidjson::Value &children);
    // mutable Google3DTilesReader::TextureCache _cache;

    static size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata);

    Google3DTilesPlugin();
    virtual ~Google3DTilesPlugin();

    static int loadGoogle3DTiles(const char *filename, osg::Group *loadParent, const char *ck = "");
    static int unloadGoogle3DTiles(const char *filename, const char *ck = "");

    int loadFile(const std::string& fileName, osg::Group* parent);
    int unloadFile(const std::string& fileName);

    osgDB::ReaderWriter::ReadResult readNode(const std::string& location, const osgDB::Options* options);
    osgDB::ReaderWriter::ReadResult readNode(std::istream& inputStream, const osgDB::Options* options);

    // this will be called in PreFrame
    virtual void preFrame();
    virtual bool init();

    osg::ref_ptr<osg::MatrixTransform> Google3DTilesRoot;
    osg::Vec3 myPosition;
    std::list<std::string> jsonURIs;
    std::list<std::string> toProcessURIs;
    std::list<std::string> glbURIs;

private:
    projPJ pj_from, pj_to;
    CURL* curl=nullptr;
};

#endif
