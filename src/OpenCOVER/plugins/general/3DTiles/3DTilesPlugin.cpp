/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2021 HLRS  **
**                                                                          **
** Description: Google3DTiles Plugin (read Google3DTiles files)                                 **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Sep-21  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <PluginUtil/coLOD.h>
#include <config/CoviseConfig.h>
#include <cover/coVRShader.h>
#include <codecvt>
#include <string>


#include <cover/RenderObject.h>
#include <cover/VRRegisterSceneGraph.h>
#include <osg/LOD>

#include <osg/GL>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/TexMat>
#include <osg/TexEnvCombine>
#include <osg/Texture>
#include <osg/TextureCubeMap>
#include <osg/Texture2D>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/BlendFunc>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osg/Fog>
#include <osg/AlphaFunc>
#include <osg/ColorMask>
#include <osgDB/ReadFile>

#include <osgUtil/SmoothingVisitor>

#include <osgUtil/Optimizer>
#include "3DTilesPlugin.h"

#ifdef HAVE_OSGNV
#include <osgNVExt/RegisterCombiners>
#include <osgNVExt/CombinerInput>
#include <osgNVExt/CombinerOutput>
#include <osgNVExt/FinalCombinerInput>
#endif

#include <osgText/Font>
#include <osgText/Text>


#include <sys/types.h>
#include <string.h>

#include <boost/locale.hpp>

// Define radii for WGS84 ellipsoid
const osg::Vec3d _radii = osg::Vec3d(6378137.0, 6378137.0, 6356752.3142451793);
const osg::Vec3d _radiiSquared = osg::Vec3d(_radii[0] * _radii[0], _radii[1] * _radii[1], _radii[2] * _radii[2]);

// Function to convert from geographic coordinates to Cartesian coordinates
osg::Vec3d fromGeographic(double lat, double lon, double height) {
    lat = osg::DegreesToRadians(lat);
    lon = osg::DegreesToRadians(lon);

    double cosLatitude = cos(lat);

    osg::Vec3d scratchN(
        cosLatitude * cos(lon),
        cosLatitude * sin(lon),
        sin(lat)
    );
    scratchN.normalize();

    osg::Vec3d scratchK(
        _radiiSquared[0] * scratchN.x(),
        _radiiSquared[1] * scratchN.y(),
        _radiiSquared[2] * scratchN.z()
    );

    double gamma = sqrt(scratchN * scratchK);
    scratchK /= gamma;
    scratchN *= height;

    return scratchK + scratchN;
}


Google3DTilesPlugin *Google3DTilesPlugin::plugin = NULL;

Google3DTilesPlugin::Google3DTilesPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
   
    GoogleAPIKey = coCoviseConfig::getEntry("value","COVER.Plugin.GTiles.APIKey", "");
    Google3DTilesRoot = new osg::MatrixTransform();
    Google3DTilesRoot->setName("GoogleRoot");
	cover->getObjectsRoot()->addChild(Google3DTilesRoot);

    if (!(pj_from = pj_init_plus("+proj=latlong +datum=WGS84")))
    {
        std::cerr << "pj_init_plus(\"+proj=latlong +datum=WGS84\")  failed" << std::endl;
    }
    //epsg:3857
    if (!(pj_to = pj_init_plus("+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs +type=crs")))
    {
        std::cerr << "pj_init_plus(\" epsg:3857\")  failed" << std::endl;
    }
    
    //double lat = 48.19947;
    //double lon = 16.36757;
    double lat = 48.739634;
    double lon = 9.096791;
    double height = 100;

    double x = lon, y = lat,  z = 0.0;

    x *= DEG_TO_RAD;
    y *= DEG_TO_RAD;

    pj_transform(pj_from, pj_to, 1, 1, &x, &y, &z);
    myPosition.set(x,y,z);
    myPosition = fromGeographic(lat, lon, height);

    curl = curl_easy_init();
    if (!curl)
    {
        std::cerr << "Failed to create cURL handle" << std::endl;
    }
    plugin = this;
}

size_t Google3DTilesPlugin::writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    std::string* response = static_cast<std::string*>(userdata);
    response->append(ptr, size * nmemb);
    return size * nmemb;
}

bool Google3DTilesPlugin::request(const std::string &url, std::string &response)
{
    if(curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "GET");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            std::cerr << "Failed to execute HTTP request: " << curl_easy_strerror(res) << std::endl;
            return false;
        }
    }
    else
    {
        return false;
    }
    return true;

}
inline bool ends_with(std::string const& value, std::string const& ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

bool Google3DTilesPlugin::parseChildren(const rapidjson::Value& children)
{
    for (auto& child : children.GetArray())
    {
        if (child.HasMember("boundingVolume") && child["boundingVolume"]["box"].IsArray())
        {
            auto &bbox = child["boundingVolume"]["box"];
            osg::Vec3 bboxCenter(bbox[0].GetDouble(), bbox[1].GetDouble(), bbox[2].GetDouble());

            // direction and half-length for the local X-axis of <bbox>
            osg::Vec3 bboxX(bbox[3].GetDouble(), bbox[4].GetDouble(), bbox[5].GetDouble());

            // direction and half-length for the local Y-axis of <bbox>
            osg::Vec3 bboxY(bbox[6].GetDouble(), bbox[7].GetDouble(), bbox[8].GetDouble());

            // direction and half-length for the local Z-axis of <bbox>
            osg::Vec3 bboxZ(bbox[9].GetDouble(), bbox[10].GetDouble(), bbox[11].GetDouble());

            osg::Vec3 p0 = bboxCenter - bboxX - bboxY - bboxZ;
            osg::Vec3 p1 = bboxCenter + bboxX - bboxY - bboxZ;
            osg::Vec3 p2 = bboxCenter + bboxX + bboxY - bboxZ;
            osg::Vec3 p3 = bboxCenter - bboxX + bboxY - bboxZ;
            osg::Vec3 p4 = bboxCenter - bboxX - bboxY + bboxZ;
            osg::Vec3 p5 = bboxCenter + bboxX - bboxY + bboxZ;
            osg::Vec3 p6 = bboxCenter + bboxX + bboxY + bboxZ;
            osg::Vec3 p7 = bboxCenter - bboxX + bboxY + bboxZ;
            
            osg::BoundingBox osgBBox;
            osgBBox.expandBy(p0);
            osgBBox.expandBy(p1);
            osgBBox.expandBy(p2);
            osgBBox.expandBy(p3);
            osgBBox.expandBy(p4);
            osgBBox.expandBy(p5);
            osgBBox.expandBy(p6);
            osgBBox.expandBy(p7);
            if (!osgBBox.contains(myPosition))
                continue;
        }
        if (child.HasMember("content") && child["content"]["uri"].IsString())
        {
            std::string uri = child["content"]["uri"].GetString();
            if (uri.find(".json") != std::string::npos)
            {
                jsonURIs.push_back(uri);
            }
            else if (uri.find(".glb") != std::string::npos)
            {
                glbURIs.push_back(uri);
            }
            
            std::cerr << "content: " << child["content"]["uri"].GetString() << std::endl;
        }
        if (child.HasMember("children") && child["children"].IsArray())
        {
            return parseChildren(child["children"]);
        }
    }
    return true;
}

bool Google3DTilesPlugin::parseResponse(const std::string & response)
{
    rapidjson::Document document;
    document.Parse(response.c_str());

    if (document.HasMember("root"))
    {
        return parseChildren(document["root"]["children"]);
    }
    else
    {
        return false;
    }


    return true;
}

bool Google3DTilesPlugin::init()
{
    if (curl == nullptr)
        return false;
    std::string response;
    jsonURIs.clear();
    glbURIs.clear();
    int level = 0;
    if (request(std::string("https://tile.googleapis.com/v1/3dtiles/root.json?key=") + GoogleAPIKey, response))
    {
        bool res =  parseResponse(response);
        if (res == false)
            return false;
    }
    else
    {
        return false;
    }
    // extract session key
    std::string firstURI = *jsonURIs.begin();
    std::string sessionKey;
    if (firstURI.find(".json") != std::string::npos)
        sessionKey = firstURI.substr(firstURI.find(".json") + 5);
    while (level < 4)
    {
        toProcessURIs = jsonURIs;
        jsonURIs.clear();
        glbURIs.clear();
        for (const auto& uri : toProcessURIs)
        {
            response = "";
            std::string URL;
            if (level == 0)
            {
                URL = "https://tile.googleapis.com/" + std::string(uri) + "&key=" + GoogleAPIKey;
            }
            else
            {
                URL = "https://tile.googleapis.com/" + std::string(uri) + sessionKey + "&key=" + GoogleAPIKey;
            }
            if (request(URL, response))
            {
                parseResponse(response);
            }
        }
        level++;
    }

    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
Google3DTilesPlugin::~Google3DTilesPlugin()
{
    curl_easy_cleanup(curl);
	cover->getObjectsRoot()->removeChild(Google3DTilesRoot);
}

void
Google3DTilesPlugin::preFrame()
{
}

COVERPLUGIN(Google3DTilesPlugin)
