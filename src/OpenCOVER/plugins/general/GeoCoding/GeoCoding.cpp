/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GeoCoding.h"

#include <cover/coVRPluginSupport.h>

#include <geodata/GeoData.h>
#include <string>
#include "HTTPClient/CURL/methods.h"
#include "HTTPClient/CURL/request.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/rapidjson.h>

using namespace opencover;

GeoCoding *GeoCoding::s_instance = nullptr;

GeoCoding::GeoCoding()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("GeoData", cover->ui)
{
    assert(s_instance == nullptr);
    s_instance = this;
}

bool GeoCoding::init()
{
    m_geoDataMenu = dynamic_cast<ui::Menu *>(cover->ui->getByPath("Manager.GeoData"));
    if (!m_geoDataMenu)
    {
        m_geoDataMenu = new ui::Menu("GeoData", cover->ui);
        m_geoDataMenu->setText("GeoData");
        m_geoDataMenu->allowRelayout(true);
    }
    m_geoDataMenu->setVisible(true);

    m_geoCodingGroup = new ui::Group(m_geoDataMenu, "GeoCoding");
    m_geoCodingGroup->setText("Geo coding");
    m_geoCodingGroup->allowRelayout(true);

    m_searchQueryField = new ui::EditField(m_geoCodingGroup, "SearchQuery");
    m_searchQueryField->setText("Search location");
    m_searchQueryField->setCallback([this](const std::string &val)
        { jumpToAddress(val); });

    m_currentLocationLabel = new ui::Label(m_geoCodingGroup, "CurrentLocation");
    m_currentLocationLabel->setText("...");

    m_actionGeocode = new ui::Button(m_geoCodingGroup, "ActionGeocode");
    m_actionGeocode->setText("Where am I?");
    m_actionGeocode->setCallback([this](bool _)
        {
            geocode();
            m_actionGeocode->setState(false); });

    return true;
}

// this is called if the plugin is removed at runtime
GeoCoding::~GeoCoding()
{
    s_instance = nullptr;

    delete m_geoCodingGroup;
    m_geoDataMenu->setVisible(m_geoDataMenu->numChildren() > 0);
}

GeoCoding *GeoCoding::instance()
{
    if (!s_instance)
        s_instance = new GeoCoding;
    return s_instance;
}

void GeoCoding::jumpToAddress(std::string_view searchQuery)
{
    using namespace opencover::httpclient::curl;
    std::string jsonData;

    CURL *curl = curl_easy_init();
    if (!curl)
    {
        std::cerr << "Failed to initialize CURL." << std::endl;
        return;
    }

    std::string address(searchQuery);
    char *encoded = curl_easy_escape(curl, address.c_str(), address.length());
    if (!encoded)
    {
        std::cerr << "Failed to encode address for query: " << searchQuery << std::endl;
        curl_easy_cleanup(curl);
        return;
    }

    // Nominatim request URL
    std::string url = "https://nominatim.openstreetmap.org/search?q=" + std::string(encoded) + "&format=geocodejson&limit=1&addressdetails=1&accept-language=en";
    curl_free(encoded);

    GET getRequest(url);
    Request::Options options = {
        { CURLOPT_USERAGENT, "Covise Plugin GeoCoding (https://github.com/hlrs-vis/covise)" },
    };

    if (!Request().httpRequest(getRequest, jsonData, options))
    {
        std::cerr << "Failed to fetch data from Nominatim. With request: " << url << std::endl;
        curl_easy_cleanup(curl);
        return;
    }

    curl_easy_cleanup(curl);

    rapidjson::Document document;
    if (document.Parse(jsonData.c_str()).HasParseError())
        return;

    if (!document.IsObject() || !document.HasMember("features") || !document["features"].IsArray() || document["features"].Empty())
        return;

    const rapidjson::Value &coords = document["features"][0]["geometry"]["coordinates"];
    if (!coords.IsArray() || coords.Size() < 2)
        return;

    double latitude = 0.0, longitude = 0.0;
    if (coords[0].IsNumber() && coords[1].IsNumber())
    {
        longitude = coords[0].GetDouble();
        latitude = coords[1].GetDouble();
    }
    else
    {
        try
        {
            if (coords[0].IsString())
                longitude = std::stod(coords[0].GetString());
            if (coords[1].IsString())
                latitude = std::stod(coords[1].GetString());
        }
        catch (const std::exception &)
        {
            std::cerr << "Failed to parse geo coordinates from geocoding result." << std::endl;
            return;
        }
    }

    auto projectLocation = GeoData::instance()->globalToProject(osg::Vec3(longitude, latitude, 500.0));
    GeoData::instance()->jumpToLocation(projectLocation, 100.0);

    const rapidjson::Value &locationProperties = document["features"][0]["properties"]["geocoding"];
    std::string text = formatAddressLabel(locationProperties);
    m_currentLocationLabel->setText(text);
}

#include <osg/io_utils>
void GeoCoding::geocode()
{
    auto projectLocation = GeoData::instance()->getProjectPosition();
    std::cout << "Local coords (relative to project origin): " << projectLocation << std::endl;
    auto global = GeoData::instance()->getGlobalPosition();
    std::cout << "Global coords (WGS84): " << global << std::endl;

    using namespace opencover::httpclient::curl;
    std::string jsonData;

    CURL *curl = curl_easy_init();
    if (!curl)
    {
        std::cerr << "Failed to initialize CURL." << std::endl;
        return;
    }

    std::string longitude = std::to_string(global.x());
    std::string latitude = std::to_string(global.y());

    std::string text = "Longitude: " + longitude + "; Latitude: " + latitude;
    m_currentLocationLabel->setText(text);

    // Nominatim request URL
    std::string url = "https://nominatim.openstreetmap.org/reverse?lon=" + longitude + "&lat=" + latitude + "&format=geocodejson&limit=1&addressdetails=1&accept-language=en";

    GET getRequest(url);
    Request::Options options = {
        { CURLOPT_USERAGENT, "Covise Plugin GeoCoding (https://github.com/hlrs-vis/covise)" },
    };

    if (!Request().httpRequest(getRequest, jsonData, options))
    {
        std::cerr << "Failed to fetch data from Nominatim. With request: " << url << std::endl;
        curl_easy_cleanup(curl);
        return;
    }

    curl_easy_cleanup(curl);

    rapidjson::Document document;
    if (document.Parse(jsonData.c_str()).HasParseError())
        return;

    if (!document.IsObject() || !document.HasMember("features") || !document["features"].IsArray() || document["features"].Empty())
        return;

    const rapidjson::Value &locationProperties = document["features"][0]["properties"]["geocoding"];
    std::string addressText = formatAddressLabel(locationProperties);
    m_currentLocationLabel->setText(text + "\n" + addressText);
    std::cout << "Geocoded: " << addressText << std::endl;
}

std::string GeoCoding::formatAddressLabel(const rapidjson::Value &locationProperties)
{
    std::string line1;
    std::string line2;
    std::string line3;
    std::string line4;

    if (locationProperties.HasMember("name"))
        line1 += locationProperties["name"].GetString();

    if (locationProperties.HasMember("street"))
        line2 += std::string(locationProperties["street"].GetString()) + " ";

    if (locationProperties.HasMember("housenumber"))
        line2 += locationProperties["housenumber"].GetString();

    if (locationProperties.HasMember("postcode"))
        line3 += std::string(locationProperties["postcode"].GetString()) + " ";

    if (locationProperties.HasMember("city"))
        line3 += locationProperties["city"].GetString();

    if (locationProperties.HasMember("county"))
    {
        if (!line3.empty())
            line3 += ", ";

        line3 += locationProperties["county"].GetString();
    }

    if (locationProperties.HasMember("state"))
    {
        if (!line4.empty())
            line4 += ", ";

        line4 += locationProperties["state"].GetString();
    }

    if (locationProperties.HasMember("country"))
    {
        if (!line4.empty())
            line4 += ", ";

        line4 += locationProperties["country"].GetString();
    }

    std::vector<std::string> lines;

    if (!line1.empty())
        lines.push_back(line1);

    if (!line2.empty())
        lines.push_back(line2);

    if (!line3.empty())
        lines.push_back(line3);

    if (!line4.empty())
        lines.push_back(line4);

    std::string text;

    for (size_t i = 0; i < lines.size(); ++i)
    {
        if (i > 0)
            text += "\n";

        text += lines[i];
    }

    return text;
}

COVERPLUGIN(GeoCoding)
