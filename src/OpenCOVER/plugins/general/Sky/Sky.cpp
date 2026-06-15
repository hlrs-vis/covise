/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2024 HLRS  **
 **                                                                          **
 ** Description: Sky Plugin                                                  **
 **                                                                          **
 **                                                                          **
 ** Author: Uwe Woessner 		                                             **
 **                                                                          **
 ** History:  								                                 **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Sky.h"

#include <exiv2/exiv2.hpp>

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <cover/ui/Manager.h>

#include <config/CoviseConfig.h>
#include <geodata/GeoData.h>

#include <osg/Math>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osg/Node>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/CullFace>
#include <osg/Texture2D>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/LOD>
#include <osg/ShapeDrawable>
#include <osgViewer/Renderer>
#include <iostream>
#include <regex>
#include <string>
#include "cover/coVRConfig.h"
#include <PluginUtil/PluginMessageTypes.h>
#include <filesystem>

#include <vrml97/vrml/vrmlexport.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeChild.h>

#include "VrmlNodeSky.h"

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
using namespace vrml;

Sky *Sky::s_instance = nullptr;

Sky::Sky()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("Sky", cover->ui)
{
    assert(s_instance == nullptr);
    s_instance = this;
}
bool Sky::init()
{
    VrmlNamespace::addBuiltIn(VrmlNode::defineType<VrmlNodeSky>());

    geoDataMenu = dynamic_cast<ui::Menu *>(cover->ui->getByPath("Manager.GeoData"));
    if (!geoDataMenu)
    {
        geoDataMenu = new ui::Menu("GeoData", cover->ui);
        geoDataMenu->setText("GeoData");
        geoDataMenu->allowRelayout(true);
    }
    geoDataMenu->setVisible(true);

    skyRootNode = new osg::MatrixTransform();
    skyRootNode->setName("sky");
    cover->getScene()->addChild(skyRootNode);

    auto configFile = config();

    texturedSphere = new osg::Geode;
    osg::Sphere *_Sphere = new osg::Sphere();
    _Sphere->setRadius(1.0);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(1.0);
    hint->setCreateBackFace(true);
    hint->setCreateFrontFace(false);
    hint->setCreateTextureCoords(true);
    // hint->setCreateNormals(true);
    osg::ShapeDrawable *_sphereDrawable = new osg::ShapeDrawable(_Sphere, hint);
    _sphereDrawable->setColor(osg::Vec4(1, 1, 1, 1));
    _sphereDrawable->setUseDisplayList(false); // turn off display list so that we can change the pointer length
    texturedSphere->addDrawable(_sphereDrawable);
    osg::StateSet *stateset = texturedSphere->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::Material *spheremtl = new osg::Material;
    spheremtl->setColorMode(osg::Material::OFF);
    spheremtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1));
    spheremtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1));
    spheremtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    spheremtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
    spheremtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    stateset->setAttributeAndModes(spheremtl, osg::StateAttribute::ON);

    osg::CullFace *cF = new osg::CullFace();
    cF->setMode(osg::CullFace::BACK);
    stateset->setAttributeAndModes(cF, osg::StateAttribute::OFF);
    texturedSphere->setStateSet(stateset);

    skyGroup = new ui::Group(geoDataMenu, "sky");
    skyGroup->setText("Sky");
    skyGroup->allowRelayout(true);

    skyList = new ui::SelectionList(skyGroup, "Sky");
    skyList->setCallback([this](int selection)
        {
            if (selection <= 0)
            {
                setSkyDisabled();
                return;
            }
            int index = selection - 1;
            if (index >= 0 && index < (int)m_skies.size())
            {
                setSkyEntry(m_skies[index], true);
            } });

    skyPath = configString("sky", "skyDir", "/data/Geodata/sky")->value();
    loadSkies();

    autoSkyButton = new ui::Button(skyGroup, "autoSky");
    autoSkyButton->setText("Auto Sky");
    autoSkyButton->setState(false);
    autoSkyButton->setCallback([this](bool state)
        {
            m_autoSkyEnabled = state;
            if (state)
                updateAutoSky(); });

    ephemeralSkyButton = new ui::Button(skyGroup, "ephemeralSky");
    ephemeralSkyButton->setText("Ephemeral Sky");
    ephemeralSkyButton->setState(false);
    ephemeralSkyButton->setCallback([this](bool state)
        { setSkyEphemeris(state); });

    jumpToSkyButton = new ui::Button(skyGroup, "jumpToSky");
    jumpToSkyButton->setText("Jump to Sky");
    jumpToSkyButton->setCallback([this](bool state)
        {
            if (!state)
                return;
            int selection = skyList->selectedIndex();
            int index = selection - 1;

            if (index >= 0 && index < (int)m_skies.size())
            {
                const auto &sky = m_skies[index];
                osg::Vec3d globalSkyOrigin(sky.longitude, sky.latitude, sky.altitude);
                osg::Vec3 projectSkyOrigin = GeoData::instance()->globalToProject(globalSkyOrigin);
                GeoData::instance()->jumpToLocation(projectSkyOrigin);
            }

            jumpToSkyButton->setState(false); });

    northAngle = 0;
    update();

    skyNorthSlider = new ui::Slider(skyGroup, "skyNorth");
    skyNorthSlider->setText("Sky True North (°)");
    skyNorthSlider->setBounds(-180.0, 180.0);
    skyNorthSlider->setValue(northAngle);
    skyNorthSlider->setCallback([this](double value, bool released)
        { northAngle = value; update(); });

    auto defaultSky = configString("sky", "defaultSky", "")->value();
    if (!defaultSky.empty())
    {
        setSky(defaultSky);
    }

    return true;
}

// this is called if the plugin is removed at runtime
Sky::~Sky()
{
    s_instance = nullptr;
    delete skyGroup;
    while (texturedSphere->getNumParents())
        texturedSphere->getParent(0)->removeChild(texturedSphere);
    while (skyRootNode->getNumParents())
        skyRootNode->getParent(0)->removeChild(skyRootNode);
    geoDataMenu->setVisible(geoDataMenu->numChildren() > 0);
}

void Sky::loadSkies()
{
    std::function<void(const std::filesystem::path &)> read_dir;
    read_dir = [&read_dir, this](const std::filesystem::path &path) -> void
    {
        try
        {
            for (const auto &entry : std::filesystem::directory_iterator(path))
            {
                if (entry.is_directory())
                {
                    read_dir(entry.path());
                }
                else if (entry.is_regular_file())
                {
                    const auto &path = entry.path();
                    addSkyFile(path);
                }
            }
        }
        catch (const std::filesystem::filesystem_error &err)
        {
            std::cerr << "Filesystem error: " << err.what() << std::endl;
        }
        catch (const std::exception &ex)
        {
            std::cerr << "General error: " << ex.what() << std::endl;
        }
    };

    read_dir(skyPath);
    updateSkyMenu();
}

void parseExifData(const std::filesystem::path &path, double &longitude, double &latitude, double &altitude, double &trueNorth)
{
    std::unique_ptr<Exiv2::Image> image = Exiv2::ImageFactory::open(path.string().c_str());
    if (!image)
        return;

    image->readMetadata();
    auto &exif = image->exifData();
    if (exif.empty())
        return;

    auto latIt = exif.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLatitude"));
    if (latIt != exif.end())
    {
        auto deg = exif["Exif.GPSInfo.GPSLatitude"].toRational(0);
        auto min = exif["Exif.GPSInfo.GPSLatitude"].toRational(1);
        auto sec = exif["Exif.GPSInfo.GPSLatitude"].toRational(2);
        latitude = deg.first / (1.0 * deg.second) + min.first / (60.0 * min.second) + sec.first / (3600.0 * sec.second);
    }

    auto lonIt = exif.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSLongitude"));
    if (lonIt != exif.end())
    {
        auto deg = exif["Exif.GPSInfo.GPSLongitude"].toRational(0);
        auto min = exif["Exif.GPSInfo.GPSLongitude"].toRational(1);
        auto sec = exif["Exif.GPSInfo.GPSLongitude"].toRational(2);
        longitude = deg.first / (1.0 * deg.second) + min.first / (60.0 * min.second) + sec.first / (3600.0 * sec.second);
    }

    auto altIt = exif.findKey(Exiv2::ExifKey("Exif.GPSInfo.GPSAltitude"));
    if (altIt != exif.end())
    {
        altitude = exif["Exif.GPSInfo.GPSAltitude"].toFloat();
    }

    Exiv2::XmpData &xmpData = image->xmpData();
    if (!xmpData.empty())
    {
        auto gimbalYawIt = xmpData.findKey(Exiv2::XmpKey("Xmp.drone-dji.GimbalYawDegree"));
        auto flightYawIt = xmpData.findKey(Exiv2::XmpKey("Xmp.drone-dji.FlightYawDegree"));

        if (gimbalYawIt != xmpData.end())
            trueNorth = xmpData["Xmp.drone-dji.GimbalYawDegree"].toFloat() + 90;
        if (flightYawIt != xmpData.end())
            trueNorth = xmpData["Xmp.drone-dji.FlightYawDegree"].toFloat() + 90;
    }
}

std::optional<std::reference_wrapper<SkyEntry>> Sky::addSkyFile(std::filesystem::path path)
{
    std::string fileName = path.filename().string();
    std::string extension = path.extension().string();

    auto relative = std::filesystem::relative(path, std::filesystem::path(skyPath)).string();
    std::string name = relative.substr(0, relative.length() - extension.length());

    // Transform extension to lower case
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char c)
        { return std::tolower(c); });

    if (extension != ".jpg")
    {
        std::cerr << "Cannot load sky file at " << path << "; only JPG is supported now." << std::endl;
        return std::nullopt;
    }

    double latitude = 0, longitude = 0, altitude = 0, trueNorth = 0;
    parseExifData(path, longitude, latitude, altitude, trueNorth);

    trueNorth = std::fmod(trueNorth + 360.0 + 180.0, 360.0) - 180.0;
    longitude = std::fmod(longitude + 360.0 + 180.0, 360.0) - 180.0;
    latitude = std::clamp(latitude, -90.0, 90.0);

    std::cout << "EXIF: Parsed " << path.parent_path().filename().string() << "/" << fileName << " latitude=" << latitude << " longitude=" << longitude << " altitude=" << altitude << " trueNorth=" << trueNorth << std::endl;

    return m_skies.emplace_back(SkyEntry(name, std::regex_replace(name, std::regex("/"), " → "), path.string(), longitude, latitude, altitude, trueNorth));

    return std::nullopt;
}

void Sky::updateSkyMenu()
{
    // Sort m_skies by name
    std::sort(m_skies.begin(), m_skies.end(),
        [&](const auto &a, const auto &b)
        {
            return a.name < b.name;
        });

    std::vector<std::string> skyNames = {
        "None"
    };
    for (const auto &sky : m_skies)
    {
        skyNames.push_back(sky.displayName);
    }
    skyList->setList(skyNames);
}

void Sky::message(int toWhom, int type, int length, const void *data)
{
    if (type == PluginMessageTypes::setSky)
    {
        setSky((const char *)data);
    }
}

void Sky::removeExistingSky()
{
    m_currentSky = nullptr;
    // Remove all existing sky nodes
    while (skyRootNode->getNumChildren())
        skyRootNode->removeChild(skyRootNode->getChild(0));

#ifdef HAVE_EPHEMERIS
    if (m_ephemeralSky)
        m_ephemeralSky = nullptr;
#endif
}

void Sky::setSky(std::string_view nameOrFile)
{
    if (nameOrFile == "None")
    {
        setSkyDisabled();
    }
    else if (nameOrFile == "Ephemeris")
    {
        setSkyEphemeris();
    }
    else if (nameOrFile == "Auto")
    {
        setSkyAuto();
    }
    else
    {
        setSkyTexture(nameOrFile);
    }
}

void Sky::setSkyDisabled()
{
    m_autoSkyEnabled = false;
    m_ephemeralSkyEnabled = false;
    m_currentSky = nullptr;

    if (autoSkyButton)
        autoSkyButton->setState(false);

    if (ephemeralSkyButton)
        ephemeralSkyButton->setState(false);

    removeExistingSky();
}

void Sky::setSkyEphemeris(bool enable)
{
    if (enable)
    {
#ifdef HAVE_EPHEMERIS
        removeExistingSky();
        m_ephemeralSkyEnabled = true;
        m_autoSkyEnabled = false;

        if (autoSkyButton)
            autoSkyButton->setState(false);
        if (ephemeralSkyButton)
            ephemeralSkyButton->setState(enable);
        if (!m_ephemeralSky)
        {
            m_ephemeralSky = std::make_unique<EphemeralSky>(skyGroup, skyRootNode);
        }
#else
        std::cerr << "Cannot switch to Ephemeris sky, compiled without osgEphemeris. Disabling sky." << std::endl;
        setSkyDisabled();
#endif
    }
    else
    {
        m_ephemeralSkyEnabled = false;
        m_autoSkyEnabled = false;
        if (ephemeralSkyButton)
            ephemeralSkyButton->setState(false);
        if (autoSkyButton)
            autoSkyButton->setState(false);
        removeExistingSky();
    }
}

void Sky::setHour(int h)
{

#ifdef HAVE_EPHEMERIS
    long t = cover->frameTime();
    m_ephemeralSky->setTime((t - t % 86400) + (h * (60 * 60)));
#endif
}
void Sky::setSkyAuto()
{
    m_autoSkyEnabled = true;
    if (autoSkyButton)
        autoSkyButton->setState(true);
    updateAutoSky();
}

void Sky::setSkyTexture(std::string_view nameOrFile)
{
    removeExistingSky();

    for (auto &it : m_skies)
    {
        if (it.fileName == nameOrFile || it.name == nameOrFile || it.displayName == nameOrFile) // already have this file in the list
        {
            setSkyEntry(it, true);
            return;
        }
    }

    // If we did not find a sky with this name (or filename), try to add it
    auto addedSky = addSkyFile(nameOrFile);
    if (addedSky)
    {
        updateSkyMenu();
        setSkyEntry(addedSky->get(), true);
    }
}

void Sky::setSkyEntry(SkyEntry &sky, bool manualSelection)
{
    removeExistingSky();
    m_currentSky = &sky;

    if (manualSelection)
    {
        m_autoSkyEnabled = false;
        if (autoSkyButton)
            autoSkyButton->setState(false);
    }

    // Choose the corresponding item in the menu
    const auto &l = skyList->items();
    skyList->select(std::find_if(l.begin(), l.end(), [sky](const auto &it)
                        { return it == sky.name || it == sky.displayName; })
        - l.begin());

    if (sky.texture == nullptr)
    {
        sky.texture = coVRFileManager::instance()->loadTexture(sky.fileName.c_str());
        sky.texture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
        sky.texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        sky.texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
        sky.texture->setResizeNonPowerOfTwoHint(false);
    }
    osg::StateSet *stateset = texturedSphere->getOrCreateStateSet();
    stateset->setTextureAttributeAndModes(0, sky.texture, osg::StateAttribute::ON);

    shader = coVRShaderList::instance()->get("skySphere");
    shader->apply(stateset);
    topUniform = shader->getcoVRUniform("top");
    bottomUniform = shader->getcoVRUniform("bottom");
    floorColorUniform = shader->getcoVRUniform("floorColor");

    skyRootNode->addChild(texturedSphere);

    setTrueNorth(sky.trueNorth);
}

void Sky::setTop(float t)
{
    if (topUniform != nullptr)
    {
        topUniform->setValue(t);
    }
}

void Sky::setBottom(float b)
{
    if (bottomUniform != nullptr)
    {
        bottomUniform->setValue(b);
    }
}

void Sky::setFloorColor(osg::Vec4 fc)
{
    if (floorColorUniform != nullptr)
    {
        floorColorUniform->setValue(fc);
    }
}

void Sky::setTrueNorth(float trueNorth)
{
    northAngle = trueNorth;
    if (skyNorthSlider)
        skyNorthSlider->setValue(northAngle);
}

void Sky::updateAutoSky()
{
    if (!m_autoSkyEnabled)
        return;

    auto globalPosition = GeoData::instance()->getGlobalPosition();
    SkyEntry *closestSky = findClosestSky(globalPosition);
    if (!closestSky)
        return;

    if (!m_currentSky)
    {
        setSkyEntry(*closestSky, false);
        return;
    }

    double lon = globalPosition.x();
    double lat = globalPosition.y();

    const auto dist2 = [](const SkyEntry *s, double lon, double lat)
    {
        const double dx = s->longitude - lon;
        const double dy = s->latitude - lat;
        return dx * dx + dy * dy;
    };

    const double currentDistSq = dist2(m_currentSky, lon, lat);
    const double closestDistSq = dist2(closestSky, lon, lat);

    constexpr double SKY_THRESHOLD = 1e-4; // roughly 1 km

    if (closestSky != m_currentSky && ((currentDistSq - closestDistSq) > SKY_THRESHOLD))
    {
        setSkyEntry(*closestSky, false);
    }
}

SkyEntry *Sky::findClosestSky(const osg::Vec3 &globalPosition)
{
    SkyEntry *closestSky = nullptr;
    double closestDistance = std::numeric_limits<double>::max();

    double longitude = globalPosition.x();
    double latitude = globalPosition.y();

    for (auto &sky : m_skies)
    {
        double dlon = sky.longitude - longitude;
        double dlat = sky.latitude - latitude;
        double distance = std::sqrt(dlon * dlon + dlat * dlat);
        if (distance < closestDistance)
        {
            closestDistance = distance;
            closestSky = &sky;
        }
    }
    return closestSky;
}

bool Sky::update()
{
    float nearValue = coVRConfig::instance()->nearClip();
    float farValue = coVRConfig::instance()->farClip();
    float scale = nearValue + (farValue - nearValue) * 0.5; // scale the sphere just between the clipping distances, depth is set by shader

    const osg::Matrix &m = cover->getObjectsXform()->getMatrix();

    double projectTrueNorth = GeoData::instance()->projectTrueNorthDegree();
    skyRootNode->setMatrix(
        osg::Matrix::scale(scale, scale, scale) * osg::Matrix::rotate(osg::DegreesToRadians(-northAngle - projectTrueNorth), osg::Vec3(0, 0, 1)) * osg::Matrix::rotate(m.getRotate())
        * osg::Matrix::translate(cover->getViewerMat().getTrans()));

    if (m_autoSkyEnabled)
    {
        updateAutoSky();
    }
    if (m_ephemeralSkyEnabled)
    {
#ifdef HAVE_EPHEMERIS
        if (m_ephemeralSky)
            m_ephemeralSky->update();
#endif
    }

    return false; // don't request that scene be re-rendered
}

Sky *Sky::instance()
{
    if (!s_instance)
        s_instance = new Sky;
    return s_instance;
}

COVERPLUGIN(Sky)
