/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2024 HLRS  **
 **                                                                          **
 ** Description: SkySphere Plugin                                        **
 **                                                                          **
 **                                                                          **
 ** Author: Uwe Woessner 		                                              **
 **                                                                          **
 ** History:  								                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "SkySphere.h"

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
#include <string>
#include "cover/coVRConfig.h"
#include <PluginUtil/PluginMessageTypes.h>
#include <filesystem>

#include "VrmlNodeSkySphere.h"

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

SkySphere *SkySphere::s_instance = nullptr;

SkySphere::SkySphere()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("SkySphere", cover->ui)
{
    assert(s_instance == nullptr);
    s_instance = this;
}
bool SkySphere::init()
{
    // vrml::VrmlNamespace::addBuiltIn(vrml::VrmlNode::defineType<VrmlNodeSkySphere>());

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
        { setSky(skyList->items()[selection]); });

    skyPath = configString("sky", "skyDir", "/data/Geodata/sky")->value();
    loadSkies();

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
SkySphere::~SkySphere()
{
    s_instance = nullptr;
    delete skyGroup;
    while (texturedSphere->getNumParents())
        texturedSphere->getParent(0)->removeChild(texturedSphere);
    while (skyRootNode->getNumParents())
        skyRootNode->getParent(0)->removeChild(skyRootNode);
    geoDataMenu->setVisible(geoDataMenu->numChildren() > 0);
}

void SkySphere::loadSkies()
{
    try
    {
        for (const auto &entry : std::filesystem::directory_iterator(skyPath))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            const auto &path = entry.path();
            addSkyFile(path);
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

    updateSkyMenu();
}

void parseExifData(const std::filesystem::path &path, double &longitude, double &latitude, double &trueNorth)
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

std::optional<std::reference_wrapper<SkyEntry>> SkySphere::addSkyFile(std::filesystem::path path)
{
    std::string fileName = path.filename().string();
    std::string name = path.stem().string();
    std::string extension = path.extension();

    // Transform extension to lower case
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char c)
        { return std::tolower(c); });

    if (extension != ".jpg")
    {
        std::cerr << "Cannot load sky file at " << path << "; only JPG is supported now." << std::endl;
        return std::nullopt;
    }

    double latitude = 0, longitude = 0, trueNorth = 0;
    parseExifData(path, longitude, latitude, trueNorth);

    trueNorth = std::fmod(trueNorth + 360.0 + 180.0, 360.0) - 180.0;
    longitude = std::fmod(longitude + 360.0 + 180.0, 360.0) - 180.0;
    latitude = std::clamp(latitude, -90.0, 90.0);

    std::cout << "Parsed from EXIF: latitude=" << latitude << " longitude=" << longitude << " trueNorth=" << trueNorth << std::endl;

    return m_skies.emplace_back(SkyEntry {
        .name = name,
        .fileName = path.string(),
        .longitude = longitude,
        .latitude = latitude,
        .trueNorth = trueNorth,
    });

    return std::nullopt;
}

void SkySphere::updateSkyMenu()
{
    // Sort m_skies by name
    std::sort(m_skies.begin(), m_skies.end(),
        [&](const auto &a, const auto &b)
        {
            return a.name < b.name;
        });

    std::vector<std::string> skyNames = {
        "None",
        "Auto",
#ifdef HAVE_EPHEMERIS
        "Ephemeris",
#endif
    };
    for (const auto &sky : m_skies)
    {
        skyNames.push_back(sky.name);
    }
    skyList->setList(skyNames);
}

void SkySphere::message(int toWhom, int type, int length, const void *data)
{
    if (type == PluginMessageTypes::setSky)
    {
        setSky((const char *)data);
    }
}

void SkySphere::removeExistingSky()
{
    // Remove all existing sky nodes
    while (skyRootNode->getNumChildren())
        skyRootNode->removeChild(skyRootNode->getChild(0));

#ifdef HAVE_EPHEMERIS
    if (m_ephemeralSky)
        m_ephemeralSky = nullptr;
#endif
}

void SkySphere::setSky(std::string_view nameOrFile)
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

void SkySphere::setSkyDisabled()
{
    m_mode = DISABLED;
    removeExistingSky();
}

void SkySphere::setSkyEphemeris()
{
#ifdef HAVE_EPHEMERIS
    removeExistingSky();
    m_mode = EPHEMERIS;
    m_ephemeralSky = std::make_unique<EphemeralSky>(skyGroup, skyRootNode);
#else
    std::cerr << "Cannot switch to Ephemeris sky, compiled without osgEphemeris. Disabling sky." << std::endl;
    setSkyDisabled();
#endif
}

void SkySphere::setSkyAuto()
{
    m_mode = AUTO;
    // TODO
}

void SkySphere::setSkyTexture(std::string_view nameOrFile)
{
    removeExistingSky();

    SkyEntry *sky_ptr = nullptr;

    for (auto &it : m_skies)
    {
        if (it.fileName == nameOrFile || it.name == nameOrFile) // already have this file in the list
        {
            sky_ptr = &it;
        }
    }

    // If we did not find a sky with this name (or filename), try to add it
    if (!sky_ptr)
    {
        auto addedSky = addSkyFile(nameOrFile);
        if (addedSky)
        {
            updateSkyMenu();
            sky_ptr = &(addedSky->get());
        }
    }

    // Still no sky (adding was unsuccessful) -- do nothing.
    if (!sky_ptr)
        return;

    SkyEntry &sky = *sky_ptr;

    // Choose the corresponding item in the menu
    auto l = skyList->items();
    skyList->select(std::find_if(l.begin(), l.end(), [sky](const auto &it)
                        { return it == sky.name; })
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

void SkySphere::setTop(float t)
{
    if (topUniform != nullptr)
    {
        topUniform->setValue(t);
    }
}

void SkySphere::setBottom(float b)
{
    if (bottomUniform != nullptr)
    {
        bottomUniform->setValue(b);
    }
}

void SkySphere::setFloorColor(osg::Vec4 fc)
{
    if (floorColorUniform != nullptr)
    {
        floorColorUniform->setValue(fc);
    }
}

void SkySphere::setTrueNorth(float trueNorth)
{
    northAngle = trueNorth;
    if (skyNorthSlider)
        skyNorthSlider->setValue(northAngle);
}
bool SkySphere::update()
{
    float nearValue = coVRConfig::instance()->nearClip();
    float farValue = coVRConfig::instance()->farClip();
    float scale = nearValue + (farValue - nearValue) * 0.5; // scale the sphere just between the clipping distances, depth is set by shader

    const osg::Matrix &m = cover->getObjectsXform()->getMatrix();

    skyRootNode->setMatrix(
        osg::Matrix::scale(scale, scale, scale) * osg::Matrix::rotate(osg::DegreesToRadians(-northAngle), osg::Vec3(0, 0, 1)) * osg::Matrix::rotate(m.getRotate())
        * osg::Matrix::translate(cover->getViewerMat().getTrans()));

    if (m_mode == AUTO)
    {
        // TODO: automatically select appropriate sky
    }
    else if (m_mode == EPHEMERIS)
    {
#ifdef HAVE_EPHEMERIS
        if (m_ephemeralSky)
        {
            m_ephemeralSky->update();
        }
#endif
    }

    return false; // don't request that scene be re-rendered
}

SkySphere *SkySphere::instance()
{
    if (!s_instance)
        s_instance = new SkySphere;
    return s_instance;
}

COVERPLUGIN(SkySphere)
