#include "SolarPanelSceneObject.h"
#include "app/osg/CityGMLSceneObject.h"
#include <lib/core/ClassLogger.h>
#include <lib/core/utils/color.h>
#include <osg/MatrixTransform>

namespace fs = boost::filesystem;
using namespace core::utils::osgUtils;

SolarPanelSceneObject::SolarPanelSceneObject(CityGMLSceneObject *gmlObj, osg::ref_ptr<osg::Group> parent, const boost::filesystem::path &modelDir, const PVDataMap &map, float maxPVIntensity, core::interface::ILogger &logger)
    : core::ClassLogger(logger, "SolarPanelSceneObject")
    , m_parent(parent)
    , m_root(new osg::Group())
{
    init(gmlObj, modelDir, map, maxPVIntensity);
}

SolarPanelSceneObject::~SolarPanelSceneObject() {
    m_parent->removeChild(m_root);
}

void SolarPanelSceneObject::enable()
{
    m_enabled = !m_enabled;

    if (m_parent->containsNode(m_root) && !m_enabled)
        m_parent->removeChild(m_root);
    else
        m_parent->addChild(m_root);
}

osg::ref_ptr<osg::Node> SolarPanelSceneObject::readPVModel(
    const fs::path &modelDir, const std::string &nameInModelDir)
{
    osg::ref_ptr<osg::Node> masterPanel;
    for (auto &file : fs::directory_iterator(modelDir))
    {
        if (fs::is_regular_file(file) && file.path().extension() == ".obj")
        {
            auto path = file.path();
            auto name = path.stem().string();
            if (name.find(nameInModelDir) == std::string::npos)
                continue;
            osg::ref_ptr<osgDB::Options> options = new osgDB::Options;
            options->setOptionString("DIFFUSE=0 SPECULAR=1 SPECULAR_EXPONENT=2 OPACITY=3");

            masterPanel = core::utils::osgUtils::readFileViaOSGDB(path.string(), options);
            if (!masterPanel)
            {
                error("Could not load solar panel model from " + path.string());
                continue;
            }
            break;
        }
    }
    return masterPanel;
}

void SolarPanelSceneObject::init(CityGMLSceneObject *gmlObj, const boost::filesystem::path &modelDir,
    const std::map<std::string, PVData> &pvDataMap,
    float maxPVIntensity)
{

    m_root->setName("PVPanels");
    auto masterPanel = readPVModel(modelDir, "solarpanel_1k_resized");

    if (!masterPanel)
    {
        error("Could not load solar panel model. Make sure to define the "
                     "correct 3DModelDir in EnergyCampus.toml.");
        return;
    }

    // for only textured geometry data
    auto masterGeometryData = instancing::extractAllGeometryData(masterPanel);
    if (masterGeometryData.empty())
    {
        error("No geometry data found in the solar panel model.");
        return;
    }

    processPVDataMap(gmlObj, masterGeometryData, pvDataMap, maxPVIntensity);
}

void SolarPanelSceneObject::processPVDataMap(CityGMLSceneObject *gmlObj,
    const std::vector<core::utils::osgUtils::instancing::GeometryData>
        &masterGeometryData,
    const std::map<std::string, PVData> &pvDataMap, float maxPVIntensity)
{
    using namespace core::utils::osgUtils;

    if (!m_parent)
    {
        error("No parent found to attach solarpanels to.");
        return;
    }

    m_parent->addChild(m_root);

    m_panels = SolarPanelList();
    // Rotate the solar panel by 90 degrees around the Z-axis to align it with the
    // desired orientation.
    auto rotationZ = osg::Matrix::rotate(osg::DegreesToRadians(90.0f), 0, 0, 1);
    auto rotationX = osg::Matrix::rotate(osg::DegreesToRadians(45.0f), 1, 0, 0);
    SolarPanelConfig config;
    config.masterGeometryData = masterGeometryData;
    config.rotation = rotationZ * rotationX;
    config.parent = m_root;
    // panel is 1.7m x 1.0m x 0.4m
    config.panelWidth = 1.0f;
    config.panelHeight = 1.7f;
    config.zOffset = sin(osg::PI / 4) * config.panelHeight;

    for (const auto &[id, data] : pvDataMap)
    {
        try
        {
            auto cityGMLObj = gmlObj->find(id);
            config.colorIntensity = core::utils::color::getTrafficLightColor(
                data.energyYearlyKWhMax / data.area, maxPVIntensity);
            processSolarPanelDrawables(data, cityGMLObj->getDrawables(), m_panels, config);
        }
        catch (const std::out_of_range &)
        {
            warn("Could not find cityGML object with ID " + id
                      + " in m_cityGMLObjs to attach solarpanels.");
            continue;
        }
    }
}

void SolarPanelSceneObject::processSolarPanelDrawable(SolarPanelList &solarPanels,
    const SolarPanelConfig &config)
{
    if (!config.valid())
    {
        error("Invalid SolarPanelConfig.");
        return;
    }
    auto bb = config.geode->getBoundingBox();
    auto minBB = bb._min;
    auto maxBB = bb._max;
    auto roofWidth = maxBB.x() - minBB.x();
    auto roofHeight = maxBB.y() - minBB.y();
    auto roofCenter = bb.center();
    auto z = maxBB.z() + config.zOffset;

    osg::ref_ptr<osg::Group> pvPanelsTransform = new osg::Group();
    pvPanelsTransform->setName("PVPanels");

    int dividedBy = 10;
    int maxPanels = config.numMaxPanels / dividedBy;
    if (maxPanels == 0)
        maxPanels = 1;

    int numPanelsPerRow = static_cast<int>(std::sqrt(maxPanels));
    int numPanelRows = (maxPanels + numPanelsPerRow - 1) / numPanelsPerRow;

    float availableWidthForSpacingX = roofWidth - (numPanelsPerRow * config.panelWidth);
    float availableHeightForSpacingY = roofHeight - (numPanelRows * config.panelHeight);

    float spacingX = (numPanelsPerRow > 1)
        ? std::min(0.5f, availableWidthForSpacingX / (numPanelsPerRow - 1))
        : 0.0f;
    float spacingY = (numPanelRows > 1)
        ? std::min(0.5f, availableHeightForSpacingY / (numPanelRows - 1))
        : 0.0f;

    float totalWidthOfPanelsX = (numPanelsPerRow * config.panelWidth) + ((numPanelsPerRow - 1) * spacingX);
    float totalHeightOfPanelsY = (numPanelRows * config.panelHeight) + ((numPanelRows - 1) * spacingY);

    auto startX = roofCenter.x() - (totalWidthOfPanelsX / 2.0f) + (config.panelWidth / 2.0f);
    auto startY = roofCenter.y() - (totalHeightOfPanelsY / 2.0f) + (config.panelHeight / 2.0f);

    for (int i = 0; i < maxPanels; ++i)
    {
        int row = i / numPanelsPerRow;
        int col = i % numPanelsPerRow;

        auto x = startX + (col * (config.panelWidth + spacingX));
        auto y = startY + (row * (config.panelHeight + spacingY));

        auto position = osg::Vec3(x, y, z);
        osg::Matrix matrix = config.rotation * osg::Matrix::translate(position);
        auto solarPanel = createSolarPanel("SolarPanel_" + std::to_string(i), pvPanelsTransform,
            config.masterGeometryData, matrix, config.colorIntensity);
        solarPanels.push_back(std::move(solarPanel));
    }

    config.parent->addChild(pvPanelsTransform);
}

std::unique_ptr<SolarPanel> SolarPanelSceneObject::createSolarPanel(
    const std::string &name, osg::ref_ptr<osg::Group> parent,
    const std::vector<core::utils::osgUtils::instancing::GeometryData>
        &masterGeometryData,
    const osg::Matrix &matrix, const Color &colorIntensity)
{
    using namespace core::utils::osgUtils;
    auto solarPanelInstance = instancing::createInstance(masterGeometryData, matrix);
    solarPanelInstance->setName(name);

    auto solarPanel = std::make_unique<SolarPanel>(solarPanelInstance, getLogger());
    solarPanel->applyColor(colorIntensity);
    parent->addChild(solarPanelInstance);
    return std::move(solarPanel);
}

void SolarPanelSceneObject::processSolarPanelDrawables(
    const PVData &data, const std::vector<osg::ref_ptr<osg::Node>> drawables,
    SolarPanelList &solarPanels, SolarPanelConfig &config)
{
    for (auto drawable : drawables)
    {
        const auto &name = drawable->getName();
        if (name.find("RoofSurface") == std::string::npos)
        {
            continue;
        }

        if (data.numPanelsMax == 0)
            continue;
        config.numMaxPanels = data.numPanelsMax;
        config.geode = drawable->asGeode();
        if (!config.geode)
        {
            warn("Drawable is not a Geode. Solarpanels cannot be attached.");
            continue;
        }
        processSolarPanelDrawable(m_panels, config);
    }
}
