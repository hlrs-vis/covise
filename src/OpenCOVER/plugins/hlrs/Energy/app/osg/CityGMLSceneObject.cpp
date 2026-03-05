#include "CityGMLSceneObject.h"
#include "app/osg/ui/citygml/CityGMLDeviceSensor.h"
#include "presentation/CityGMLBuilding.h"
#include "presentation/OsgTxtInfoboard.h"

#include <osg/MatrixTransform>

#include <cassert>

using namespace core::utils::osgUtils;
namespace fs = boost::filesystem;

CityGMLSceneObject::CityGMLSceneObject(osg::ref_ptr<osg::ClipNode> rootGroup,
    osg::ref_ptr<osg::Switch> parent)
    : m_coverRootGroup(rootGroup)
    , m_parent(parent)
    , m_root(new osg::Group())
    , m_pv(new osg::Group())
{
    assert(parent && "CityGMLSystem: parent must not be null");
    init();
}

CityGMLSceneObject::~CityGMLSceneObject()
{
    if (m_root)
    {
        restoreCityGMLDefaultStatesets();
        for (unsigned int i = 0; i < m_root->getNumChildren(); ++i)
        {
            auto child = m_root->getChild(i);
            m_coverRootGroup->addChild(child);
        }
        core::utils::osgUtils::deleteChildrenFromOtherGroup(m_root,
            m_coverRootGroup);
    }
}

void CityGMLSceneObject::init()
{
    m_root->setName("CityGML");
    m_pv->setName("PVPanels");
    m_parent->addChild(m_root);
}

void CityGMLSceneObject::update()
{
    for (auto &[name, sensor] : m_sensorMap)
        sensor->update();
}

void CityGMLSceneObject::updateTime(int timestep)
{
    for (auto &[name, sensor] : m_sensorMap)
        sensor->updateTime(timestep);
}

void CityGMLSceneObject::addCityGMLObject(const std::string &name,
    osg::ref_ptr<osg::Group> citygmlObjGroup)
{
    if (!citygmlObjGroup->getNumChildren())
        return;

    if (m_sensorMap.find(name) != m_sensorMap.end())
        return;

    auto geodes = core::utils::osgUtils::getGeodes(citygmlObjGroup);
    if (geodes->empty())
        return;

    // store default stateset
    saveCityGMLObjectDefaultStateSet(name, *geodes);

    auto boundingbox = core::utils::osgUtils::getBoundingBox(*geodes);
    auto infoboardPos = Pos(boundingbox.center().x(), boundingbox.center().y(), boundingbox.center().z());
    infoboardPos.z += (boundingbox.zMax() - boundingbox.zMin()) / 2 + boundingbox.zMin();
    auto infoboard = std::make_unique<OsgTxtInfoboard>(
        infoboardPos, name, "DroidSans-Bold.ttf", 50, 50, 2.0f, 0.1, 2);
    auto building = std::make_unique<CityGMLBuilding>(*geodes);
    auto sensor = std::make_unique<CityGMLDeviceSensor>(
        citygmlObjGroup, std::move(infoboard), std::move(building));
    m_sensorMap.insert({ name, std::move(sensor) });
}

void CityGMLSceneObject::addCityGMLObjects(osg::ref_ptr<osg::Group> citygmlGroup)
{
    for (unsigned int i = 0; i < citygmlGroup->getNumChildren(); ++i)
    {
        osg::ref_ptr<osg::Group> child = dynamic_cast<osg::Group *>(citygmlGroup->getChild(i));
        if (!child)
            continue;
        const auto &name = child->getName();

        // handle quad tree optimized scenegraph
        if (name == "GROUP" || name == "")
        {
            addCityGMLObjects(child);
            continue;
        }

        addCityGMLObject(name, child);
    }
}

void CityGMLSceneObject::move(const Pos &pos)
{
    if (!isActive(m_parent, m_root))
        return;
    transform(osg::Vec3(pos.x, pos.y, pos.z), {});
}

void CityGMLSceneObject::enable(const osg::Vec3 &translation)
{
    if (m_sensorMap.empty())
    {
        for (unsigned int i = 0; i < m_coverRootGroup->getNumChildren(); ++i)
        {
            osg::ref_ptr<osg::MatrixTransform> child = dynamic_cast<osg::MatrixTransform *>(m_coverRootGroup->getChild(i));
            if (!child)
                continue;

            auto name = child->getName();
            if (name.find(".gml") == std::string::npos)
                continue;

            addCityGMLObjects(child);
            m_root->addChild(child);
            child->setMatrix(osg::Matrix::translate(translation));
            transform(translation, {});
        }
        core::utils::osgUtils::deleteChildrenFromOtherGroup(m_coverRootGroup,
            m_root);
    }
    m_enabled = !m_enabled;
    if (m_enabled)
        switchTo(m_root, m_parent);
}

void CityGMLSceneObject::enablePV(bool on)
{
    // if (!m_enabled)
    // {
    //     std::cerr << "Not enabled!" << "\n";
    //     return;
    // }

    if (m_pv == nullptr)
    {
        std::cerr << "Error: No PV group found. Please enable GML first." << "\n";
        return;
    }
    // TODO: add a check if the group is already added and make sure its safe to
    // remove it
    osg::ref_ptr<osg::MatrixTransform> gmlRoot = dynamic_cast<osg::MatrixTransform *>(m_root->getChild(0));
    if (gmlRoot->containsNode(m_pv))
    {
        gmlRoot->removeChild(m_pv);
    }
    else
    {
        gmlRoot->addChild(m_pv);
    }
}

osg::ref_ptr<osg::Node> CityGMLSceneObject::readPVModel(
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
                std::cerr << "Error: Could not load solar panel model from " << path
                          << std::endl;
                continue;
            }
            break;
        }
    }
    return masterPanel;
}

void CityGMLSceneObject::initPV(const boost::filesystem::path &modelDir,
    const std::map<std::string, PVData> &pvDataMap,
    float maxPVIntensity)
{
    auto masterPanel = readPVModel(modelDir, "solarpanel_1k_resized");

    if (!masterPanel)
    {
        std::cerr << "Error: Could not load solar panel model. Make sure to define the "
                     "correct 3DModelDir in EnergyCampus.toml."
                  << std::endl;
        return;
    }


    // for only textured geometry data
    auto masterGeometryData = instancing::extractAllGeometryData(masterPanel);
    if (masterGeometryData.empty())
    {
        std::cerr << "Error: No geometry data found in the solar panel model."
                  << std::endl;
        return;
    }

    processPVDataMap(masterGeometryData, pvDataMap, maxPVIntensity);
}

void CityGMLSceneObject::processPVDataMap(
    const std::vector<core::utils::osgUtils::instancing::GeometryData>
        &masterGeometryData,
    const std::map<std::string, PVData> &pvDataMap, float maxPVIntensity)
{
    using namespace core::utils::osgUtils;

    if (m_sensorMap.empty())
    {
        std::cerr << "Error: No cityGML objects found." << std::endl;
        return;
    }
    
    // m_pv = new osg::Group();
    // m_pv->setName("PVPanels");

    osg::ref_ptr<osg::Group> gmlRoot = m_root->getChild(0)->asGroup();
    if (gmlRoot)
    {
        gmlRoot->addChild(m_pv);
    }
    else
    {
        std::cerr << "Error: m_cityGML->getChild(0) is not a valid group." << std::endl;
        return;
    }

    m_panels = SolarPanelList();
    // Rotate the solar panel by 90 degrees around the Z-axis to align it with the
    // desired orientation.
    auto rotationZ = osg::Matrix::rotate(osg::DegreesToRadians(90.0f), 0, 0, 1);
    auto rotationX = osg::Matrix::rotate(osg::DegreesToRadians(45.0f), 1, 0, 0);
    SolarPanelConfig config;
    config.masterGeometryData = masterGeometryData;
    config.rotation = rotationZ * rotationX;
    config.parent = m_pv;
    // panel is 1.7m x 1.0m x 0.4m
    config.panelWidth = 1.0f;
    config.panelHeight = 1.7f;
    config.zOffset = sin(osg::PI / 4) * config.panelHeight;

    for (const auto &[id, data] : pvDataMap)
    {
        try
        {
            auto &cityGMLObj = m_sensorMap.at(id);
            config.colorIntensity = core::utils::color::getTrafficLightColor(
                data.energyYearlyKWhMax / data.area, maxPVIntensity);
            processSolarPanelDrawables(data, cityGMLObj->getDrawables(), m_panels, config);
        }
        catch (const std::out_of_range &)
        {
            std::cerr << "Error: Could not find cityGML object with ID " << id
                      << " in m_cityGMLObjs." << std::endl;
            continue;
        }
    }
}

void CityGMLSceneObject::processSolarPanelDrawable(SolarPanelList &solarPanels,
    const SolarPanelConfig &config)
{
    if (!config.valid())
    {
        std::cerr << "Error: Invalid SolarPanelConfig." << std::endl;
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

std::unique_ptr<SolarPanel> CityGMLSceneObject::createSolarPanel(
    const std::string &name, osg::ref_ptr<osg::Group> parent,
    const std::vector<core::utils::osgUtils::instancing::GeometryData>
        &masterGeometryData,
    const osg::Matrix &matrix, const Color &colorIntensity)
{
    using namespace core::utils::osgUtils;
    auto solarPanelInstance = instancing::createInstance(masterGeometryData, matrix);
    solarPanelInstance->setName(name);

    auto solarPanel = std::make_unique<SolarPanel>(solarPanelInstance);
    solarPanel->applyColor(colorIntensity);
    parent->addChild(solarPanelInstance);
    return std::move(solarPanel);
}

void CityGMLSceneObject::processSolarPanelDrawables(
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
            std::cerr << "Error: Drawable is not a Geode." << std::endl;
            continue;
        }
        processSolarPanelDrawable(m_panels, config);
    }
}

void CityGMLSceneObject::transform(const osg::Vec3 &translation,
    const osg::Quat &rotation, const osg::Vec3 &scale)
{
    assert(m_root && "CityGML group is not initialized.");
    if (m_root->getNumChildren() == 0)
    {
        std::cout << "No CityGML objects to transform." << std::endl;
        return;
    }
    for (unsigned int i = 0; i < m_root->getNumChildren(); ++i)
    {
        osg::ref_ptr<osg::Node> child = m_root->getChild(i);
        if (auto mt = dynamic_cast<osg::MatrixTransform *>(child.get()))
        {
            osg::Matrix matrix = osg::Matrix::translate(translation) * osg::Matrix::rotate(rotation) * osg::Matrix::scale(scale);
            mt->setMatrix(matrix);
        }
        else
        {
            std::cerr << "Child is not a MatrixTransform." << std::endl;
        }
    }
}

void CityGMLSceneObject::saveCityGMLObjectDefaultStateSet(const std::string &name,
    const Geodes &citygmlGeodes)
{
    Geodes geodesCopy(citygmlGeodes.size());
    for (auto i = 0; i < citygmlGeodes.size(); ++i)
    {
        auto geode = citygmlGeodes[i];
        geodesCopy[i] = dynamic_cast<osg::Geode *>(geode->clone(osg::CopyOp::DEEP_COPY_STATESETS));
    }
    m_defaultStateSets.insert({ name, std::move(geodesCopy) });
}

void CityGMLSceneObject::restoreGeodesStatesets(CityGMLDeviceSensor &sensor,
    const std::string &name,
    const Geodes &citygmlGeodes)
{
    if (m_defaultStateSets.find(name) == m_defaultStateSets.end())
        return;

    if (citygmlGeodes.empty())
        return;

    for (auto i = 0; i < citygmlGeodes.size(); ++i)
    {
        auto gmlDefault = citygmlGeodes[i];
        osg::ref_ptr<osg::Geode> toRestore = sensor.getDrawable(i)->asGeode();
        if (toRestore)
        {
            toRestore->setStateSet(gmlDefault->getStateSet());
        }
    }
}

void CityGMLSceneObject::restoreCityGMLDefaultStatesets()
{
    for (auto &[name, sensor] : m_sensorMap)
    {
        osg::ref_ptr<osg::Group> sensorParent = sensor->getParent();
        if (!sensorParent)
            continue;

        restoreGeodesStatesets(*sensor, name, m_defaultStateSets[name]);
    }
    m_defaultStateSets.clear();
}
