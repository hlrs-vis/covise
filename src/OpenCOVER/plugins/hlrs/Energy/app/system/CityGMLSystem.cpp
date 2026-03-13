#include "CityGMLSystem.h"
#include "app/cover/ui/CityGMLUI.h"
#include "app/osg/SolarPanelSceneObject.h"

#include <cover/coVRAnimationManager.h>

#include <memory>

namespace fs = boost::filesystem;
using namespace opencover;
using namespace opencover::utils::read;
using namespace core::utils::osgUtils;
using namespace core::simulation::power;

namespace
{
struct StaticPowerData
{
    std::string name;
    int id;
    float val2019;
    float val2023;
    float average;
    std::string citygml_id;
};

struct StaticPowerCampusData
{
    std::string citygml_id;
    float yearlyConsumption;
};

void setAnimationTimesteps(size_t maxTimesteps, const void *who)
{
    if (maxTimesteps > opencover::coVRAnimationManager::instance()->getNumTimesteps())
        opencover::coVRAnimationManager::instance()->setNumTimesteps(maxTimesteps, who);
}
} // namespace

CityGMLSystem::CityGMLSystem(opencover::coVRPlugin *plugin,
    opencover::ui::Menu *parentMenu,
    osg::ref_ptr<osg::ClipNode> rootGroup,
    osg::ref_ptr<osg::Switch> parent,
    core::interface::ILogger &logger)
    : core::ClassLogger(logger, "CityGMLSystem")
    , m_cityGMLUI("CityGMLSystem", parentMenu, { plugin->configFloat("CityGML", "X", 0.0f)->value(), plugin->configFloat("CityGML", "Y", 0.0f)->value(), plugin->configFloat("CityGML", "Z", 0.0f)->value() })
    , m_gmlSceneObject(rootGroup, parent, logger)
    , m_config {
        plugin->configString("Simulation", "pvDir", "default")->value(),
        plugin->configString("Simulation", "staticInfluxCSV", "default")->value(),
        plugin->configString("Simulation", "campusPath", "default")->value(),
        plugin->configString("Simulation", "staticPower", "default")->value(),
        plugin->configString("Simulation", "3dModelDir", "default")->value()
    }
    , m_enabled(false)
    // , m_logger(logger, "CityGMLSystem")
{
}

void CityGMLSystem::init()
{
    initUICallbacks();
}

void CityGMLSystem::initUICallbacks()
{
    m_cityGMLUI.setButtonGroupCallback([&](bool on)
        { enableScene(on); });

    m_cityGMLUI.InfluxCSV()->setCallback([&](bool on)
        {
        if (on)
            applyInfluxCSVToCityGML(m_config.influxPath, true); });

    m_cityGMLUI.StaticCampusPower()->setCallback([&](bool on)
        {
        if (on)
            applyStaticDataCampusToCityGML(m_config.campusPath, true); });

    m_cityGMLUI.StaticPower()->setCallback([&](bool on)
        {
        if (on)
            applyStaticDataToCityGML(m_config.staticPower, true); });

    m_cityGMLUI.setPVBtnCallback([&](bool on)
        { 
            if (!m_pvSceneObject)
            {
                auto solarPanelsDir = fs::path(m_config.modelDir + "/power/SolarPanel");
                addSolarPanels(solarPanelsDir);
            }
            m_pvSceneObject->enable(); });

    auto updateFunction = [this](auto &value)
    {
        m_gmlSceneObject.move(m_cityGMLUI.getTranslation());
    };

    for (auto &[_, editField] : m_cityGMLUI.getEditFields())
        editField->setCallback(updateFunction);

    m_cityGMLUI.setColorMapCallback([this](const opencover::ColorMap &cm)
        {
            if (m_gmlSceneObject.enabled()) {
              enableScene(false, false);
              enableScene(true, false);
            } });
}

void CityGMLSystem::enable(bool on)
{
    m_enabled = on;
    enableScene(on, true);
}

bool CityGMLSystem::isEnabled() const { return m_enabled; }

void CityGMLSystem::update()
{
    m_gmlSceneObject.update();
}

void CityGMLSystem::updateTime(int timestep)
{
    m_gmlSceneObject.updateTime(timestep);
}

void CityGMLSystem::processPVRow(const CSVStream::CSVRow &row,
    std::map<std::string, PVData> &pvDataMap,
    float &maxPVIntensity)
{
    PVData pvData;
    ACCESS_CSV_ROW(row, "gml_id", pvData.cityGMLID);

    if (!m_gmlSceneObject.contains(pvData.cityGMLID))
    {
        std::cerr << "Error: Could not find cityGML object with ID " << pvData.cityGMLID
                  << std::endl;
        return;
    }

    ACCESS_CSV_ROW(row, "energy_yearly_kwh_max", pvData.energyYearlyKWhMax);
    ACCESS_CSV_ROW(row, "pv_area_qm", pvData.pvAreaQm);
    ACCESS_CSV_ROW(row, "area_qm", pvData.area);
    ACCESS_CSV_ROW(row, "n_modules_max", pvData.numPanelsMax);

    if (pvData.pvAreaQm == 0)
    {
        error("pvAreaQm is 0 for cityGML object with ID " + pvData.cityGMLID);
        return;
    }

    maxPVIntensity = std::max(pvData.energyYearlyKWhMax / pvData.area, maxPVIntensity);
    pvDataMap.insert({ pvData.cityGMLID, pvData });
}

void CityGMLSystem::updateInfluxColorMaps(
    float min, float max,
    std::shared_ptr<core::simulation::Simulation> powerSimulation,
    const std::string &colormapName, const std::string &species,
    const std::string &unit)
{
    if (!m_cityGMLUI.InfluxArrow()->state())
        return;

    auto cb = m_cityGMLUI.colorBar();

    cb->setMinMax(min, max);
    cb->setSpecies(species);
    cb->setUnit(unit);
    auto halfSpan = (max - min) / 2;
    cb->setMinBounds(min - halfSpan, min + halfSpan);
    cb->setMaxBounds(max - halfSpan, max + halfSpan);

    for (auto &[name, sensor] : m_gmlSceneObject)
    {
        std::string sensorName = name;
        auto values = powerSimulation->getTimedependentScalar("res_mw", sensorName);
        if (!values)
        {
            // std::cerr << "No res_mw data found for sensor: " << sensorName << std::endl;
            error("No res_mw data found for sensor: " + sensorName);
            continue;
        }

        auto steps = cb->colorMap().steps();
        auto colorMapName = colormapName;
        if (colorMapName == core::simulation::INVALID_UNIT)
            colorMapName = cb->colorMap().name();
        cb->setColorMap(colorMapName);
        cb->setSteps(steps);
        sensor->setColorMapInShader(m_cityGMLUI.colorBar()->colorMap());
        sensor->setDataInShader(*values, min, max);

        std::vector<std::string> texts;
        std::transform(
            values->begin(), values->end(), std::back_inserter(texts),
            [unit](const auto &v)
            { return std::to_string(v) + " " + unit; });
        sensor->updateTxtBoxTexts(texts);
    }
}

std::pair<std::map<std::string, PVData>, float> CityGMLSystem::loadPVData(
    CSVStream &pvStream)
{
    CSVStream::CSVRow row;
    std::map<std::string, PVData> pvDataMap;
    float maxPVIntensity = 0;

    while (pvStream.readNextRow(row))
        processPVRow(row, pvDataMap, maxPVIntensity);

    return { pvDataMap, maxPVIntensity };
}

void CityGMLSystem::addSolarPanels(const fs::path &dirPath)
{
    if (!fs::exists(dirPath))
        return;

    fs::path pvDirPath(m_config.pvDir);
    if (!fs::exists(pvDirPath))
    {
        error("Error: PV directory does not exist: " + m_config.pvDir);
        return;
    }

    auto pvStreams = getCSVStreams(pvDirPath);
    auto it = pvStreams.find("pv");
    if (it == pvStreams.end())
    {
        error("Error: Could not find PV data in " + m_config.pvDir);
        return;
    }

    CSVStream &pvStream = it->second;
    if (!pvStream)
    {
        error("Error: Could not load solar panel data from PV stream.");
        return;
    }

    auto [pvDataMap, maxPVIntensity] = loadPVData(pvStream);
    m_pvSceneObject = std::make_unique<SolarPanelSceneObject>(&m_gmlSceneObject, m_gmlSceneObject.getRoot()->getChild(0)->asGroup(), dirPath, pvDataMap, maxPVIntensity, getLogger());
}

auto CityGMLSystem::readStaticCampusData(CSVStream &stream, float &max, float &min,
    float &sum)
{
    std::vector<StaticPowerCampusData> yearlyValues;
    if (!stream || stream.getHeader().size() < 1)
        return yearlyValues;
    CSVStream::CSVRow row;
    while (stream.readNextRow(row))
    {
        StaticPowerCampusData data;
        ACCESS_CSV_ROW(row, "gml_id", data.citygml_id);
        ACCESS_CSV_ROW(row, "energy_mwh", data.yearlyConsumption);

        max = std::max(max, data.yearlyConsumption);

        if (min == -1)
        {
            min = data.yearlyConsumption;
        }
        min = std::min(min, data.yearlyConsumption);

        yearlyValues.push_back(data);
    }
    return yearlyValues;
}

void CityGMLSystem::applyStaticDataCampusToCityGML(const std::string &filePath,
    bool updateColorMap)
{
    if (m_gmlSceneObject.enabled())
        return;
    if (!fs::exists(filePath))
        return;
    auto csvStream = CSVStream(filePath);
    float max = 0, min = -1;
    float sum = 0;

    auto values = readStaticCampusData(csvStream, max, min, sum);

    auto cb = m_cityGMLUI.colorBar();
    //   max = 400.00f;
    if (updateColorMap)
    {
        cb->setMinMax(min, max);
        cb->setSpecies("Yearly Consumption");
        cb->setUnit("MWh");
        auto halfSpan = (max - min) / 2;
        cb->setMinBounds(min - halfSpan, min + halfSpan);
        cb->setMaxBounds(max - halfSpan, max + halfSpan);
    }

    for (const auto &v : values)
    {
        if (auto sensor = m_gmlSceneObject.find(v.citygml_id))
        {
            sensor->updateTimestepColors({ v.yearlyConsumption }, cb->colorMap());
            sensor->updateTxtBoxTexts(
                { "Yearly Consumption: " + std::to_string(v.yearlyConsumption) + " MWh" });
        }
    }
    setAnimationTimesteps(1, m_gmlSceneObject.getRoot());
}

void CityGMLSystem::enableScene(bool on, bool updateColorMap)
{
    if (on)
    {
        auto translation = m_cityGMLUI.getTranslation();
        osg::Vec3 trans(translation.x, translation.y, translation.z);
        m_gmlSceneObject.enable(trans);
    }
}

auto CityGMLSystem::readStaticPowerData(CSVStream &stream, float &max, float &min,
    float &sum)
{
    std::vector<StaticPowerData> powerData;
    if (!stream || stream.getHeader().size() < 1)
        return powerData;
    CSVStream::CSVRow row;
    while (stream.readNextRow(row))
    {
        StaticPowerData data;
        ACCESS_CSV_ROW(row, "name", data.name);
        ACCESS_CSV_ROW(row, "2019", data.val2019);
        ACCESS_CSV_ROW(row, "2023", data.val2023);
        ACCESS_CSV_ROW(row, "average", data.average);
        ACCESS_CSV_ROW(row, "building_id", data.id);
        ACCESS_CSV_ROW(row, "citygml_id", data.citygml_id);

        max = std::max(max, data.val2019);
        max = std::max(max, data.val2023);
        max = std::max(max, data.average);

        if (min == -1)
        {
            min = data.val2019;
            min = data.val2023;
        }
        min = std::min(min, data.val2019);
        min = std::min(min, data.val2023);
        min = std::min(min, data.average);

        powerData.push_back(data);
    }
    return powerData;
}

void CityGMLSystem::applyStaticDataToCityGML(const std::string &filePathToInfluxCSV,
    bool updateColorMap)
{
    if (m_gmlSceneObject.enabled())
        return;
    if (!fs::exists(filePathToInfluxCSV))
        return;
    auto csvStream = CSVStream(filePathToInfluxCSV);
    float max = 0, min = -1;
    float sum = 0;

    auto values = readStaticPowerData(csvStream, max, min, sum);
    //   max = 7000.0f;
    auto cb = m_cityGMLUI.colorBar();
    if (updateColorMap)
    {
        cb->setMinMax(min, max);
        cb->setSpecies("Yearly Consumption");
        cb->setUnit("kWh");
        auto halfSpan = (max - min) / 2;
        cb->setMinBounds(min - halfSpan, min + halfSpan);
        cb->setMaxBounds(max - halfSpan, max + halfSpan);
    }
    for (const auto &v : values)
    {
        if (auto sensor = m_gmlSceneObject.find(v.citygml_id))
        {
            sensor->updateTimestepColors({ v.val2019, v.val2023, v.average },
                cb->colorMap());
            sensor->updateTxtBoxTexts({ "2019: " + std::to_string(v.val2019) + " kWh",
                "2023: " + std::to_string(v.val2023) + " kWh",
                "Average: " + std::to_string(v.average) + " kWh" });
            sensor->updateTitleOfInfoboard(v.name);
        }
    }
    setAnimationTimesteps(3, m_gmlSceneObject.getRoot());
}

void CityGMLSystem::applyInfluxCSVToCityGML(const std::string &filePathToInfluxCSV,
    bool updateColorMap)
{
    if (m_gmlSceneObject.enabled())
        return;
    if (!fs::exists(filePathToInfluxCSV))
        return;
    auto csvStream = CSVStream(filePathToInfluxCSV);
    float max = 0, min = -1;
    float sum = 0;
    int timesteps = 0;
    auto values = getInfluxDataFromCSV(csvStream, max, min, sum, timesteps);
    auto cb = m_cityGMLUI.colorBar();

    if (updateColorMap)
    {
        auto distributionCenter = sum / (timesteps * values->size());
        cb->setMinMax(min, max);
        cb->setMinBounds(0, distributionCenter);
        cb->setMaxBounds(distributionCenter, max);
    }

    for (auto &[name, values] : *values)
    {
        if (auto sensor = m_gmlSceneObject.find(name))
        {
            sensor->updateTimestepColors(values, cb->colorMap());
            sensor->updateTxtBoxTexts({ "NOT IMPLEMENTED YET" });
        }
    }
    setAnimationTimesteps(timesteps, m_gmlSceneObject.getRoot());
}

std::unique_ptr<std::map<std::string, std::vector<float>>>
CityGMLSystem::getInfluxDataFromCSV(opencover::utils::read::CSVStream &stream,
    float &max, float &min, float &sum,
    int &timesteps)
{
    using FloatMap = std::map<std::string, std::vector<float>>;
    const auto &headers = stream.getHeader();
    FloatMap values;
    if (stream && headers.size() > 1)
    {
        CSVStream::CSVRow row;
        // while (stream >> row) {
        while (stream.readNextRow(row))
        {
            for (auto cityGMLBuildingName : headers)
            {
                if (!m_gmlSceneObject.contains(cityGMLBuildingName))
                    continue;
                float value = 0;
                ACCESS_CSV_ROW(row, cityGMLBuildingName, value);
                if (value > max)
                    max = value;
                else if (value < min || min == -1)
                    min = value;
                sum += value;
                if (values.find(cityGMLBuildingName) == values.end())
                    values.insert({ cityGMLBuildingName, { value } });
                else
                    values[cityGMLBuildingName].push_back(value);
            }
            ++timesteps;
        }
    }
    return std::make_unique<FloatMap>(values);
}
