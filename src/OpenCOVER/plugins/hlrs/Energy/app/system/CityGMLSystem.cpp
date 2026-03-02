#include "CityGMLSystem.h"
#include "app/cover/ui/CityGMLUI.h"
#include "app/osg/presentation/CityGMLBuilding.h"
#include "app/osg/presentation/OsgTxtInfoboard.h"

#include <cover/coVRAnimationManager.h>

#include <memory>
#include <osg/ClipNode>
#include <osg/MatrixTransform>

namespace fs = boost::filesystem;
using namespace opencover;
using namespace opencover::utils::read;
using namespace core::utils::osgUtils;
using namespace core::simulation::power;

namespace {
struct StaticPowerData {
  std::string name;
  int id;
  float val2019;
  float val2023;
  float average;
  std::string citygml_id;
};

struct StaticPowerCampusData {
  std::string citygml_id;
  float yearlyConsumption;
};

void setAnimationTimesteps(size_t maxTimesteps, const void *who) {
  if (maxTimesteps > opencover::coVRAnimationManager::instance()->getNumTimesteps())
    opencover::coVRAnimationManager::instance()->setNumTimesteps(maxTimesteps, who);
}
}  // namespace

CityGMLSystem::CityGMLSystem(opencover::coVRPlugin *plugin,
                             opencover::ui::Menu *parentMenu,
                             osg::ref_ptr<osg::ClipNode> rootGroup,
                             osg::ref_ptr<osg::Switch> parent)
    : m_coverRootGroup(rootGroup)
    , m_parent(parent)
    , m_cityGMLGroup(new osg::Group())
    , m_pvGroup(new osg::Group())
    , m_cityGMLUI("CityGMLSystem", parentMenu, CityGMLOrigin{ 
        plugin->configFloat("CityGML", "X", 0.0f)->value(), 
        plugin->configFloat("CityGML", "Y", 0.0f)->value(), 
        plugin->configFloat("CityGML", "Z", 0.0f)->value() 
      })
    , m_pvDir(plugin->configString("Simulation", "pvDir", "default")->value())
    , m_influxPath(plugin->configString("Simulation", "staticInfluxCSV", "default")->value())
    , m_campusPath(plugin->configString("Simulation", "campusPath", "default")->value())
    , m_staticPower(plugin->configString("Simulation", "staticPower", "default")->value())
    , m_modelDir(plugin->configString("Simulation", "3dModelDir", "default")->value())
    , m_enabled(false) {
  assert(parent && "CityGMLSystem: parent must not be null");
  assert(plugin && "CityGMLSystem: plugin must not be null");
  m_parent->addChild(m_cityGMLGroup);
}

CityGMLSystem::~CityGMLSystem() {
  if (m_cityGMLGroup) {
    restoreCityGMLDefaultStatesets();
    for (unsigned int i = 0; i < m_cityGMLGroup->getNumChildren(); ++i) {
      auto child = m_cityGMLGroup->getChild(i);
      m_coverRootGroup->addChild(child);
    }
    core::utils::osgUtils::deleteChildrenFromOtherGroup(m_cityGMLGroup,
                                                        m_coverRootGroup);
  }
}

void CityGMLSystem::init() {
  m_cityGMLGroup->setName("CityGML");
  initUICallbacks();
}

void CityGMLSystem::initUICallbacks()
{
    auto bindBackEnableCityGML = [&](bool on)
    { enableCityGML(on); };
    m_cityGMLUI.setInfluxCSVBtnCallback(bindBackEnableCityGML);
    m_cityGMLUI.setInfluxArrowBtnCallback(bindBackEnableCityGML);
    m_cityGMLUI.setStaticPowerBtnCallback(bindBackEnableCityGML);
    m_cityGMLUI.setStaticCampusPowerBtnCallback(bindBackEnableCityGML);
    m_cityGMLUI.setPVBtnCallback([&](bool on)
        {
            if (m_pvGroup == nullptr) {
              std::cerr << "Error: No PV group found. Please enable GML first." << std::endl;
              return;
            }
            // TODO: add a check if the group is already added and make sure its safe to
            // remove it
            osg::ref_ptr<osg::MatrixTransform> gmlRoot =
                dynamic_cast<osg::MatrixTransform *>(m_cityGMLGroup->getChild(0));
            if (gmlRoot->containsNode(m_pvGroup)) {
              gmlRoot->removeChild(m_pvGroup);
            } else {
              gmlRoot->addChild(m_pvGroup);
            } 
        }
    );

    auto updateFunction = [this](auto &value)
    {
        if (!isActive(m_parent, m_cityGMLGroup))
            return;
        auto translation = m_cityGMLUI.getTranslation();
        transform(osg::Vec3(translation.x, translation.y, translation.z), {});
    };
    m_cityGMLUI.setXCallback(updateFunction);
    m_cityGMLUI.setYCallback(updateFunction);
    m_cityGMLUI.setZCallback(updateFunction);

    m_cityGMLUI.setColorMapCallback([this](const opencover::ColorMap &cm)
        {
            if (isActive(m_parent, m_cityGMLGroup)) {
              enableCityGML(false, false);
              enableCityGML(true, false);
            }
        }
    );
}

void CityGMLSystem::enable(bool on) {
  m_enabled = on;
  enableCityGML(on, true);
}

bool CityGMLSystem::isEnabled() const { return m_enabled; }

void CityGMLSystem::update() {
  for (auto &[name, sensor] : m_sensorMap) sensor->update();
}

void CityGMLSystem::updateTime(int timestep) {
  for (auto &[name, sensor] : m_sensorMap) {
    sensor->updateTime(timestep);
  }
}

void CityGMLSystem::processPVRow(const CSVStream::CSVRow &row,
                                 std::map<std::string, PVData> &pvDataMap,
                                 float &maxPVIntensity) {
  PVData pvData;
  ACCESS_CSV_ROW(row, "gml_id", pvData.cityGMLID);

  if (m_sensorMap.find(pvData.cityGMLID) == m_sensorMap.end()) {
    std::cerr << "Error: Could not find cityGML object with ID " << pvData.cityGMLID
              << std::endl;
    return;
  }

  ACCESS_CSV_ROW(row, "energy_yearly_kwh_max", pvData.energyYearlyKWhMax);
  ACCESS_CSV_ROW(row, "pv_area_qm", pvData.pvAreaQm);
  ACCESS_CSV_ROW(row, "area_qm", pvData.area);
  ACCESS_CSV_ROW(row, "n_modules_max", pvData.numPanelsMax);

  if (pvData.pvAreaQm == 0) {
    std::cerr << "Error: pvAreaQm is 0 for cityGML object with ID "
              << pvData.cityGMLID << std::endl;
    return;
  }

  maxPVIntensity = std::max(pvData.energyYearlyKWhMax / pvData.area, maxPVIntensity);
  pvDataMap.insert({pvData.cityGMLID, pvData});
}

void CityGMLSystem::updateInfluxColorMaps(
    float min, float max,
    std::shared_ptr<core::simulation::Simulation> powerSimulation,
    const std::string &colormapName, const std::string &species,
    const std::string &unit)
{
    if (!m_cityGMLUI.getInfluxArrowBtnState())
        return;
    
    auto cb = m_cityGMLUI.colorBar();

    cb->setMinMax(min, max);
    cb->setSpecies(species);
    cb->setUnit(unit);
    auto halfSpan = (max - min) / 2;
    cb->setMinBounds(min - halfSpan, min + halfSpan);
    cb->setMaxBounds(max - halfSpan, max + halfSpan);

    for (auto &[name, sensor] : m_sensorMap)
    {
        std::string sensorName = name;
        auto values = powerSimulation->getTimedependentScalar("res_mw", sensorName);
        if (!values)
        {
            std::cerr << "No res_mw data found for sensor: " << sensorName << std::endl;
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
    CSVStream &pvStream) {
  CSVStream::CSVRow row;
  std::map<std::string, PVData> pvDataMap;
  float maxPVIntensity = 0;

  while (pvStream.readNextRow(row)) processPVRow(row, pvDataMap, maxPVIntensity);

  return {pvDataMap, maxPVIntensity};
}

osg::ref_ptr<osg::Node> CityGMLSystem::readPVModel(
    const fs::path &modelDir, const std::string &nameInModelDir) {
  osg::ref_ptr<osg::Node> masterPanel;
  for (auto &file : fs::directory_iterator(modelDir)) {
    if (fs::is_regular_file(file) && file.path().extension() == ".obj") {
      auto path = file.path();
      auto name = path.stem().string();
      if (name.find(nameInModelDir) == std::string::npos) continue;
      osg::ref_ptr<osgDB::Options> options = new osgDB::Options;
      options->setOptionString("DIFFUSE=0 SPECULAR=1 SPECULAR_EXPONENT=2 OPACITY=3");

      masterPanel = core::utils::osgUtils::readFileViaOSGDB(path.string(), options);
      if (!masterPanel) {
        std::cerr << "Error: Could not load solar panel model from " << path
                  << std::endl;
        continue;
      }
      break;
    }
  }
  return masterPanel;
}

std::unique_ptr<SolarPanel> CityGMLSystem::createSolarPanel(
    const std::string &name, osg::ref_ptr<osg::Group> parent,
    const std::vector<core::utils::osgUtils::instancing::GeometryData>
        &masterGeometryData,
    const osg::Matrix &matrix, const Color &colorIntensity) {
  using namespace core::utils::osgUtils;
  auto solarPanelInstance = instancing::createInstance(masterGeometryData, matrix);
  solarPanelInstance->setName(name);

  auto solarPanel = std::make_unique<SolarPanel>(solarPanelInstance);
  solarPanel->applyColor(colorIntensity);
  parent->addChild(solarPanelInstance);
  return std::move(solarPanel);
}

void CityGMLSystem::processSolarPanelDrawable(SolarPanelList &solarPanels,
                                              const SolarPanelConfig &config) {
  if (!config.valid()) {
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
  if (maxPanels == 0) maxPanels = 1;

  int numPanelsPerRow = static_cast<int>(std::sqrt(maxPanels));
  int numPanelRows = (maxPanels + numPanelsPerRow - 1) / numPanelsPerRow;

  float availableWidthForSpacingX =
      roofWidth - (numPanelsPerRow * config.panelWidth);
  float availableHeightForSpacingY =
      roofHeight - (numPanelRows * config.panelHeight);

  float spacingX =
      (numPanelsPerRow > 1)
          ? std::min(0.5f, availableWidthForSpacingX / (numPanelsPerRow - 1))
          : 0.0f;
  float spacingY =
      (numPanelRows > 1)
          ? std::min(0.5f, availableHeightForSpacingY / (numPanelRows - 1))
          : 0.0f;

  float totalWidthOfPanelsX =
      (numPanelsPerRow * config.panelWidth) + ((numPanelsPerRow - 1) * spacingX);
  float totalHeightOfPanelsY =
      (numPanelRows * config.panelHeight) + ((numPanelRows - 1) * spacingY);

  auto startX =
      roofCenter.x() - (totalWidthOfPanelsX / 2.0f) + (config.panelWidth / 2.0f);
  auto startY =
      roofCenter.y() - (totalHeightOfPanelsY / 2.0f) + (config.panelHeight / 2.0f);

  for (int i = 0; i < maxPanels; ++i) {
    int row = i / numPanelsPerRow;
    int col = i % numPanelsPerRow;

    auto x = startX + (col * (config.panelWidth + spacingX));
    auto y = startY + (row * (config.panelHeight + spacingY));

    auto position = osg::Vec3(x, y, z);
    osg::Matrix matrix = config.rotation * osg::Matrix::translate(position);
    auto solarPanel =
        createSolarPanel("SolarPanel_" + std::to_string(i), pvPanelsTransform,
                         config.masterGeometryData, matrix, config.colorIntensity);
    solarPanels.push_back(std::move(solarPanel));
  }

  config.parent->addChild(pvPanelsTransform);
}

void CityGMLSystem::processSolarPanelDrawables(
    const PVData &data, const std::vector<osg::ref_ptr<osg::Node>> drawables,
    SolarPanelList &solarPanels, SolarPanelConfig &config) {
  for (auto drawable : drawables) {
    const auto &name = drawable->getName();
    if (name.find("RoofSurface") == std::string::npos) {
      continue;
    }

    if (data.numPanelsMax == 0) continue;
    config.numMaxPanels = data.numPanelsMax;
    config.geode = drawable->asGeode();
    if (!config.geode) {
      std::cerr << "Error: Drawable is not a Geode." << std::endl;
      continue;
    }
    processSolarPanelDrawable(m_panels, config);
  }
}

void CityGMLSystem::processPVDataMap(
    const std::vector<core::utils::osgUtils::instancing::GeometryData>
        &masterGeometryData,
    const std::map<std::string, PVData> &pvDataMap, float maxPVIntensity) {
  using namespace core::utils::osgUtils;

  if (m_sensorMap.empty()) {
    std::cerr << "Error: No cityGML objects found." << std::endl;
    return;
  }

  m_pvGroup = new osg::Group();
  m_pvGroup->setName("PVPanels");

  osg::ref_ptr<osg::Group> gmlRoot = m_cityGMLGroup->getChild(0)->asGroup();
  if (gmlRoot) {
    gmlRoot->addChild(m_pvGroup);
  } else {
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
  config.parent = m_pvGroup;
  // panel is 1.7m x 1.0m x 0.4m
  config.panelWidth = 1.0f;
  config.panelHeight = 1.7f;
  config.zOffset = sin(osg::PI / 4) * config.panelHeight;

  for (const auto &[id, data] : pvDataMap) {
    try {
      auto &cityGMLObj = m_sensorMap.at(id);
      config.colorIntensity = core::utils::color::getTrafficLightColor(
          data.energyYearlyKWhMax / data.area, maxPVIntensity);
      processSolarPanelDrawables(data, cityGMLObj->getDrawables(), m_panels, config);

    } catch (const std::out_of_range &) {
      std::cerr << "Error: Could not find cityGML object with ID " << id
                << " in m_cityGMLObjs." << std::endl;
      continue;
    }
  }
}

void CityGMLSystem::initPV(osg::ref_ptr<osg::Node> masterPanel,
                           const std::map<std::string, PVData> &pvDataMap,
                           float maxPVIntensity) {
  using namespace core::utils::osgUtils;

  // for only textured geometry data
  auto masterGeometryData = instancing::extractAllGeometryData(masterPanel);
  if (masterGeometryData.empty()) {
    std::cerr << "Error: No geometry data found in the solar panel model."
              << std::endl;
    return;
  }

  processPVDataMap(masterGeometryData, pvDataMap, maxPVIntensity);
}

void CityGMLSystem::addSolarPanels(const fs::path &dirPath) {
  if (!fs::exists(dirPath)) return;

  fs::path pvDirPath(m_pvDir);
  if (!fs::exists(pvDirPath)) {
    std::cerr << "Error: PV directory does not exist: " << m_pvDir << std::endl;
    return;
  }

  auto pvStreams = getCSVStreams(pvDirPath);
  auto it = pvStreams.find("pv");
  if (it == pvStreams.end()) {
    std::cerr << "Error: Could not find PV data in " << m_pvDir << std::endl;
    return;
  }

  CSVStream &pvStream = it->second;
  if (!pvStream) {
    std::cerr << "Error: Could not load solar panel data from PV stream."
              << std::endl;
    return;
  }

  auto [pvDataMap, maxPVIntensity] = loadPVData(pvStream);

  auto masterPanel = readPVModel(dirPath, "solarpanel_1k_resized");

  if (!masterPanel) {
    std::cerr << "Error: Could not load solar panel model. Make sure to define the "
                 "correct 3DModelDir in EnergyCampus.toml."
              << std::endl;
    return;
  }
  initPV(masterPanel, pvDataMap, maxPVIntensity);
}

void CityGMLSystem::transform(const osg::Vec3 &translation,
                              const osg::Quat &rotation, const osg::Vec3 &scale) {
  assert(m_cityGMLGroup && "CityGML group is not initialized.");
  if (m_cityGMLGroup->getNumChildren() == 0) {
    std::cout << "No CityGML objects to transform." << std::endl;
    return;
  }
  for (unsigned int i = 0; i < m_cityGMLGroup->getNumChildren(); ++i) {
    osg::ref_ptr<osg::Node> child = m_cityGMLGroup->getChild(i);
    if (auto mt = dynamic_cast<osg::MatrixTransform *>(child.get())) {
      osg::Matrix matrix = osg::Matrix::translate(translation) *
                           osg::Matrix::rotate(rotation) * osg::Matrix::scale(scale);
      mt->setMatrix(matrix);
    } else {
      std::cerr << "Child is not a MatrixTransform." << std::endl;
    }
  }
}

auto CityGMLSystem::getTranslation() const {
  return m_cityGMLUI.getTranslation();
}

auto CityGMLSystem::readStaticCampusData(CSVStream &stream, float &max, float &min,
                                         float &sum) {
  std::vector<StaticPowerCampusData> yearlyValues;
  if (!stream || stream.getHeader().size() < 1) return yearlyValues;
  CSVStream::CSVRow row;
  while (stream.readNextRow(row)) {
    StaticPowerCampusData data;
    ACCESS_CSV_ROW(row, "gml_id", data.citygml_id);
    ACCESS_CSV_ROW(row, "energy_mwh", data.yearlyConsumption);

    max = std::max(max, data.yearlyConsumption);

    if (min == -1) {
      min = data.yearlyConsumption;
    }
    min = std::min(min, data.yearlyConsumption);

    yearlyValues.push_back(data);
  }
  return yearlyValues;
}

void CityGMLSystem::applyStaticDataCampusToCityGML(const std::string &filePath,
                                                   bool updateColorMap) {
  if (m_sensorMap.empty()) return;
  if (!fs::exists(filePath)) return;
  auto csvStream = CSVStream(filePath);
  float max = 0, min = -1;
  float sum = 0;

  auto values = readStaticCampusData(csvStream, max, min, sum);

  auto cb = m_cityGMLUI.colorBar();
  //   max = 400.00f;
  if (updateColorMap) {
    cb->setMinMax(min, max);
    cb->setSpecies("Yearly Consumption");
    cb->setUnit("MWh");
    auto halfSpan = (max - min) / 2;
    cb->setMinBounds(min - halfSpan, min + halfSpan);
    cb->setMaxBounds(max - halfSpan, max + halfSpan);
  }

  for (const auto &v : values) {
    if (auto it = m_sensorMap.find(v.citygml_id); it != m_sensorMap.end()) {
      auto &gmlObj = it->second;
      gmlObj->updateTimestepColors({v.yearlyConsumption}, cb->colorMap());
      gmlObj->updateTxtBoxTexts(
          {"Yearly Consumption: " + std::to_string(v.yearlyConsumption) + " MWh"});
    }
  }
  setAnimationTimesteps(1, m_cityGMLGroup);
}

void CityGMLSystem::enableCityGML(bool on, bool updateColorMap) {
  if (on) {
    if (m_sensorMap.empty()) {
      for (unsigned int i = 0; i < m_coverRootGroup->getNumChildren(); ++i) {
        osg::ref_ptr<osg::MatrixTransform> child =
            dynamic_cast<osg::MatrixTransform *>(m_coverRootGroup->getChild(i));
        if (child) {
          auto name = child->getName();
          if (name.find(".gml") != std::string::npos) {
            addCityGMLObjects(child);
            m_cityGMLGroup->addChild(child);
            auto translation = getTranslation();
            osg::Vec3 trans(translation.x, translation.y, translation.z);
            child->setMatrix(osg::Matrix::translate(trans));
            transform(trans, {});
          }
        }
      }
      core::utils::osgUtils::deleteChildrenFromOtherGroup(m_coverRootGroup,
                                                          m_cityGMLGroup);
    }

    // if (!m_cityGMLUI.colorBar())
    //     m_cityGMLUI.initColorBar();

    // TODO: this needs to be set in the root of system
    switchTo(m_cityGMLGroup, m_parent);

    // TODO: add a check if the group is already added and make sure its safe to
    if (m_cityGMLUI.getInfluxCSVBtnState())
      applyInfluxCSVToCityGML(m_influxPath, updateColorMap);

    if (m_cityGMLUI.getStaticCampusPowerBtnState())
      applyStaticDataCampusToCityGML(m_campusPath, updateColorMap);

    if (m_cityGMLUI.getStaticPowerBtnState())
      applyStaticDataToCityGML(m_staticPower, updateColorMap);

    if (m_panels.empty()) {
      auto solarPanelsDir = fs::path(m_modelDir + "/power/SolarPanel");
      addSolarPanels(solarPanelsDir);
    }
  }
}

auto CityGMLSystem::readStaticPowerData(CSVStream &stream, float &max, float &min,
                                        float &sum) {
  std::vector<StaticPowerData> powerData;
  if (!stream || stream.getHeader().size() < 1) return powerData;
  CSVStream::CSVRow row;
  while (stream.readNextRow(row)) {
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

    if (min == -1) {
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
                                             bool updateColorMap) {
  if (m_sensorMap.empty()) return;
  if (!fs::exists(filePathToInfluxCSV)) return;
  auto csvStream = CSVStream(filePathToInfluxCSV);
  float max = 0, min = -1;
  float sum = 0;

  auto values = readStaticPowerData(csvStream, max, min, sum);
  //   max = 7000.0f;
  auto cb = m_cityGMLUI.colorBar();
  if (updateColorMap) {
    cb->setMinMax(min, max);
    cb->setSpecies("Yearly Consumption");
    cb->setUnit("kWh");
    auto halfSpan = (max - min) / 2;
    cb->setMinBounds(min - halfSpan, min + halfSpan);
    cb->setMaxBounds(max - halfSpan, max + halfSpan);
  }
  for (const auto &v : values) {
    if (auto it = m_sensorMap.find(v.citygml_id); it != m_sensorMap.end()) {
      auto &gmlObj = it->second;
      gmlObj->updateTimestepColors({v.val2019, v.val2023, v.average},
                                   cb->colorMap());
      gmlObj->updateTxtBoxTexts({"2019: " + std::to_string(v.val2019) + " kWh",
                                 "2023: " + std::to_string(v.val2023) + " kWh",
                                 "Average: " + std::to_string(v.average) + " kWh"});
      gmlObj->updateTitleOfInfoboard(v.name);
    }
  }
  setAnimationTimesteps(3, m_cityGMLGroup);
}

void CityGMLSystem::applyInfluxCSVToCityGML(const std::string &filePathToInfluxCSV,
                                            bool updateColorMap) {
  if (m_sensorMap.empty()) return;
  if (!fs::exists(filePathToInfluxCSV)) return;
  auto csvStream = CSVStream(filePathToInfluxCSV);
  float max = 0, min = -1;
  float sum = 0;
  int timesteps = 0;
  auto values = getInfluxDataFromCSV(csvStream, max, min, sum, timesteps);
  auto cb = m_cityGMLUI.colorBar();

  if (updateColorMap) {
    auto distributionCenter = sum / (timesteps * values->size());
    cb->setMinMax(min, max);
    cb->setMinBounds(0, distributionCenter);
    cb->setMaxBounds(distributionCenter, max);
  }

  for (auto &[name, values] : *values) {
    auto sensorIt = m_sensorMap.find(name);
    if (sensorIt != m_sensorMap.end()) {
      sensorIt->second->updateTimestepColors(values, cb->colorMap());
      sensorIt->second->updateTxtBoxTexts({"NOT IMPLEMENTED YET"});
    }
  }
  setAnimationTimesteps(timesteps, m_cityGMLGroup);
}

std::unique_ptr<std::map<std::string, std::vector<float>>>
CityGMLSystem::getInfluxDataFromCSV(opencover::utils::read::CSVStream &stream,
                                    float &max, float &min, float &sum,
                                    int &timesteps) {
  using FloatMap = std::map<std::string, std::vector<float>>;
  const auto &headers = stream.getHeader();
  FloatMap values;
  if (stream && headers.size() > 1) {
    CSVStream::CSVRow row;
    // while (stream >> row) {
    while (stream.readNextRow(row)) {
      for (auto cityGMLBuildingName : headers) {
        auto sensor = m_sensorMap.find(cityGMLBuildingName);
        if (sensor == m_sensorMap.end()) continue;
        float value = 0;
        ACCESS_CSV_ROW(row, cityGMLBuildingName, value);
        if (value > max)
          max = value;
        else if (value < min || min == -1)
          min = value;
        sum += value;
        if (values.find(cityGMLBuildingName) == values.end())
          values.insert({cityGMLBuildingName, {value}});
        else
          values[cityGMLBuildingName].push_back(value);
      }
      ++timesteps;
    }
  }
  return std::make_unique<FloatMap>(values);
}

void CityGMLSystem::addCityGMLObject(const std::string &name,
                                     osg::ref_ptr<osg::Group> citygmlObjGroup) {
  if (!citygmlObjGroup->getNumChildren()) return;

  if (m_sensorMap.find(name) != m_sensorMap.end()) return;

  auto geodes = core::utils::osgUtils::getGeodes(citygmlObjGroup);
  if (geodes->empty()) return;

  // store default stateset
  saveCityGMLObjectDefaultStateSet(name, *geodes);

  auto boundingbox = core::utils::osgUtils::getBoundingBox(*geodes);
  auto infoboardPos = Pos(boundingbox.center().x(), boundingbox.center().y(), boundingbox.center().z());
  infoboardPos.z +=
      (boundingbox.zMax() - boundingbox.zMin()) / 2 + boundingbox.zMin();
  auto infoboard = std::make_unique<OsgTxtInfoboard>(
      infoboardPos, name, "DroidSans-Bold.ttf", 50, 50, 2.0f, 0.1, 2);
  auto building = std::make_unique<CityGMLBuilding>(*geodes);
  auto sensor = std::make_unique<CityGMLDeviceSensor>(
      citygmlObjGroup, std::move(infoboard), std::move(building));
  m_sensorMap.insert({name, std::move(sensor)});
}

void CityGMLSystem::addCityGMLObjects(osg::ref_ptr<osg::Group> citygmlGroup) {
  for (unsigned int i = 0; i < citygmlGroup->getNumChildren(); ++i) {
    osg::ref_ptr<osg::Group> child =
        dynamic_cast<osg::Group *>(citygmlGroup->getChild(i));
    if (child) {
      const auto &name = child->getName();

      // handle quad tree optimized scenegraph
      if (name == "GROUP" || name == "") {
        addCityGMLObjects(child);
        continue;
      }

      addCityGMLObject(name, child);
    }
  }
}

void CityGMLSystem::saveCityGMLObjectDefaultStateSet(const std::string &name,
                                                     const Geodes &citygmlGeodes) {
  Geodes geodesCopy(citygmlGeodes.size());
  for (auto i = 0; i < citygmlGeodes.size(); ++i) {
    auto geode = citygmlGeodes[i];
    geodesCopy[i] =
        dynamic_cast<osg::Geode *>(geode->clone(osg::CopyOp::DEEP_COPY_STATESETS));
  }
  m_defaultStatesets.insert({name, std::move(geodesCopy)});
}

void CityGMLSystem::restoreGeodesStatesets(CityGMLDeviceSensor &sensor,
                                           const std::string &name,
                                           const Geodes &citygmlGeodes) {
  if (m_defaultStatesets.find(name) == m_defaultStatesets.end()) return;

  if (citygmlGeodes.empty()) return;

  for (auto i = 0; i < citygmlGeodes.size(); ++i) {
    auto gmlDefault = citygmlGeodes[i];
    osg::ref_ptr<osg::Geode> toRestore = sensor.getDrawable(i)->asGeode();
    if (toRestore) {
      toRestore->setStateSet(gmlDefault->getStateSet());
    }
  }
}

void CityGMLSystem::restoreCityGMLDefaultStatesets() {
  for (auto &[name, sensor] : m_sensorMap) {
    osg::ref_ptr<osg::Group> sensorParent = sensor->getParent();
    if (!sensorParent) continue;

    restoreGeodesStatesets(*sensor, name, m_defaultStatesets[name]);
  }
  m_defaultStatesets.clear();
}
