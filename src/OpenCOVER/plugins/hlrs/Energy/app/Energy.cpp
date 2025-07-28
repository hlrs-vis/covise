/****************************************************************************\
 **                                                          (C)2024 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading building energy data        **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern, Marko Djuric                                       **
 **                                                                        **
 ** TODO:                                                                  **
 **  [ ] fetch lat lon from googlemaps                                     **
 **  [x] make REST lib independent from ennovatis general use              **
 **  [x] update via REST in background                                     **
 **                                                                        **
 ** History:                                                               **
 **  2024  v1                                                              **
 **  Marko Djuric 02.2024: add ennovatis client                            **
 **  Marko Djuric 10.2024: add citygml interface                           **
 **                                                                        **
 **  TODO: mapdrap for z coord of EnergyGrids                              **
 **                                                                        **
\****************************************************************************/

// NEEDS TO BE INCLUDED FIRST
// apache arrow
// colliding with qt signals => compiler sees this as parameter name => QT replaces
// signals with QT_SIGNALS which expands to public
// QT_ANNOTATE_ACCESS_SPECIFIER(qt_signal)
// By simply put the includes for arrow before qt
// includes. In our case before everything we can resolve the issue.
#include <lib/apache/arrow.h>
#include "Energy.h"
#include "lib/core/simulation/simulation.h"
#include "ui/historic/Device.h"
#include "ui/ennovatis/EnnovatisDevice.h"
#include "ui/ennovatis/EnnovatisDeviceSensor.h"
#include <build_options.h>
#include <config/CoviseConfig.h>
#include <util/string_util.h>
// COVER
#include <PluginUtil/colors/coColorMap.h>

#include <PluginUtil/coShaderUtil.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Group.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Slider.h>
#include <cover/ui/View.h>
#include <utils/read/csv/csv.h>
#include <utils/string/LevenshteinDistane.h>

// std
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <osg/Vec4>
#include <osgDB/Options>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <regex>

// OSG
#include <osg/Geometry>
#include <osg/Group>
#include <osg/LineWidth>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Sequence>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Switch>
#include <osg/Vec3>
#include <osg/Version>
#include <osg/ref_ptr>
#include <osgUtil/Optimizer>

// boost
#include <boost/filesystem.hpp>
#include <boost/filesystem/directory.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/tokenizer.hpp>

#include "OpenConfig/covconfig/array.h"

// presentation
#include <app/presentation/SolarPanel.h>
#include <app/presentation/CityGMLBuilding.h>
#include <app/presentation/EnergyGrid.h>
#include <app/presentation/PrototypeBuilding.h>
#include <app/presentation/TxtInfoboard.h>

// core
// #include <lib/core/simulation/grid.h>
#include <lib/core/utils/color.h>
#include <lib/core/simulation/heating.h>
#include <lib/core/utils/osgUtils.h>
#include <lib/core/constants.h>
#include <lib/core/simulation/object.h>
#include <lib/core/simulation/power.h>

// Ennovatis
#include <lib/ennovatis/building.h>
#include <lib/ennovatis/csv.h>
#include <lib/ennovatis/date.h>
#include <lib/ennovatis/rest.h>
#include <lib/ennovatis/sax.h>

#include <utils/thread/ConcurrentQueue.h>

using namespace opencover;
using namespace COVERUtils::read;
using namespace COVERUtils::string;
using namespace energy;

namespace fs = boost::filesystem;

namespace {

constexpr bool debug = build_options.debug_ennovatis;
constexpr bool skipRedundance = false;

const std::array<std::string, 13> skipInfluxTables{
    "timestamp",      "district", "hkw",           "new-buildings",
    "pv-penetration", "loc_emob", "n_emob",        "awz_scaling",
    "loc_ev",         "n_ev",     "new_buildings", "operation_mode",
    "pv_scaling"};

auto isSkippedInfluxTable(const std::string &name) {
  return std::any_of(skipInfluxTables.begin(), skipInfluxTables.end(),
                     [&](const auto &s) { return s == name; });
}

// regex for dd.mm.yyyy
const std::regex dateRgx(
    R"(((0[1-9])|([12][0-9])|(3[01]))\.((0[0-9])|(1[012]))\.((20[012]\d|19\d\d)|(1\d|2[0123])))");
ennovatis::rest_request_handler m_debug_worker;

template <typename T>
void printLoadingPercentDistribution(
    const ObjectContainer<T> &container, float min, float max, int numBins = 20,
    const std::string &species = "loading_percent") {
  static_assert(std::is_base_of_v<Object, T>,
                "T must be derived from core::simulation::Object");
  std::vector<int> histogram(numBins, 0);
  int total = 0;

  for (const auto &object : container) {
    auto it = object.second.getData().find(species);
    if (it == object.second.getData().end()) continue;
    const auto &data = it->second;
    for (double value : data) {
      int bin = static_cast<int>(numBins * (value - min) / (max - min + 1e-8));
      if (bin < 0) bin = 0;
      if (bin >= numBins) bin = numBins - 1;
      histogram[bin]++;
      total++;
    }
  }

  std::cout << "Distribution of " << species << " (" << total << " values):\n";
  for (int i = 0; i < numBins; ++i) {
    float binMin = min + i * (max - min) / numBins;
    float binMax = min + (i + 1) * (max - min) / numBins;
    std::cout << "[" << binMin << ", " << binMax << "): ";
    int stars =
        histogram[i] * 50 /
        (total > 0 ? *std::max_element(histogram.begin(), histogram.end()) : 1);
    for (int s = 0; s < stars; ++s) std::cout << "*";
    std::cout << " (" << histogram[i] << ")\n";
  }
}

// Compare two string numbers as integer using std::stoi
bool helper_cmpStrNo_as_int(const std::string &strtNo, const std::string &strtNo2) {
  try {
    int intStrtNo = std::stoi(strtNo), intStrtNo2 = std::stoi(strtNo2);
    auto validConversion =
        strtNo == std::to_string(intStrtNo) && strtNo2 == std::to_string(intStrtNo2);
    if (intStrtNo2 == intStrtNo && validConversion) return true;
  } catch (...) {
  }
  return false;
}

/**
 * @brief Compares two string street numbers in the format ("<streetname>
 * <streetnumber>").
 *
 * The function compares the street numbers of the two street names as string
 * and integer. If the street numbers are equal, the function returns true.
 * @param strtName The first string street name.
 * @param strtName2 The second string street name.
 * @return true if the street numbers are equal, otherwise false.
 */
bool cmpStrtNo(const std::string &strtName, const std::string &strtName2) {
  auto strtNo = strtName.substr(strtName.find_last_of(" ") + 1);
  auto strtNo2 = strtName2.substr(strtName2.find_last_of(" ") + 1);

  // compare in lower case str
  auto lower = [](unsigned char c) { return std::tolower(c); };
  std::transform(strtNo2.begin(), strtNo2.end(), strtNo2.begin(), lower);
  std::transform(strtNo.begin(), strtNo.end(), strtNo.begin(), lower);
  if (strtNo2 == strtNo) return true;

  // compare as integers
  return helper_cmpStrNo_as_int(strtNo, strtNo2);
};

float computeDistributionCenter(const std::vector<float> &values) {
  float sum = 0;
  for (auto &value : values) sum += value;
  return sum / values.size();
}
}  // namespace

/* #region GENERAL */
EnergyPlugin *EnergyPlugin::m_plugin = nullptr;

EnergyPlugin::EnergyPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME),
      ui::Owner("EnergyPlugin", cover->ui),
      m_offset(3),
      m_req(nullptr),
      m_ennovatis(new osg::Group()),
      m_switch(new osg::Switch()),
      m_grid(new osg::Switch()),
      m_sequenceList(new osg::Sequence()),
      m_Energy(new osg::MatrixTransform()),
      m_cityGML(new osg::Group()),
      m_energyGrids({
          //   EnergySimulation{"PowerGrid", "vm_pu", "V",
          //   EnergyGridType::PowerGrid},
          //   EnergySimulation{"HeatingGrid", "mass_flow", "kg/s",
          //                    EnergyGridType::HeatingGrid},
          //   EnergySimulation{"PowerGridSonder", "Leistung", "kWh",
          //                    EnergyGridType::PowerGridSonder},
          EnergySimulation{"PowerGrid", EnergyGridType::PowerGrid},
          //   EnergySimulation{"status_quo", EnergyGridType::PowerGrid},
          //   EnergySimulation{"future_ev", EnergyGridType::PowerGrid},
          //   EnergySimulation{"future_ev_pv", EnergyGridType::PowerGrid},
          EnergySimulation{"HeatingGrid", EnergyGridType::HeatingGrid}
          //   EnergySimulation{"PowerGridSonder", EnergyGridType::PowerGridSonder},
          // EnergyGrid{"CoolingGrid", "mass_flow", "kg/s", EnergyGrids::CoolingGrid,
          // Components::Kaelte},
      }) {
  // need to save the config on exit => will only be saved when COVER is closed
  // correctly via q or closing the window

  config()->setSaveOnExit(true);

  fprintf(stderr, "Starting Energy Plugin\n");
  m_plugin = this;

  m_buildings = std::make_unique<ennovatis::Buildings>();

  m_sequenceList->setName("DB");
  m_ennovatis->setName("Ennovatis");
  m_cityGML->setName("CityGML");

  m_Energy->setName("Energy");
  cover->getObjectsRoot()->addChild(m_Energy);

  m_switch->setName("Switch");
  m_switch->addChild(m_sequenceList);
  m_switch->addChild(m_ennovatis);
  m_switch->addChild(m_cityGML);

  m_grid->setName("EnergyGrids");

  m_Energy->addChild(m_switch);
  m_Energy->addChild(m_grid);

  GDALAllRegister();

  m_SDlist.clear();

  initUI();
  m_offset =
      configFloatArray("General", "offset", std::vector<double>{0, 0, 0})->value();
}

std::string EnergyPlugin::getScenarioName(Scenario scenario) {
  switch (scenario) {
    case Scenario::status_quo:
      return "status_quo";
    case Scenario::future_ev:
      return "future_ev";
    case Scenario::future_ev_pv:
      return "future_ev_pv";
    case Scenario::optimized_bigger_awz:
      return "optimized_bigger_awz";
    case Scenario::optimized:
      return "optimized";
    default:
      return "unknown_scenario";
  }
}

EnergyPlugin::~EnergyPlugin() {
  auto root = cover->getObjectsRoot();

  if (m_cityGML) {
    restoreCityGMLDefaultStatesets();
    for (unsigned int i = 0; i < m_cityGML->getNumChildren(); ++i) {
      auto child = m_cityGML->getChild(i);
      root->addChild(child);
    }
    CoreUtils::osgUtils::deleteChildrenFromOtherGroup(m_cityGML, root);
  }

  if (m_Energy) {
    root->removeChild(m_Energy.get());
  }

  config()->save();
  m_plugin = nullptr;
}

void EnergyPlugin::initUI() {
  m_EnergyTab = new ui::Menu("Energy_Campus", m_plugin);
  m_EnergyTab->setText("Energy Campus");

  initOverview();
  initHistoricUI();
  initEnnovatisUI();
  initCityGMLUI();
  initSimUI();
}

void EnergyPlugin::initOverview() {
  checkEnergyTab();
  m_controlPanel = new ui::Menu(m_EnergyTab, "Control");
  m_controlPanel->setText("Control_Panel");

  m_energySwitchControlButton = new ui::Button(m_controlPanel, "BuildingSwitch");
  m_energySwitchControlButton->setText("Buildings");
  m_energySwitchControlButton->setCallback([this](bool value) {
    if (value) {
      m_Energy->addChild(m_switch);
    } else {
      m_Energy->removeChild(m_switch);
    }
  });
  m_energySwitchControlButton->setState(true);

  m_gridControlButton = new ui::Button(m_controlPanel, "GridSwitch");
  m_gridControlButton->setText("Grid");
  m_gridControlButton->setCallback([this](bool value) {
    if (value) {
      m_Energy->addChild(m_grid);
    } else {
      m_Energy->removeChild(m_grid);
    }
  });
  m_gridControlButton->setState(true);
}

void EnergyPlugin::preFrame() {
  ColorBar::HudPosition hudPos;
  auto numHuds = 0;
  for (auto &energyGrid : m_energyGrids) {
    if (!energyGrid.scalarSelector) continue;
    const auto &selectedScalar = energyGrid.scalarSelector->selectedItem();
    auto &colorMapMenu = energyGrid.colorMapRegistry[selectedScalar];
    if (colorMapMenu.selector && colorMapMenu.selector->hudVisible()) {
      hudPos.setNumHuds(numHuds++);
      colorMapMenu.selector->setHudPosition(hudPos);
    }
    // auto &colorMapSelector = energyGrid.colorMapRegistry[selectedScalar];
    // if (colorMapSelector && colorMapSelector->hudVisible()) {
    //   hudPos.setNumHuds(numHuds++);
    //   colorMapSelector->setHudPosition(hudPos);
    // }
  }
}

bool EnergyPlugin::isActiv(osg::ref_ptr<osg::Switch> switchToCheck,
                           osg::ref_ptr<osg::Group> group) {
  if (!switchToCheck || !group) return false;
  const auto valueList = switchToCheck->getValueList();
  const auto idx = switchToCheck->getChildIndex(group);
  return valueList[idx];
}

std::pair<PJ *, PJ_COORD> EnergyPlugin::initProj() {
  ProjTrans pjTrans;
  pjTrans.projFrom = configString("General", "projFrom", "default")->value();
  pjTrans.projTo = configString("General", "projTo", "default")->value();
  auto P = proj_create_crs_to_crs(PJ_DEFAULT_CTX, pjTrans.projFrom.c_str(),
                                  pjTrans.projTo.c_str(), NULL);
  PJ_COORD coord;
  coord.lpzt.z = 0.0;
  coord.lpzt.t = HUGE_VAL;
  bool mapdrape = true;

  if (!P) {
    fprintf(stderr,
            "Energy Plugin: Ignore mapping. No valid projection was "
            "found between given proj string in "
            "config EnergyCampus.toml\n");
    mapdrape = false;
  }
  return std::make_pair(P, coord);
}

void EnergyPlugin::projTransLatLon(float &lat, float &lon) {
  auto [P, coord] = initProj();
  coord.lpzt.lam = lon;
  coord.lpzt.phi = lat;

  coord = proj_trans(P, PJ_FWD, coord);

  lon = coord.xy.x + m_offset[0];
  lat = coord.xy.y + m_offset[1];
}

bool EnergyPlugin::update() {
  for (auto s = m_SDlist.begin(); s != m_SDlist.end(); s++) {
    if (s->second.empty()) continue;
    for (auto timeElem : s->second) {
      if (timeElem) timeElem->update();
    }
  }

  if constexpr (debug) {
    auto result = m_debug_worker.getResult();
    if (result)
      for (auto &requ : *result) std::cout << "Response:\n" << requ << "\n";
  }

  for (auto &sensor : m_ennovatisDevicesSensors) sensor->update();

  for (auto &[name, sensor] : m_cityGMLObjs) sensor->update();

  for (auto &energyGrid : m_energyGrids) {
    if (!energyGrid.grid) continue;
    energyGrid.grid->update();
  }

  return false;
}

void EnergyPlugin::setTimestep(int t) {
  m_sequenceList->setValue(t);
  for (auto &sensor : m_ennovatisDevicesSensors) sensor->setTimestep(t);
  for (auto &[_, sensor] : m_cityGMLObjs) sensor->updateTime(t);

  // this is a workaround for the fact that the energy grids are added in the same
  // order as they appear in the the constructor

  auto &energyGrid = m_energyGrids[m_energygridBtnGroup->value()];
  if (energyGrid.grid) energyGrid.grid->updateTime(t);
}

void EnergyPlugin::switchTo(const osg::ref_ptr<osg::Node> child,
                            osg::ref_ptr<osg::Switch> parent) {
  parent->setAllChildrenOff();
  parent->setChildValue(child, true);
}

bool EnergyPlugin::init() {
  auto dbPath = configString("CSV", "filename", "default")->value();
  auto channelIdJSONPath = configString("Ennovatis", "jsonPath", "default")->value();
  // csv contains only updated buildings
  auto channelIdCSVPath = configString("Ennovatis", "csvPath", "default")->value();
  ProjTrans pjTrans;
  pjTrans.projFrom = configString("General", "projFrom", "default")->value();
  pjTrans.projTo = configString("General", "projTo", "default")->value();

  initRESTRequest();

  if constexpr (debug) {
    std::cout << "Load database: " << dbPath << std::endl;
    std::cout << "Load channelIDs: " << channelIdJSONPath << std::endl;
  }

  if (loadDB(dbPath, pjTrans))
    std::cout << "Database loaded in cache" << std::endl;
  else
    std::cout << "Database not loaded" << std::endl;

  if (loadChannelIDs(channelIdJSONPath, channelIdCSVPath))
    std::cout << "Ennovatis channelIDs loaded in cache" << std::endl;
  else
    std::cout << "Ennovatis channelIDs not loaded" << std::endl;

  auto noMatches = updateEnnovatisBuildings(m_SDlist);

  if constexpr (debug) {
    int i = 0;
    std::cout << "Matches between devices and buildings:" << std::endl;
    for (auto &[device, building] : m_devBuildMap)
      std::cout << ++i << ": Device: " << device->getInfo()->strasse
                << " -> Building: " << building->getName() << std::endl;

    std::cout << "No matches for the following buildings:" << std::endl;
    for (auto &building : *noMatches) std::cout << building->getName() << std::endl;
  }
  initEnnovatisDevices();
  switchTo(m_sequenceList, m_switch);
  initGrid();
  return true;
}

EnergyPlugin::CSVStreamMap EnergyPlugin::getCSVStreams(
    const boost::filesystem::path &dirPath) {
  auto csv_files = CSVStreamMap();
  for (auto &entry : fs::directory_iterator(dirPath)) {
    if (fs::is_regular_file(entry)) {
      if (entry.path().extension() == ".csv") {
        auto path = entry.path();
        csv_files.emplace(path.stem().string(), path.string());
      }
    }
  }
  return csv_files;
}

void EnergyPlugin::setAnimationTimesteps(size_t maxTimesteps, const void *who) {
  if (maxTimesteps > opencover::coVRAnimationManager::instance()->getNumTimesteps())
    opencover::coVRAnimationManager::instance()->setNumTimesteps(maxTimesteps, who);
}
/* #endregion */

/* #region CITYGML */
void EnergyPlugin::initCityGMLUI() {
  checkEnergyTab();
  m_cityGMLMenu = new ui::Menu(m_EnergyTab, "CityGML");
  m_cityGMLEnableInfluxCSV = new ui::Button(m_cityGMLMenu, "InfluxCSV");
  m_cityGMLEnableInfluxCSV->setCallback([this](bool on) {
    if (on) {
      m_staticPower->setState(false);
      m_staticCampusPower->setState(false);
      m_cityGMLEnableInfluxArrow->setState(false);
    }
    enableCityGML(on);
  });

  m_cityGMLEnableInfluxArrow = new ui::Button(m_cityGMLMenu, "InfluxArrow");
  m_cityGMLEnableInfluxArrow->setCallback([this](bool on) {
    if (on) {
      m_staticPower->setState(false);
      m_staticCampusPower->setState(false);
      m_cityGMLEnableInfluxCSV->setState(false);
    }
    enableCityGML(on);
  });
  m_PVEnable = new ui::Button(m_cityGMLMenu, "PV");
  m_PVEnable->setText("PV");
  m_PVEnable->setState(true);
  m_PVEnable->setCallback([this](bool on) {
    if (m_pvGroup == nullptr) {
      std::cerr << "Error: No PV group found. Please enable GML first." << std::endl;
      return;
    }
    // TODO: add a check if the group is already added and make sure its safe to
    // remove it
    osg::ref_ptr<osg::MatrixTransform> gmlRoot =
        dynamic_cast<osg::MatrixTransform *>(m_cityGML->getChild(0));
    if (gmlRoot->containsNode(m_pvGroup)) {
      gmlRoot->removeChild(m_pvGroup);
    } else {
      gmlRoot->addChild(m_pvGroup);
    }
  });

  m_staticPower = new ui::Button(m_cityGMLMenu, "Static");
  m_staticPower->setText("StaticPower");
  m_staticPower->setState(false);
  m_staticPower->setCallback([&](bool on) {
    if (on) {
      m_cityGMLEnableInfluxCSV->setState(false);
      m_cityGMLEnableInfluxArrow->setState(false);
      m_staticCampusPower->setState(false);
    }
    enableCityGML(on);
  });

  m_staticCampusPower = new ui::Button(m_cityGMLMenu, "StaticCampus");
  m_staticCampusPower->setText("StaticPowerCampus");
  m_staticCampusPower->setState(false);
  m_staticCampusPower->setCallback([&](bool on) {
    if (on) {
      m_cityGMLEnableInfluxCSV->setState(false);
      m_cityGMLEnableInfluxArrow->setState(false);
      m_staticPower->setState(false);
    }
    enableCityGML(on);
  });

  m_cityGMLX = new ui::EditField(m_cityGMLMenu, "X");
  m_cityGMLY = new ui::EditField(m_cityGMLMenu, "Y");
  m_cityGMLZ = new ui::EditField(m_cityGMLMenu, "Z");
  auto x = configFloat("CityGML", "X", 0.0);
  auto y = configFloat("CityGML", "Y", 0.0);
  auto Z = configFloat("CityGML", "Z", 0.0);
  m_cityGMLX->setValue(x->value());
  m_cityGMLY->setValue(y->value());
  m_cityGMLZ->setValue(Z->value());
  auto updateFunction = [this](auto &value) {
    if (!isActiv(m_switch, m_cityGML)) return;
    auto translation = getCityGMLTranslation();
    transformCityGML(translation, {});
  };
  m_cityGMLX->setCallback(updateFunction);
  m_cityGMLY->setCallback(updateFunction);
  m_cityGMLZ->setCallback(updateFunction);
}

void EnergyPlugin::initCityGMLColorMap() {
  auto menu = new ui::Menu(m_simulationMenu, "CityGml_grid");

  m_cityGmlColorMap = std::make_unique<opencover::CoverColorBar>(menu);
  m_cityGmlColorMap->setSpecies("Leistung");
  m_cityGmlColorMap->setUnit("kWh");
  m_cityGmlColorMap->setCallback([this](const opencover::ColorMap &cm) {
    if (isActiv(m_switch, m_cityGML)) {
      enableCityGML(false, false);
      enableCityGML(true, false);
    }
  });
  m_cityGmlColorMap->setName("CityGML");
}

void EnergyPlugin::processPVRow(const CSVStream::CSVRow &row,
                                std::map<std::string, PVData> &pvDataMap,
                                float &maxPVIntensity) {
  PVData pvData;
  ACCESS_CSV_ROW(row, "gml_id", pvData.cityGMLID);

  if (m_cityGMLObjs.find(pvData.cityGMLID) == m_cityGMLObjs.end()) {
    std::cerr << "Error: Could not find cityGML object with ID " << pvData.cityGMLID
              << std::endl;
    return;
  }

  ACCESS_CSV_ROW(row, "energy_yearly_kwh_max", pvData.energyYearlyKWhMax);
  ACCESS_CSV_ROW(row, "pv_area_qm", pvData.pvAreaQm);
  ACCESS_CSV_ROW(row, "area_qm", pvData.area);
  //   ACCESS_CSV_ROW(row, "co2savings", pvData.co2savings);
  ACCESS_CSV_ROW(row, "n_modules_max", pvData.numPanelsMax);

  if (pvData.pvAreaQm == 0) {
    std::cerr << "Error: pvAreaQm is 0 for cityGML object with ID "
              << pvData.cityGMLID << std::endl;
    return;
  }

  maxPVIntensity =
      //   std::max(pvData.energyYearlyKWhMax / pvData.pvAreaQm, maxPVIntensity);
      std::max(pvData.energyYearlyKWhMax / pvData.area, maxPVIntensity);
  pvDataMap.insert({pvData.cityGMLID, pvData});
}

std::pair<std::map<std::string, PVData>, float> EnergyPlugin::loadPVData(
    CSVStream &pvStream) {
  CSVStream::CSVRow row;
  std::map<std::string, PVData> pvDataMap;
  float maxPVIntensity = 0;

  //   while (pvStream >> row) {
  while (pvStream.readNextRow(row)) {
    processPVRow(row, pvDataMap, maxPVIntensity);
  }

  return {pvDataMap, maxPVIntensity};
}

osg::ref_ptr<osg::Node> EnergyPlugin::readPVModel(
    const fs::path &modelDir, const std::string &nameInModelDir) {
  osg::ref_ptr<osg::Node> masterPanel;
  for (auto &file : fs::directory_iterator(modelDir)) {
    if (fs::is_regular_file(file) && file.path().extension() == ".obj") {
      auto path = file.path();
      auto name = path.stem().string();
      if (name.find(nameInModelDir) == std::string::npos) continue;
      osg::ref_ptr<osgDB::Options> options = new osgDB::Options;
      options->setOptionString("DIFFUSE=0 SPECULAR=1 SPECULAR_EXPONENT=2 OPACITY=3");

      masterPanel = CoreUtils::osgUtils::readFileViaOSGDB(path.string(), options);
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

SolarPanel EnergyPlugin::createSolarPanel(
    const std::string &name, osg::ref_ptr<osg::Group> parent,
    const std::vector<CoreUtils::osgUtils::instancing::GeometryData>
        &masterGeometryData,
    const osg::Matrix &matrix, const osg::Vec4 &colorIntensity) {
  using namespace CoreUtils::osgUtils;
  auto solarPanelInstance = instancing::createInstance(masterGeometryData, matrix);
  solarPanelInstance->setName(name);

  SolarPanel solarPanel(solarPanelInstance);
  solarPanel.updateColor(colorIntensity);
  parent->addChild(solarPanelInstance);
  return solarPanel;
}

void EnergyPlugin::processSolarPanelDrawable(SolarPanelList &solarPanels,
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
    solarPanels.push_back(std::make_unique<SolarPanel>(solarPanel));
  }

  config.parent->addChild(pvPanelsTransform);
}

void EnergyPlugin::processSolarPanelDrawables(
    const PVData &data, const std::vector<osg::ref_ptr<osg::Node>> drawables,
    SolarPanelList &solarPanels, SolarPanelConfig &config) {
  for (auto drawable : drawables) {
    const auto &name = drawable->getName();
    if (name.find("RoofSurface") == std::string::npos) {
      continue;
    }

    // osg::ref_ptr<osg::Geode> geode = drawable->asGeode();
    // core::utils::color::overrideGeodeColor(geode, config.colorIntensity);

    if (data.numPanelsMax == 0) continue;
    config.numMaxPanels = data.numPanelsMax;
    config.geode = drawable->asGeode();
    if (!config.geode) {
      std::cerr << "Error: Drawable is not a Geode." << std::endl;
      continue;
    }
    processSolarPanelDrawable(m_solarPanels, config);
  }
}

void EnergyPlugin::processPVDataMap(
    const std::vector<CoreUtils::osgUtils::instancing::GeometryData>
        &masterGeometryData,
    const std::map<std::string, PVData> &pvDataMap, float maxPVIntensity) {
  using namespace CoreUtils::osgUtils;

  if (m_cityGMLObjs.empty()) {
    std::cerr << "Error: No cityGML objects found." << std::endl;
    return;
  }

  m_pvGroup = new osg::Group();
  m_pvGroup->setName("PVPanels");

  osg::ref_ptr<osg::Group> gmlRoot = m_cityGML->getChild(0)->asGroup();
  if (gmlRoot) {
    gmlRoot->addChild(m_pvGroup);
  } else {
    std::cerr << "Error: m_cityGML->getChild(0) is not a valid group." << std::endl;
    return;
  }

  m_solarPanels = SolarPanelList();
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
      auto &cityGMLObj = m_cityGMLObjs.at(id);
      config.colorIntensity = CoreUtils::color::getTrafficLightColor(
          data.energyYearlyKWhMax / data.area, maxPVIntensity);
      processSolarPanelDrawables(data, cityGMLObj->getDrawables(), m_solarPanels,
                                 config);

    } catch (const std::out_of_range &) {
      std::cerr << "Error: Could not find cityGML object with ID " << id
                << " in m_cityGMLObjs." << std::endl;
      continue;
    }
  }
}

void EnergyPlugin::initPV(osg::ref_ptr<osg::Node> masterPanel,
                          const std::map<std::string, PVData> &pvDataMap,
                          float maxPVIntensity) {
  using namespace CoreUtils::osgUtils;

  // for only textured geometry data
  auto masterGeometryData = instancing::extractAllGeometryData(masterPanel);
  if (masterGeometryData.empty()) {
    std::cerr << "Error: No geometry data found in the solar panel model."
              << std::endl;
    return;
  }

  processPVDataMap(masterGeometryData, pvDataMap, maxPVIntensity);
}

void EnergyPlugin::addSolarPanelsToCityGML(const fs::path &dirPath) {
  if (!fs::exists(dirPath)) return;

  auto pvDir = configString("Simulation", "pvDir", "default")->value();
  fs::path pvDirPath(pvDir);
  if (!fs::exists(pvDirPath)) {
    std::cerr << "Error: PV directory does not exist: " << pvDir << std::endl;
    return;
  }

  auto pvStreams = getCSVStreams(pvDirPath);
  auto it = pvStreams.find("pv");
  if (it == pvStreams.end()) {
    std::cerr << "Error: Could not find PV data in " << pvDir << std::endl;
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

void EnergyPlugin::transformCityGML(const osg::Vec3 &translation,
                                    const osg::Quat &rotation,
                                    const osg::Vec3 &scale) {
  assert(m_cityGML && "CityGML group is not initialized.");
  if (m_cityGML->getNumChildren() == 0) {
    std::cout << "No CityGML objects to transform." << std::endl;
    return;
  }
  for (unsigned int i = 0; i < m_cityGML->getNumChildren(); ++i) {
    osg::ref_ptr<osg::Node> child = m_cityGML->getChild(i);
    if (auto mt = dynamic_cast<osg::MatrixTransform *>(child.get())) {
      osg::Matrix matrix = osg::Matrix::translate(translation) *
                           osg::Matrix::rotate(rotation) * osg::Matrix::scale(scale);
      mt->setMatrix(matrix);
    } else {
      std::cerr << "Child is not a MatrixTransform." << std::endl;
    }
  }
}

osg::Vec3 EnergyPlugin::getCityGMLTranslation() const {
  return osg::Vec3(m_cityGMLX->number(), m_cityGMLY->number(), m_cityGMLZ->number());
}

auto EnergyPlugin::readStaticCampusData(CSVStream &stream, float &max, float &min,
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

void EnergyPlugin::applyStaticDataCampusToCityGML(const std::string &filePath,
                                                  bool updateColorMap) {
  if (m_cityGMLObjs.empty()) return;
  if (!fs::exists(filePath)) return;
  auto csvStream = CSVStream(filePath);
  float max = 0, min = -1;
  float sum = 0;

  auto values = readStaticCampusData(csvStream, max, min, sum);

  //   max = 400.00f;
  if (updateColorMap) {
    m_cityGmlColorMap->setMinMax(min, max);
    m_cityGmlColorMap->setSpecies("Yearly Consumption");
    m_cityGmlColorMap->setUnit("MWh");
    auto halfSpan = (max - min) / 2;
    m_cityGmlColorMap->setMinBounds(min - halfSpan, min + halfSpan);
    m_cityGmlColorMap->setMaxBounds(max - halfSpan, max + halfSpan);
  }

  for (const auto &v : values) {
    if (auto it = m_cityGMLObjs.find(v.citygml_id); it != m_cityGMLObjs.end()) {
      auto &gmlObj = it->second;
      gmlObj->updateTimestepColors({v.yearlyConsumption},
                                   m_cityGmlColorMap->colorMap());
      gmlObj->updateTxtBoxTexts(
          {"Yearly Consumption: " + std::to_string(v.yearlyConsumption) + " MWh"});
    }
  }
  setAnimationTimesteps(1, m_cityGML);
}

// void EnergyPlugin::enableCityGML(bool on) {
void EnergyPlugin::enableCityGML(bool on, bool updateColorMap) {
  if (on) {
    if (m_cityGMLObjs.empty()) {
      auto root = cover->getObjectsRoot();
      for (unsigned int i = 0; i < root->getNumChildren(); ++i) {
        osg::ref_ptr<osg::MatrixTransform> child =
            dynamic_cast<osg::MatrixTransform *>(root->getChild(i));
        if (child) {
          auto name = child->getName();
          if (name.find(".gml") != std::string::npos) {
            addCityGMLObjects(child);
            m_cityGML->addChild(child);
            auto translation = getCityGMLTranslation();
            child->setMatrix(osg::Matrix::translate(translation));
            transformCityGML(translation, {});
          }
        }
      }
      CoreUtils::osgUtils::deleteChildrenFromOtherGroup(root, m_cityGML);
    }
    if (!m_cityGmlColorMap) initCityGMLColorMap();
    switchTo(m_cityGML, m_switch);

    // TODO: add a check if the group is already added and make sure its safe to
    if (m_cityGMLEnableInfluxCSV->state()) {
      auto influxCSVPath =
          configString("Simulation", "staticInfluxCSV", "default")->value();
      applyInfluxCSVToCityGML(influxCSVPath, updateColorMap);
    }

    if (m_cityGMLEnableInfluxArrow->state()) {
    }

    if (m_staticCampusPower->state()) {
      auto campusPath = configString("Simulation", "campusPath", "default")->value();
      applyStaticDataCampusToCityGML(campusPath, updateColorMap);
    }

    if (m_staticPower->state()) {
      auto staticPower =
          configString("Simulation", "staticPower", "default")->value();
      applyStaticDataToCityGML(staticPower, updateColorMap);
    }

    if (m_solarPanels.empty()) {
      auto modelDirPath =
          configString("Simulation", "3dModelDir", "default")->value();
      auto solarPanelsDir = fs::path(modelDirPath + "/power/SolarPanel");

      addSolarPanelsToCityGML(solarPanelsDir);
    }

  } else {
    switchTo(m_sequenceList, m_switch);
  }
}

void EnergyPlugin::addCityGMLObject(const std::string &name,
                                    osg::ref_ptr<osg::Group> citygmlObjGroup) {
  if (!citygmlObjGroup->getNumChildren()) return;

  if (m_cityGMLObjs.find(name) != m_cityGMLObjs.end()) return;

  auto geodes = CoreUtils::osgUtils::getGeodes(citygmlObjGroup);
  if (geodes->empty()) return;

  // store default stateset
  saveCityGMLObjectDefaultStateSet(name, *geodes);

  auto boundingbox = CoreUtils::osgUtils::getBoundingBox(*geodes);
  auto infoboardPos = boundingbox.center();
  infoboardPos.z() +=
      (boundingbox.zMax() - boundingbox.zMin()) / 2 + boundingbox.zMin();
  auto infoboard = std::make_unique<TxtInfoboard>(
      infoboardPos, name, "DroidSans-Bold.ttf", 50, 50, 2.0f, 0.1, 2);
  auto building = std::make_unique<CityGMLBuilding>(*geodes);
  auto sensor = std::make_unique<CityGMLDeviceSensor>(
      citygmlObjGroup, std::move(infoboard), std::move(building));
  m_cityGMLObjs.insert({name, std::move(sensor)});
}

void EnergyPlugin::addCityGMLObjects(osg::ref_ptr<osg::Group> citygmlGroup) {
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

void EnergyPlugin::saveCityGMLObjectDefaultStateSet(const std::string &name,
                                                    const Geodes &citygmlGeodes) {
  Geodes geodesCopy(citygmlGeodes.size());
  for (auto i = 0; i < citygmlGeodes.size(); ++i) {
    auto geode = citygmlGeodes[i];
    geodesCopy[i] =
        dynamic_cast<osg::Geode *>(geode->clone(osg::CopyOp::DEEP_COPY_STATESETS));
  }
  m_cityGMLDefaultStatesets.insert({name, std::move(geodesCopy)});
}

void EnergyPlugin::restoreGeodesStatesets(CityGMLDeviceSensor &sensor,
                                          const std::string &name,
                                          const Geodes &citygmlGeodes) {
  if (m_cityGMLDefaultStatesets.find(name) == m_cityGMLDefaultStatesets.end())
    return;

  if (citygmlGeodes.empty()) return;

  for (auto i = 0; i < citygmlGeodes.size(); ++i) {
    auto gmlDefault = citygmlGeodes[i];
    osg::ref_ptr<osg::Geode> toRestore = sensor.getDrawable(i)->asGeode();
    if (toRestore) {
      toRestore->setStateSet(gmlDefault->getStateSet());
    }
  }
}

void EnergyPlugin::restoreCityGMLDefaultStatesets() {
  for (auto &[name, sensor] : m_cityGMLObjs) {
    osg::ref_ptr<osg::Group> sensorParent = sensor->getParent();
    if (!sensorParent) continue;

    restoreGeodesStatesets(*sensor, name, m_cityGMLDefaultStatesets[name]);
  }
  m_cityGMLDefaultStatesets.clear();
}
/* #endregion */

/* #region ENNOVATIS */
void EnergyPlugin::initEnnovatisUI() {
  checkEnergyTab();
  m_ennovatisMenu = new ui::Menu(m_EnergyTab, "Ennovatis");
  m_ennovatisMenu->setText("Ennovatis");

  m_ennovatisSelectionsList =
      new ui::SelectionList(m_ennovatisMenu, "Ennovatis_ChannelType");
  m_ennovatisSelectionsList->setText("Channel Type: ");
  std::vector<std::string> ennovatisSelections;
  for (int i = 0; i < static_cast<int>(ennovatis::ChannelGroup::None); ++i)
    ennovatisSelections.push_back(
        ennovatis::ChannelGroupToString(static_cast<ennovatis::ChannelGroup>(i)));

  m_ennovatisSelectionsList->setList(ennovatisSelections);
  m_enabledEnnovatisDevices =
      new opencover::ui::SelectionList(m_ennovatisMenu, "Enabled_Devices");
  m_enabledEnnovatisDevices->setText("Enabled Devices: ");
  m_enabledEnnovatisDevices->setCallback(
      [this](int value) { selectEnabledDevice(); });
  m_ennovatisChannelList =
      new opencover::ui::SelectionList(m_ennovatisMenu, "Channels");
  m_ennovatisChannelList->setText("Channels: ");

  // TODO: add calender widget instead of txtfields
  m_ennovatisFrom = new ui::EditField(m_ennovatisMenu, "from");
  m_ennovatisTo = new ui::EditField(m_ennovatisMenu, "to");

  m_ennovatisUpdate = new ui::Button(m_ennovatisMenu, "Update");
  m_ennovatisUpdate->setCallback([this](bool on) { updateEnnovatis(); });

  m_ennovatisSelectionsList->setCallback(
      [this](int value) { setEnnovatisChannelGrp(ennovatis::ChannelGroup(value)); });
  m_ennovatisFrom->setCallback(
      [this](const std::string &toSet) { setRESTDate(toSet, true); });
  m_ennovatisTo->setCallback(
      [this](const std::string &toSet) { setRESTDate(toSet, false); });
}

void EnergyPlugin::selectEnabledDevice() {
  auto selected = m_enabledEnnovatisDevices->selectedItem();
  for (auto &sensor : m_ennovatisDevicesSensors) {
    auto building = sensor->getDevice()->getBuildingInfo().building;
    if (building->getName() == selected) {
      sensor->disactivate();
      sensor->activate();
      return;
    }
  }
}

void EnergyPlugin::updateEnnovatis() { updateEnnovatisChannelGrp(); }

void EnergyPlugin::setRESTDate(const std::string &toSet, bool isFrom = false) {
  std::string fromOrTo = (isFrom) ? "From: " : "To: ";
  fromOrTo += toSet;
  if (!std::regex_match(toSet, dateRgx)) {
    std::cout << "Invalid date format for " << fromOrTo
              << " Please use the following format: " << ennovatis::date::dateformat
              << std::endl;
    return;
  }

  auto time = ennovatis::date::str_to_time_point(toSet, ennovatis::date::dateformat);
  bool validTime = (isFrom) ? (time <= m_req->dtt) : (time >= m_req->dtf);
  if (!validTime) {
    std::cout << "Invalid date. (To >= From)" << std::endl;
    if (isFrom)
      m_ennovatisFrom->setValue(ennovatis::date::time_point_to_str(
          m_req->dtf, ennovatis::date::dateformat));
    else
      m_ennovatisTo->setValue(ennovatis::date::time_point_to_str(
          m_req->dtt, ennovatis::date::dateformat));
    return;
  }

  if (isFrom)
    m_req->dtf = time;
  else
    m_req->dtt = time;
}

CylinderAttributes EnergyPlugin::getCylinderAttributes() {
  auto configDefaultColorVec = configFloatArray("Ennovatis", "defaultColorCylinder",
                                                std::vector<double>{0, 0, 0, 1.f})
                                   ->value();
  auto configMaxColorVec = configFloatArray("Ennovatis", "maxColorCylinder",
                                            std::vector<double>{0.0, 0.1, 0.0, 1.f})
                               ->value();
  auto configMinColorVec = configFloatArray("Ennovatis", "minColorCylinder",
                                            std::vector<double>{0.0, 1.0, 0.0, 1.f})
                               ->value();
  auto configDefaultHeightCycl =
      configFloat("Ennovatis", "defaultHeightCylinder", 100.0)->value();
  auto configRadiusCycl = configFloat("Ennovatis", "radiusCylinder", 3.0)->value();
  auto defaultColor = osg::Vec4(configDefaultColorVec[0], configDefaultColorVec[1],
                                configDefaultColorVec[2], configDefaultColorVec[3]);
  auto maxColor = osg::Vec4(configMaxColorVec[0], configMaxColorVec[1],
                            configMaxColorVec[2], configMaxColorVec[3]);
  auto minColor = osg::Vec4(configMinColorVec[0], configMinColorVec[1],
                            configMinColorVec[2], configMinColorVec[3]);
  return CylinderAttributes(configRadiusCycl, configDefaultHeightCycl, maxColor,
                            minColor, defaultColor);
}

void EnergyPlugin::initEnnovatisDevices() {
  m_ennovatis->removeChildren(0, m_ennovatis->getNumChildren());
  m_ennovatisDevicesSensors.clear();
  auto cylinderAttributes = getCylinderAttributes();
  for (auto &b : *m_buildings) {
    cylinderAttributes.position = osg::Vec3(b.getX(), b.getY(), b.getHeight());
    auto drawableBuilding = std::make_unique<PrototypeBuilding>(cylinderAttributes);
    auto infoboardPos = osg::Vec3(b.getX() + cylinderAttributes.radius + 5,
                                  b.getY() + cylinderAttributes.radius + 5,
                                  b.getHeight() + cylinderAttributes.height);
    auto infoboard = std::make_unique<TxtInfoboard>(
        infoboardPos, b.getName(), "DroidSans-Bold.ttf",
        cylinderAttributes.radius * 20, cylinderAttributes.radius * 21, 2.0f, 0.1,
        2);
    auto enDev = std::make_unique<EnnovatisDevice>(
        b, m_ennovatisChannelList, m_req, m_channelGrp, std::move(infoboard),
        std::move(drawableBuilding));
    m_ennovatis->addChild(enDev->getDeviceGroup());
    m_ennovatisDevicesSensors.push_back(std::make_unique<EnnovatisDeviceSensor>(
        std::move(enDev), enDev->getDeviceGroup(), m_enabledEnnovatisDevices));
  }
}

void EnergyPlugin::updateEnnovatisChannelGrp() {
  for (auto &sensor : m_ennovatisDevicesSensors)
    sensor->getDevice()->setChannelGroup(m_channelGrp);
}

void EnergyPlugin::setEnnovatisChannelGrp(ennovatis::ChannelGroup group) {
  switchTo(m_ennovatis, m_switch);
  m_channelGrp = std::make_shared<ennovatis::ChannelGroup>(group);

  if constexpr (debug) {
    auto &b = m_buildings->at(0);
    m_debug_worker.fetchChannels(group, b, *m_req);
  }
  updateEnnovatisChannelGrp();
}

bool EnergyPlugin::updateChannelIDsFromCSV(const std::string &pathToCSV) {
  auto csvPath = std::filesystem::path(pathToCSV);
  if (csvPath.extension() == ".csv") {
    std::ifstream csvFilestream(pathToCSV);
    if (!csvFilestream.is_open()) {
      std::cout << "File does not exist or cannot be opened: " << pathToCSV
                << std::endl;
      return false;
    }
    ennovatis::csv_channelid_parser csvParser;
    if (!csvParser.update_buildings_by_buildingid(pathToCSV, m_buildings))
      return false;
  }
  return true;
}

bool EnergyPlugin::loadChannelIDs(const std::string &pathToJSON,
                                  const std::string &pathToCSV) {
  std::ifstream inputFilestream(pathToJSON);
  if (!inputFilestream.is_open()) {
    std::cout << "File does not exist or cannot be opened: " << pathToJSON
              << std::endl;
    return false;
  }
  auto jsonPath = std::filesystem::path(pathToJSON);
  if (jsonPath.extension() == ".json") {
    ennovatis::sax_channelid_parser slp(m_buildings);
    if (!slp.parse_filestream(inputFilestream)) return false;

    if (!updateChannelIDsFromCSV(pathToCSV)) return false;

    if constexpr (debug)
      for (auto &log : slp.getDebugLogs()) std::cout << log << std::endl;
  }
  return true;
}

void EnergyPlugin::initRESTRequest() {
  m_req = std::make_unique<ennovatis::rest_request>();
  m_req->url = configString("Ennovatis", "restUrl", "default")->value();
  m_req->projEid = configString("Ennovatis", "projEid", "default")->value();
  m_req->channelId = "";
  m_req->dtf = std::chrono::system_clock::now() - std::chrono::hours(24);
  m_req->dtt = std::chrono::system_clock::now();
  m_ennovatisFrom->setValue(
      ennovatis::date::time_point_to_str(m_req->dtf, ennovatis::date::dateformat));
  m_ennovatisTo->setValue(
      ennovatis::date::time_point_to_str(m_req->dtt, ennovatis::date::dateformat));
}

std::unique_ptr<EnergyPlugin::const_buildings>
EnergyPlugin::updateEnnovatisBuildings(const DeviceList &deviceList) {
  auto lastDst = 0;
  auto noDeviceMatches = const_buildings();
  Device::ptr devicePickInteractor;
  auto fillLatLon = [&](ennovatis::Building &b) {
    b.setX(devicePickInteractor->getInfo()->lon);
    b.setY(devicePickInteractor->getInfo()->lat);
  };

  auto updateBuildingInfo = [&](ennovatis::Building &b, Device::ptr dev) {
    if (m_devBuildMap.find(dev) != m_devBuildMap.end()) return;
    m_devBuildMap[dev] = &b;
    // name in ennovatis is the street => first set it for street => then set
    // the name
    b.setStreet(b.getName());
    b.setName(dev->getInfo()->name);
    b.setHeight(dev->getInfo()->height);
    b.setArea(dev->getInfo()->flaeche);
    fillLatLon(b);
  };

  for (auto &building : *m_buildings) {
    lastDst = 100;
    devicePickInteractor = nullptr;
    const auto &ennovatis_strt = building.getName();
    for (const auto &[_, devices] : deviceList) {
      const auto &d = devices.front()->getDevice();
      const auto &device_strt = d->getInfo()->strasse;
      auto lvnstnDst = computeLevensteinDistance(ennovatis_strt, device_strt);

      // if the distance is 0, we have a perfect match
      if (!lvnstnDst) {
        lastDst = 0;
        devicePickInteractor = d;
        break;
      }

      // if the distance is less than the last distance, we have a better match
      if (lvnstnDst < lastDst) {
        lastDst = lvnstnDst;
        devicePickInteractor = d;
      }
      // if the distance is the same as the last distance, we have a better
      // match if the street number is the same
      else if (lvnstnDst == lastDst && cmpStrtNo(ennovatis_strt, device_strt)) {
        devicePickInteractor = d;
      }
    }
    if (!lastDst && devicePickInteractor) {
      updateBuildingInfo(building, devicePickInteractor);
      continue;
    }
    if (devicePickInteractor) {
      auto hit = m_devBuildMap.find(devicePickInteractor);
      if (hit == m_devBuildMap.end()) {
        updateBuildingInfo(building, devicePickInteractor);
      } else {
        noDeviceMatches.push_back(&building);
      }
    }
  }
  return std::make_unique<const_buildings>(noDeviceMatches);
}

/* #endregion */

/* #region HISTORIC */

void EnergyPlugin::initHistoricUI() {
  checkEnergyTab();
  componentGroup = new ui::ButtonGroup(m_EnergyTab, "ComponentGroup");
  componentGroup->setDefaultValue(Strom);
  componentList = new ui::Menu(m_EnergyTab, "Component");
  componentList->setText("Messwerte (jährlich)");
  StromBt = new ui::Button(componentList, "Strom", componentGroup, Strom);
  WaermeBt = new ui::Button(componentList, "Waerme", componentGroup, Waerme);
  KaelteBt = new ui::Button(componentList, "Kaelte", componentGroup, Kaelte);

  componentGroup->setCallback(
      [this](int value) { setComponent(Components(value)); });
}

void EnergyPlugin::reinitDevices(int comp) {
  for (auto s : m_SDlist) {
    if (s.second.empty()) continue;
    for (auto devSens : s.second) {
      auto t = devSens->getDevice();
      t->init(rad, scaleH, comp);
    }
  }
}

void EnergyPlugin::setComponent(Components c) {
  switchTo(m_sequenceList, m_switch);
  switch (c) {
    case Strom:
      StromBt->setState(true, false);
      break;
    case Waerme:
      WaermeBt->setState(true, false);
      break;
    case Kaelte:
      KaelteBt->setState(true, false);
      break;
    default:
      break;
  }

  m_selectedComp = c;
  reinitDevices(c);
}

bool EnergyPlugin::loadDB(const std::string &path, const ProjTrans &projTrans) {
  if (!loadDBFile(path, projTrans)) {
    return false;
  }

  setAnimationTimesteps(m_sequenceList->getNumChildren(), m_sequenceList);

  rad = 3.;
  scaleH = 0.1;

  reinitDevices(m_selectedComp);
  return true;
}

void EnergyPlugin::helper_initTimestepGrp(size_t maxTimesteps,
                                          osg::ref_ptr<osg::Group> &timestepGroup) {
  for (int t = 0; t < maxTimesteps; ++t) {
    timestepGroup = new osg::Group();
    std::string groupName = "timestep" + std::to_string(t);
    timestepGroup->setName(groupName);
    m_sequenceList->addChild(timestepGroup);
    m_sequenceList->setValue(t);
  }
}

void EnergyPlugin::helper_initTimestepsAndMinYear(
    size_t &maxTimesteps, int &minYear, const std::vector<std::string> &header) {
  for (const auto &h : header) {
    if (h.find("Strom") != std::string::npos) {
      auto minYearStr =
          std::regex_replace(h, std::regex("[^0-9]*"), std::string("$1"));
      int min_year_tmp = std::stoi(minYearStr);
      if (min_year_tmp < minYear) minYear = min_year_tmp;
      ++maxTimesteps;
    }
  }
}

void EnergyPlugin::helper_projTransformation(bool mapdrape, PJ *P, PJ_COORD &coord,
                                             DeviceInfo &deviceInfoPtr,
                                             const double &lat, const double &lon) {
  if (!mapdrape) {
    deviceInfoPtr.lon = lon;
    deviceInfoPtr.lat = lat;
    deviceInfoPtr.height = 0.f;
    return;
  }

  // x = lon, y = lat
  coord.lpzt.lam = lon;
  coord.lpzt.phi = lat;
  float alt = 0.;

  coord = proj_trans(P, PJ_FWD, coord);

  deviceInfoPtr.lon = coord.xy.x + m_offset[0];
  deviceInfoPtr.lat = coord.xy.y + m_offset[1];
  deviceInfoPtr.height = alt + m_offset[2];
}

void EnergyPlugin::helper_handleEnergyInfo(size_t maxTimesteps, int minYear,
                                           const CSVStream::CSVRow &row,
                                           DeviceInfo &deviceInfo) {
  auto font = configString("Billboard", "font", "default")->value();
  for (size_t year = minYear; year < minYear + maxTimesteps; ++year) {
    auto str_yr = std::to_string(year);
    auto strom = "Strom " + str_yr;
    auto waerme = "Wärme " + str_yr;
    auto kaelte = "Kälte " + str_yr;
    auto deviceInfoTimestep = std::make_shared<energy::DeviceInfo>(deviceInfo);
    float strom_val = 0.f;
    ACCESS_CSV_ROW(row, strom, strom_val);
    ACCESS_CSV_ROW(row, waerme, deviceInfoTimestep->waerme);
    ACCESS_CSV_ROW(row, kaelte, deviceInfoTimestep->kaelte);
    deviceInfoTimestep->strom = strom_val / 1000.;  // kW -> MW
    auto timestep = year - 2000;
    deviceInfoTimestep->timestep = timestep;
    auto device = std::make_shared<energy::Device>(
        deviceInfoTimestep, m_sequenceList->getChild(timestep)->asGroup(), font);

    m_SDlist[deviceInfo.ID].push_back(
        std::make_shared<energy::DeviceSensor>(device, device->getGroup()));
  }
}

bool EnergyPlugin::loadDBFile(const std::string &fileName,
                              const ProjTrans &projTrans) {
  try {
    auto csvStream = CSVStream(fileName);
    size_t maxTimesteps = 0;
    int minYear = 2000;
    const auto &header = csvStream.getHeader();
    helper_initTimestepsAndMinYear(maxTimesteps, minYear, header);

    CSVStream::CSVRow row;

    std::string sensorType;
    osg::ref_ptr<osg::Group> timestepGroup;
    bool mapdrape = true;

    auto P = proj_create_crs_to_crs(PJ_DEFAULT_CTX, projTrans.projFrom.c_str(),
                                    projTrans.projTo.c_str(), NULL);
    PJ_COORD coord;
    coord.lpzt.z = 0.0;
    coord.lpzt.t = HUGE_VAL;

    if (!P) {
      fprintf(stderr,
              "Energy Plugin: Ignore mapping. No valid projection was "
              "found between given proj string in "
              "config EnergyCampus.toml\n");
      mapdrape = false;
    }

    helper_initTimestepGrp(maxTimesteps, timestepGroup);

    // while (csvStream >> row) {
    while (csvStream.readNextRow(row)) {
      DeviceInfo deviceInfo;
      auto lat = std::stod(row["lat"]);
      auto lon = std::stod(row["lon"]);

      // location
      helper_projTransformation(mapdrape, P, coord, deviceInfo, lat, lon);

      deviceInfo.ID = row["GebäudeID"];
      deviceInfo.strasse = row["Straße"] + " " + row["Nr"];
      deviceInfo.name = row["Details"];
      ACCESS_CSV_ROW(row, "Baujahr", deviceInfo.baujahr);
      ACCESS_CSV_ROW(row, "Grundfläche (GIS)", deviceInfo.flaeche);

      // electricity, heat, cold
      helper_handleEnergyInfo(maxTimesteps, minYear, row, deviceInfo);
    }
    proj_destroy(P);
  } catch (const CSVStream_Exception &ex) {
    std::cout << ex.what() << std::endl;
    return false;
  }
  return true;
}

/* #endregion */

/* #region SIMULATION_DATA */

void EnergyPlugin::updateEnergyGridColorMapInShader(const opencover::ColorMap &map,
                                                    EnergyGridType type) {
  auto gridTypeIndex = getEnergyGridTypeIndex(type);
  auto &grid = m_energyGrids[gridTypeIndex];
  if (grid.group && isActiv(m_grid, grid.group) && grid.simUI) {
    // grid.simUI->updateTimestepColors(map);
    // grid.grid->setColorMap(map);
    // TODO: remove this later
    // HACK: this is a workaround
    grid.grid->setColorMap(map, m_vmPuColorMap->colorMap());
  }
}

void EnergyPlugin::initSimMenu() {
  checkEnergyTab();
  if (m_simulationMenu != nullptr) return;
  m_simulationMenu = new ui::Menu(m_EnergyTab, "Simulation");
  m_simulationMenu->setText("Simulation");

  m_liftGrids = new ui::Button(m_simulationMenu, "LiftGrids");
  constexpr float uplift(30.0f);
  m_liftGrids->setText("Up");
  m_liftGrids->setCallback([this, &uplift](bool on) {
    auto active = on ? 1 : -1;
    for (auto &energyGrid : m_energyGrids) {
      energyGrid.group->setMatrix(osg::Matrix::translate(
          0, 0, energyGrid.group->getMatrix().getTrans().z() + uplift * active));
    }
  });

  m_scenarios = new opencover::ui::ButtonGroup(m_simulationMenu, "Scenarios");
  m_scenarios->setDefaultValue(getScenarioIndex(Scenario::status_quo));

  m_status_quo = new ui::Button(m_simulationMenu, "status_quo", m_scenarios,
                                getScenarioIndex(Scenario::status_quo));
  m_future_ev = new ui::Button(m_simulationMenu, "future_ev", m_scenarios,
                               getScenarioIndex(Scenario::future_ev));
  m_future_ev_pv = new ui::Button(m_simulationMenu, "future_ev_pv", m_scenarios,
                                  getScenarioIndex(Scenario::future_ev_pv));
  m_optimized = new ui::Button(m_simulationMenu, "optimized", m_scenarios,
                               getScenarioIndex(Scenario::optimized));
  m_optimized_bigger_awz =
      new ui::Button(m_simulationMenu, "optimized_bigger_awz", m_scenarios,
                     getScenarioIndex(Scenario::optimized_bigger_awz));

  m_scenarios->setCallback([this](int value) {
    switchTo(m_energyGrids[getEnergyGridTypeIndex(EnergyGridType::PowerGrid)].group,
             m_grid);
    // auto scenarioIndex = getScenarioIndex(Scenario(value));
    std::string scenarioName = getScenarioName(Scenario(value));
    auto simPath = configString("Simulation", "powerSimDir", "default")->value();
    simPath += "/" + scenarioName;

    applySimulationDataToPowerGrid(simPath);
    auto &energyGrid =
        m_energyGrids[getEnergyGridTypeIndex(EnergyGridType::PowerGrid)];
    auto scalarSelector = energyGrid.scalarSelector;
    auto selectedScalar = scalarSelector->selectedItem();
    auto &colorMapMenu = energyGrid.colorMapRegistry[selectedScalar];
    // auto min = energyGrid.simUI->min(selectedScalar);
    // auto max = energyGrid.simUI->max(selectedScalar);
    // colorMapMenu.selector->setMinMax(min, max);
    colorMapMenu.selector->setMinMax(0.0f, 100.0f);

    updateEnergyGridColorMapInShader(colorMapMenu.selector->colorMap(),
                                     energyGrid.type);
    updateEnergyGridShaderData(energyGrid);
  });
}

void EnergyPlugin::switchEnergyGrid(EnergyGridType grid) {
  auto gridTypeIndex = getEnergyGridTypeIndex(grid);
  osg::ref_ptr<osg::Group> switch_to = m_energyGrids[gridTypeIndex].group;
  if (!switch_to) {
    std::cerr << "Cooling grid not implemented yet" << std::endl;
    return;
  }

  bool showHud = false;
  for (auto &energyGrid : m_energyGrids) {
    if (!energyGrid.sim || !energyGrid.simUI ||
        energyGrid.colorMapRegistry.empty() || !energyGrid.scalarSelector)
      continue;
    const auto &selected = energyGrid.scalarSelector->selectedItem();
    auto &colorMapMenu = energyGrid.colorMapRegistry[selected];
    if (energyGrid.type != grid) {
      auto &selector = colorMapMenu.selector;
      auto &menu = colorMapMenu.menu;
      showHud |= selector->hudVisible();
      selector->show(false);
      menu->setVisible(false);
    } else {
      auto menu = colorMapMenu.menu;
      menu->setVisible(true);
    }
  }
  auto &defaultGrid = m_energyGrids[gridTypeIndex];
  if (defaultGrid.scalarSelector) {
    const auto &selected = defaultGrid.scalarSelector->selectedItem();
    auto &colorMapMenu = defaultGrid.colorMapRegistry[selected];
    colorMapMenu.selector->show(showHud);
  }
  switchTo(switch_to, m_grid);
}

void EnergyPlugin::initEnergyGridUI() {
  if (m_simulationMenu == nullptr) {
    initSimMenu();
  }

  m_energygridGroup = new ui::Group(m_simulationMenu, "EnergyGrid");
  m_energygridBtnGroup = new ui::ButtonGroup(m_energygridGroup, "EnergyGrid");
  m_energygridBtnGroup->setCallback(
      [this](int value) { switchEnergyGrid(EnergyGridType(value)); });

  for (auto &energyGrid : m_energyGrids) {
    auto idx = getEnergyGridTypeIndex(energyGrid.type);
    energyGrid.simulationUIBtn =
        new ui::Button(m_simulationMenu, energyGrid.name, m_energygridBtnGroup, idx);
  }
  auto idx = getEnergyGridTypeIndex(EnergyGridType::HeatingGrid);
  m_energygridBtnGroup->setActiveButton(m_energyGrids[idx].simulationUIBtn);
}

void EnergyPlugin::initSimUI() {
  initSimMenu();
  initEnergyGridUI();
}

void EnergyPlugin::initPowerGridStreams() {
  auto powerGridDir = configString("Simulation", "powerGridDir", "default")->value();
  fs::path dir_path(powerGridDir);
  if (!fs::exists(dir_path)) return;
  m_powerGridStreams = getCSVStreams(dir_path);
  if (m_powerGridStreams.empty()) {
    std::cout << "No csv files found in " << powerGridDir << std::endl;
    return;
  }
}

// [X] - add a button to enable/disable the simulation data
// [ ] - plan uniform grid structure file => csv file in specific format
std::unique_ptr<EnergyPlugin::FloatMap> EnergyPlugin::getInlfuxDataFromCSV(
    COVERUtils::read::CSVStream &stream, float &max, float &min, float &sum,
    int &timesteps) {
  const auto &headers = stream.getHeader();
  FloatMap values;
  if (stream && headers.size() > 1) {
    CSVStream::CSVRow row;
    // while (stream >> row) {
    while (stream.readNextRow(row)) {
      for (auto cityGMLBuildingName : headers) {
        auto sensor = m_cityGMLObjs.find(cityGMLBuildingName);
        if (sensor == m_cityGMLObjs.end()) continue;
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

auto EnergyPlugin::readStaticPowerData(CSVStream &stream, float &max, float &min,
                                       float &sum) {
  std::vector<StaticPowerData> powerData;
  if (!stream || stream.getHeader().size() < 1) return powerData;
  CSVStream::CSVRow row;
  //   while (stream >> row) {
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

void EnergyPlugin::applyInfluxCSVToCityGML(const std::string &filePathToInfluxCSV,
                                           bool updateColorMap) {
  if (m_cityGMLObjs.empty()) return;
  if (!fs::exists(filePathToInfluxCSV)) return;
  auto csvStream = CSVStream(filePathToInfluxCSV);
  float max = 0, min = -1;
  float sum = 0;
  int timesteps = 0;
  auto values = getInlfuxDataFromCSV(csvStream, max, min, sum, timesteps);

  if (updateColorMap) {
    auto distributionCenter = sum / (timesteps * values->size());
    m_cityGmlColorMap->setMinMax(min, max);
    m_cityGmlColorMap->setMinBounds(0, distributionCenter);
    m_cityGmlColorMap->setMaxBounds(distributionCenter, max);
  }

  for (auto &[name, values] : *values) {
    auto sensorIt = m_cityGMLObjs.find(name);
    if (sensorIt != m_cityGMLObjs.end()) {
      sensorIt->second->updateTimestepColors(values, m_cityGmlColorMap->colorMap());
      sensorIt->second->updateTxtBoxTexts({"NOT IMPLEMENTED YET"});
    }
  }
  setAnimationTimesteps(timesteps, m_cityGML);
}

void EnergyPlugin::applyInfluxArrowToCityGML() {
  if (m_cityGMLObjs.empty()) return;
}

void EnergyPlugin::applyStaticDataToCityGML(const std::string &filePathToInfluxCSV,
                                            bool updateColorMap) {
  if (m_cityGMLObjs.empty()) return;
  if (!fs::exists(filePathToInfluxCSV)) return;
  auto csvStream = CSVStream(filePathToInfluxCSV);
  float max = 0, min = -1;
  float sum = 0;

  auto values = readStaticPowerData(csvStream, max, min, sum);
  //   max = 7000.0f;
  if (updateColorMap) {
    m_cityGmlColorMap->setMinMax(min, max);
    m_cityGmlColorMap->setSpecies("Yearly Consumption");
    m_cityGmlColorMap->setUnit("kWh");
    auto halfSpan = (max - min) / 2;
    m_cityGmlColorMap->setMinBounds(min - halfSpan, min + halfSpan);
    m_cityGmlColorMap->setMaxBounds(max - halfSpan, max + halfSpan);
  }
  for (const auto &v : values) {
    if (auto it = m_cityGMLObjs.find(v.citygml_id); it != m_cityGMLObjs.end()) {
      auto &gmlObj = it->second;
      gmlObj->updateTimestepColors({v.val2019, v.val2023, v.average},
                                   m_cityGmlColorMap->colorMap());
      gmlObj->updateTxtBoxTexts({"2019: " + std::to_string(v.val2019) + " kWh",
                                 "2023: " + std::to_string(v.val2023) + " kWh",
                                 "Average: " + std::to_string(v.average) + " kWh"});
      gmlObj->updateTitleOfInfoboard(v.name);
    }
  }
  setAnimationTimesteps(3, m_cityGML);
}

bool EnergyPlugin::checkBoxSelection_powergrid(const std::string &tableName,
                                               const std::string &paramName) {
  using namespace std::placeholders;
  auto eq_name = [](const std::string &compare,
                    const auto &pointerWithNameFunction) {
    return compare == pointerWithNameFunction->name();
  };
  auto eq_tableName = [&tableName, &eq_name](const auto &pair) {
    auto menu = pair.first;
    return eq_name(tableName, menu);
  };
  if (auto it = std::find_if(m_powerGridCheckboxes.begin(),
                             m_powerGridCheckboxes.end(), eq_tableName);
      it != m_powerGridCheckboxes.end()) {
    const auto &checkBoxes = it->second;
    if (auto it = std::find_if(checkBoxes.begin(), checkBoxes.end(),
                               std::bind(eq_name, paramName, _1));
        it != checkBoxes.end()) {
      return (*it)->state();
    }
  }
  return false;
}

void EnergyPlugin::rebuildPowerGrid() {
  auto idx = getEnergyGridTypeIndex(EnergyGridType::PowerGrid);
  //   auto idxSonder = getEnergyGridTypeIndex(EnergyGridType::PowerGridSonder);
  m_grid->removeChild(m_energyGrids[idx].group);
  //   m_grid->removeChild(m_energyGrids[idxSonder].group);
  initPowerGridStreams();
  buildPowerGrid();
}

void EnergyPlugin::updatePowerGridConfig(const std::string &tableName,
                                         const std::string &name, bool on) {
  int idx = 0;
  for (auto &[menuName, checkBoxes] : m_powerGridCheckboxes) {
    if (menuName->name() != tableName) {
      idx += checkBoxes.size();
      continue;
    }
    for (auto &checkBox : checkBoxes) {
      if (checkBox->name() == name) {
        (*m_powerGridSelectionPtr)[idx] = on;
        return;
      }
      ++idx;
    }
  }
}

void EnergyPlugin::updatePowerGridSelection(bool on) {
  if (!on) return;
  m_updatePowerGridSelection->setState(false);
  rebuildPowerGrid();
}

void EnergyPlugin::initPowerGridUI(const std::vector<std::string> &tablesToSkip) {
  if (m_powerGridStreams.empty()) initPowerGridStreams();
  m_powerGridMenu = new opencover::ui::Menu("PowerGridData", m_EnergyTab);

  m_updatePowerGridSelection = new opencover::ui::Button(m_powerGridMenu, "Update");
  m_updatePowerGridSelection->setState(false);
  m_updatePowerGridSelection->setCallback([this](bool enable) {
    updatePowerGridSelection(enable);
    auto idx = getEnergyGridTypeIndex(EnergyGridType::PowerGrid);
    switchTo(m_energyGrids[idx].group, m_grid);
  });

  m_powerGridSelectionPtr =
      configBoolArray("Simulation", "powerGridDataSelection", std::vector<bool>{});
  auto powerGridSelection = m_powerGridSelectionPtr->value();
  auto initConfig = powerGridSelection.empty();

  int i = 0;
  for (auto &[name, stream] : m_powerGridStreams) {
    if (std::any_of(tablesToSkip.begin(), tablesToSkip.end(),
                    [n = name](const std::string &table) { return table == n; }))
      continue;

    auto menu = new opencover::ui::Menu(m_powerGridMenu, name);
    menu->allowRelayout(true);

    auto header = stream.getHeader();
    std::vector<opencover::ui::Button *> checkBoxMap;
    for (auto &col : header) {
      if (i >= powerGridSelection.size()) break;
      if (col.find("Unnamed") == 0) continue;
      auto checkBox = new opencover::ui::Button(menu, col);
      checkBox->setCallback([this, tableName = name, col](bool on) {
        updatePowerGridConfig(tableName, col, on);
      });
      if (initConfig) {
        checkBox->setState(true);
        powerGridSelection.push_back(true);
      } else {
        checkBox->setState(powerGridSelection[i]);
      }
      checkBoxMap.push_back(checkBox);
      ++i;
    }
    if (auto it = m_powerGridCheckboxes.find(menu);
        it != m_powerGridCheckboxes.end()) {
      auto &[_, chBxMap] = *it;
      std::move(checkBoxMap.begin(), checkBoxMap.end(), std::back_inserter(chBxMap));
    } else {
      m_powerGridCheckboxes.emplace(menu, checkBoxMap);
    }
  }
  if (initConfig) {
    m_powerGridSelectionPtr->resize(powerGridSelection.size());
    for (auto j = 0; j < powerGridSelection.size(); ++j)
      (*m_powerGridSelectionPtr)[j] = powerGridSelection[j];
  }
}

void EnergyPlugin::applySimulationDataToPowerGrid(const std::string &simPath) {
  if (simPath.empty()) {
    std::cerr << "No simulation data path configured." << std::endl;
    return;
  }

  std::map<std::string, std::string> arrowFiles;
  for (auto &entry : fs::directory_iterator(simPath)) {
    if (fs::is_regular_file(entry) && entry.path().extension() == ".arrow") {
      arrowFiles.emplace(entry.path().stem().string(), entry.path().string());
    }
  }

  if (arrowFiles.empty()) {
    std::cerr << "No .arrow files found in the simulation data path." << std::endl;
    return;
  }

  auto vm_pu = arrowFiles["electrical_grid.res_bus.vm_pu"];
  auto loading_percent = arrowFiles["electrical_grid.res_line.loading_percent"];
  auto res_mw = arrowFiles["electrical_prosumer.res_mw"];

  apache::ArrowReader loadingPercentReader(loading_percent);
  apache::ArrowReader vmPuReader(vm_pu);
  apache::ArrowReader resMWReader(res_mw);

  auto tableLoadingPercent = loadingPercentReader.getTable();
  auto tableVmPu = vmPuReader.getTable();
  auto tableResMW = resMWReader.getTable();

  auto sim = std::make_shared<power::PowerSimulation>();
  auto &cables = sim->Cables();
  auto &buses = sim->Buses();
  auto &buildings = sim->Buildings();

  // Helper to process columns
  auto processColumns = [&](const std::shared_ptr<arrow::Table> &tbl,
                            auto &container, const std::string &dataKey) {
    auto columnNames = tbl->schema()->fields();
    for (int j = 0; j < tbl->num_columns(); ++j) {
      auto columnName = columnNames[j]->name();
      std::replace(columnName.begin(), columnName.end(), ' ', '_');
      std::replace(columnName.begin(), columnName.end(), '/', '-');
      if (isSkippedInfluxTable(columnName)) continue;
      auto column = tbl->column(j);
      int64_t chunk_offset = 0;
      for (int i = 0; i < column->num_chunks(); ++i) {
        auto chunk = column->chunk(i);
        if (chunk->type_id() == arrow::Type::DOUBLE) {
          auto darr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
          auto rawValues = darr->raw_values();
          if (container.find(columnName) == container.end()) {
            container.add(columnName);
            auto &data = container[columnName].getData();
            data[dataKey] = {};
            data[dataKey].resize(column->length());
          }
          auto &vec = container[columnName].getData()[dataKey];
          std::copy(rawValues, rawValues + darr->length(),
                    vec.begin() + chunk_offset);
          chunk_offset += darr->length();
        }
      }
    }
  };

  // Process bus voltages
  processColumns(tableVmPu, buses, "vm_pu");

  // Process cable loading
  processColumns(tableLoadingPercent, cables, "loading_percent");

  // Process residual load in MW
  processColumns(tableResMW, buildings, "res_mw");

  //   printLoadingPercentDistribution(cables, min, max);

  auto idx = getEnergyGridTypeIndex(EnergyGridType::PowerGrid);
  if (m_energyGrids[idx].grid == nullptr) return;
  auto &powerGrid = m_energyGrids[idx];
  powerGrid.simUI = std::make_unique<PowerSimUI>(sim, powerGrid.grid);
  powerGrid.sim = std::move(sim);

  if (m_cityGMLEnableInfluxArrow->state()) {
    const auto &[min, max] = powerGrid.sim->getMinMax("res_mw");
    m_cityGmlColorMap->setMinMax(min, max);
    m_cityGmlColorMap->setSpecies("Residuallast");
    m_cityGmlColorMap->setUnit("MW");
    auto halfSpan = (max - min) / 2;
    m_cityGmlColorMap->setMinBounds(min - halfSpan, min + halfSpan);
    m_cityGmlColorMap->setMaxBounds(max - halfSpan, max + halfSpan);
    printLoadingPercentDistribution(buildings, min, max);

    for (auto &[name, sensor] : m_cityGMLObjs) {
      std::string sensorName = name;
      auto values = powerGrid.sim->getTimedependentScalar("res_mw", sensorName);
      if (!values) {
        std::cerr << "No res_mw data found for sensor: " << sensorName << std::endl;
        continue;
      }

      auto steps = m_cityGmlColorMap->colorMap().steps();
      auto colorMapName = powerGrid.sim->getPreferredColorMap("res_mw");
      if (colorMapName == core::simulation::INVALID_UNIT) {
        colorMapName = m_cityGmlColorMap->colorMap().name();
      }
      m_cityGmlColorMap->setColorMap(colorMapName);
      m_cityGmlColorMap->setSteps(steps);
      sensor->setColorMapInShader(m_cityGmlColorMap->colorMap());
      sensor->setDataInShader(*values, min, max);

      std::vector<std::string> texts;
      std::transform(values->begin(), values->end(), std::back_inserter(texts),
                     [](const auto &v) { return std::to_string(v) + " MW"; });
      sensor->updateTxtBoxTexts(texts);
    }
  }

  // TODO: remove this later
  // HACK: this is a workaround
  if (!m_vmPuColorMap && m_simulationMenu) {
    auto menu = new opencover::ui::Menu(m_simulationMenu, "VmPuColorMap");
    m_vmPuColorMap = std::make_unique<opencover::CoverColorBar>(menu);
    auto tmp_steps = m_vmPuColorMap->colorMap().steps();
    m_vmPuColorMap->setColorMap("Voltage");
    m_vmPuColorMap->setSpecies("Voltage (VmPu)");
    m_vmPuColorMap->setUnit("");
    m_vmPuColorMap->setSteps(tmp_steps);
    m_vmPuColorMap->setCallback([this](const opencover::ColorMap &cm) {

    });
    m_vmPuColorMap->setName("VmPu");
  }
  if (m_vmPuColorMap) {
    const auto &[min, max] = powerGrid.sim->getMinMax("vm_pu");
    m_vmPuColorMap->setMinMax(min, max);
  }

  std::cout << "Number of timesteps: " << tableVmPu->num_rows() << std::endl;
  setAnimationTimesteps(tableVmPu->num_rows(), powerGrid.group);
}

void EnergyPlugin::initPowerGrid() {
  initPowerGridStreams();
  initPowerGridUI({"trafo3w_stdtypes", "trafo_std_types", "trafo", "parameters",
                   "dtypes", "bus_geodata", "fuse_std_types", "line_std_types"});
  buildPowerGrid();
  m_powerGridStreams.clear();
  auto simPath = configString("Simulation", "powerSimDir", "default")->value();
  simPath += "/status_quo";
  applySimulationDataToPowerGrid(simPath);
}

void EnergyPlugin::initEnergyGridColorMaps() {
  if (m_simulationMenu == nullptr) {
    initSimMenu();
  }

  for (auto &energyGrid : m_energyGrids) {
    if (!energyGrid.sim) {
      std::cerr << "Simulation for energygrid " << energyGrid.name
                << " not initialized before calling function initEnergyGridColorMaps"
                << std::endl;
      continue;
    }

    const auto &scalarProperties = energyGrid.sim->getScalarProperties();
    std::vector<std::string> scalarPropertyNames;
    int idx{0};
    auto scalarSelector =
        new ui::SelectionList(m_simulationMenu, energyGrid.name + "_scalarSelector");
    for (const auto &[name, scalarProperty] : scalarProperties) {
      if (std::find(scalarPropertyNames.begin(), scalarPropertyNames.end(), name) ==
          scalarPropertyNames.end())
        scalarPropertyNames.push_back(name);

      auto menu = new ui::Menu(m_simulationMenu, energyGrid.name + "_" +
                                                     scalarProperty.species + "_" +
                                                     std::to_string(idx++));
      menu->setVisible(false);
      auto cms = std::make_unique<opencover::CoverColorBar>(menu);
      cms->setSpecies(scalarProperty.species);
      cms->setUnit(scalarProperty.unit);
      auto type = energyGrid.type;
      cms->setCallback([this, type](const opencover::ColorMap &cm) {
        updateEnergyGridColorMapInShader(cm, type);
      });
      cms->setName(energyGrid.name);
      auto min = energyGrid.simUI->min(scalarProperty.species);
      auto max = energyGrid.simUI->max(scalarProperty.species);
      cms->setMinMax(min, max);
      auto halfSpan = (max - min) / 2;
      cms->setMinBounds(min - halfSpan, min + halfSpan);
      cms->setMaxBounds(max - halfSpan, max + halfSpan);
      auto steps = cms->colorMap().steps();

      auto colormapName = scalarProperty.preferredColorMap;
      if (colormapName == core::simulation::INVALID_UNIT)
        colormapName = cms->colorMap().name();

      cms->setColorMap(colormapName);
      cms->setSteps(steps);

      energyGrid.simUI->updateTimestepColors(cms->colorMap());
      energyGrid.colorMapRegistry.emplace(scalarProperty.species,
                                          ColorMapMenu{menu, std::move(cms)});
    }

    scalarSelector->setList(scalarPropertyNames);
    scalarSelector->setCallback([this, &energyGrid](int selected) {
      auto scalarSelection = energyGrid.scalarSelector->selectedItem();
      // NOTE: colormap registry and scalar selector are in sync => if not make sure
      // to adjust this
      bool hudVisible = false;
      for (const auto &colorMap : energyGrid.colorMapRegistry) {
        if (colorMap.second.selector->hudVisible()) {
          hudVisible = true;
          colorMap.second.selector->show(false);
          break;
        }
      }

      auto &colorMapMenu = energyGrid.colorMapRegistry[scalarSelection];
      colorMapMenu.selector->show(hudVisible);
      colorMapMenu.menu->setVisible(true);
      for (auto &[name, menu] : energyGrid.colorMapRegistry) {
        if (name != scalarSelection) {
          menu.menu->setVisible(false);
        }
      }
      updateEnergyGridColorMapInShader(colorMapMenu.selector->colorMap(),
                                       energyGrid.type);
      updateEnergyGridShaderData(energyGrid);
    });
    energyGrid.scalarSelector = scalarSelector;
    energyGrid.scalarSelector->select(scalarPropertyNames.size() - 1, true);
    energyGrid.colorMapRegistry[scalarPropertyNames.back()].menu->setVisible(true);
    updateEnergyGridColorMapInShader(
        energyGrid.colorMapRegistry[scalarSelector->selectedItem()]
            .selector->colorMap(),
        energyGrid.type);
    updateEnergyGridShaderData(energyGrid);
  }
}

void EnergyPlugin::updateEnergyGridShaderData(EnergySimulation &energyGrid) {
  switch (energyGrid.type) {
    case EnergyGridType::PowerGrid: {
      // case EnergyGridType::PowerGridSonder: {
      if (energyGrid.grid && energyGrid.scalarSelector) {
        energyGrid.grid->setData(*energyGrid.sim,
                                 energyGrid.scalarSelector->selectedItem(), false);
      }
      break;
    }
    case EnergyGridType::HeatingGrid: {
      if (energyGrid.grid && energyGrid.scalarSelector) {
        energyGrid.grid->setData(*energyGrid.sim,
                                 energyGrid.scalarSelector->selectedItem(), true);
      }
      break;
    }
    case EnergyGridType::NUM_ENERGY_TYPES:
      // No action needed for NUM_ENERGY_TYPES, it's just a count marker.
      break;
  }
}

void EnergyPlugin::initGrid() {
  initPowerGrid();
  initHeatingGrid();
  buildCoolingGrid();
  initEnergyGridColorMaps();
}

std::vector<EnergyPlugin::IDLookupTable> EnergyPlugin::retrieveBusNameIdMapping(
    COVERUtils::read::CSVStream &stream) {
  auto busNames = IDLookupTable();
  auto busNamesSonder = IDLookupTable();
  CSVStream::CSVRow bus;
  std::string busName(""), type("");
  int id = 0;
  //   while (stream >> bus) {
  while (stream.readNextRow(bus)) {
    ACCESS_CSV_ROW(bus, "name", busName);
    ACCESS_CSV_ROW(bus, "id", id);
    if (bus.find("grid") != bus.end())
      ACCESS_CSV_ROW(bus, "grid", type);
    else
      type = "Normalnetz";  // default type if not specified

    if (type == "Sondernetz") {
      busNamesSonder.insert({id, busName});
      continue;
    }
    busNames.insert({id, busName});
  }
  return {busNames, busNamesSonder};
}

void EnergyPlugin::helper_getAdditionalPowerGridPointData_addData(
    int busId, grid::PointDataList &additionalData, const grid::Data &data) {
  if (busId == -1) return;
  auto &existingDataMap = additionalData[busId];
  if (existingDataMap.empty())
    additionalData[busId] = data;
  else
    existingDataMap.insert(data.begin(), data.end());
}

void EnergyPlugin::helper_getAdditionalPowerGridPointData_handleDuplicate(
    std::string &name, std::map<std::string, uint> &duplicateMap) {
  if (auto it = duplicateMap.find(name); it != duplicateMap.end())
    // if there is a similar entity, add the id to the name
    name = name + "_" + std::to_string(++it->second);
  else
    duplicateMap.insert({name, 0});
}

std::unique_ptr<grid::PointDataList> EnergyPlugin::getAdditionalPowerGridPointData(
    const std::size_t &numOfBus) {
  using PDL = grid::PointDataList;

  // additional bus data
  PDL additionalData;

  for (auto &[tableName, tableStream] : m_powerGridStreams) {
    auto header = tableStream.getHeader();
    if (auto it = std::find(header.begin(), header.end(), "bus"); it == header.end())
      continue;
    auto it = std::find(header.begin(), header.end(), "bus");
    if (it == header.end()) CSVStream::CSVRow busdata;
    int busId = -1;
    std::map<std::string, uint> duplicate{};
    CSVStream::CSVRow row;
    // row
    // while (tableStream >> row) {
    while (tableStream.readNextRow(row)) {
      grid::Data data;
      // column
      for (auto &colName : header) {
        if (!checkBoxSelection_powergrid(tableName, colName)) continue;
        // get bus id without adding it
        if (colName == "bus") {
          ACCESS_CSV_ROW(row, colName, busId);
          continue;
        }
        std::string value;
        ACCESS_CSV_ROW(row, colName, value);

        // add the name of the table to the name
        std::string columnNameWithTable = tableName + " > " + colName;
        helper_getAdditionalPowerGridPointData_handleDuplicate(columnNameWithTable,
                                                               duplicate);
        data[columnNameWithTable] = value;
      }
      helper_getAdditionalPowerGridPointData_addData(busId, additionalData, data);
    }
  }
  return std::make_unique<PDL>(additionalData);
}

std::vector<grid::PointsMap> EnergyPlugin::createPowerGridPoints(
    COVERUtils::read::CSVStream &stream, size_t &numPoints,
    const float &sphereRadius, const std::vector<IDLookupTable> &busNames) {
  using PointsMap = grid::PointsMap;

  CSVStream::CSVRow point;
  float lat = 0, lon = 0;
  PointsMap points;
  PointsMap pointsSonder;
  std::string busName = "", type = "";
  int busID = 0;

  // TODO: need to be adjusted
  auto additionalData = getAdditionalPowerGridPointData(numPoints);

  //   while (stream >> point) {
  while (stream.readNextRow(point)) {
    ACCESS_CSV_ROW(point, "x", lon);
    ACCESS_CSV_ROW(point, "y", lat);
    ACCESS_CSV_ROW(point, "id", busID);

    // x = lon, y = lat
    lon += m_offset[0];
    lat += m_offset[1];

    int i = 0;
    for (const auto &busNames : busNames) {
      if (auto it = busNames.find(busID); it != busNames.end()) {
        if (i == 0)
          type = "Normalnetz";
        else
          type = "Sondernetz";
        busName = it->second;
        break;
      } else {
        busName = busName = "Base_" + std::to_string(busID);
      }
      ++i;
    }

    grid::Data busData;
    try {
      busData = additionalData->at(busID);
    } catch (const std::out_of_range &) {
      busData["base_point_data"] = "";
    }

    osg::ref_ptr<grid::Point> p =
        new grid::Point(busName, lon, lat, m_offset[2], sphereRadius, busData);
    if (type == "Sondernetz")
      pointsSonder.insert({busID, p});
    else
      points.insert({busID, p});
    ++numPoints;
  }
  return {points, pointsSonder};
}

void EnergyPlugin::processGeoBuses(grid::Indices &indices, int &from,
                                   const std::string &geoBuses_comma_seperated,
                                   grid::ConnectionDataList &additionalData,
                                   grid::Data &data) {
  std::stringstream ss(geoBuses_comma_seperated);
  std::string bus("");

  int from_last = from;
  while (std::getline(ss, bus, ',')) {
    auto to_new = std::stoi(bus);
    if (from_last == to_new) continue;
    auto &lastIndicesVec = indices[from_last];
    auto &additionalDataVec = additionalData[from_last];
    auto &toIndicesVec = indices[to_new];

    // NOTE: test implementing skip redundance
    if constexpr (skipRedundance) {
      // get rid of redundant connections
      if (std::find(lastIndicesVec.begin(), lastIndicesVec.end(), to_new) !=
              lastIndicesVec.end() ||
          std::find(toIndicesVec.begin(), toIndicesVec.end(), from_last) !=
              toIndicesVec.end()) {
        from_last = to_new;
        continue;
      }
    }

    // binary insertion to keep the indices sorted
    if (auto lower =
            std::lower_bound(lastIndicesVec.begin(), lastIndicesVec.end(), to_new);
        lower == lastIndicesVec.end()) {
      lastIndicesVec.push_back(to_new);
      additionalDataVec.push_back(data);
    } else {
      auto dataIndex = std::distance(lastIndicesVec.begin(), lower);
      lastIndicesVec.insert(lower, to_new);
      additionalDataVec.insert(additionalDataVec.begin() + dataIndex, data);
    }
    from_last = to_new;
  }
}

osg::ref_ptr<grid::Line> EnergyPlugin::createLine(
    const std::string &name, int &from, const std::string &geoBuses_comma_seperated,
    grid::Data &data, const std::vector<grid::PointsMap> &points) {
  std::stringstream ss(geoBuses_comma_seperated);
  std::string bus("");

  int from_last = from;

  grid::Connections connections;
  while (std::getline(ss, bus, ',')) {
    auto to_new = std::stoi(bus);
    if (from_last == to_new) continue;

    osg::ref_ptr<grid::Point> fromPoint = nullptr;
    osg::ref_ptr<grid::Point> toPoint = nullptr;
    for (auto points : points) {
      auto toIt = points.find(to_new);
      if (!toPoint && toIt != points.end()) toPoint = toIt->second;

      auto fromIt = points.find(from_last);
      if (!fromPoint && fromIt != points.end()) fromPoint = fromIt->second;
    }
    if (!fromPoint || !toPoint) {
      std::cerr << "Invalid bus ID: " << from_last << " or " << to_new << std::endl;
      continue;
    }

    std::string name = fromPoint->getName() + " > " + toPoint->getName();
    float radius = 0.5f;

    grid::ConnectionData conData{name,  fromPoint, toPoint, radius,
                                 false, nullptr,   data};
    connections.push_back(
        new grid::DirectedConnection(conData, grid::ConnectionType::LineWithShader));
    from_last = to_new;
  }
  return new grid::Line(name, connections);
}

std::pair<std::vector<grid::Lines>, std::vector<grid::ConnectionDataList>>
EnergyPlugin::getPowerGridLines(COVERUtils::read::CSVStream &stream,
                                const std::vector<grid::PointsMap> &points) {
  using Lines = grid::Lines;
  using CDL = grid::ConnectionDataList;
  Lines lines;
  CDL additionalData(points[0].size());
  Lines linesSonder;
  CDL additionalDataSonder(points[1].size());

  CSVStream::CSVRow row;
  int from = 0, to = 0;
  std::string geoBuses = "";
  std::string name = "", type = "";
  auto header = stream.getHeader();
  while (stream.readNextRow(row)) {
    grid::Data data;

    for (auto colName : header) {
      fs::path filename(stream.getFilename());
      auto filename_without_ext = filename.stem().string();
      if (!checkBoxSelection_powergrid(filename_without_ext, colName)) continue;
      std::string value;
      ACCESS_CSV_ROW(row, colName, value);
      data[colName] = value;
    }

    ACCESS_CSV_ROW(row, "geo_buses", geoBuses);
    ACCESS_CSV_ROW(row, "from_bus", from);
    ACCESS_CSV_ROW(row, "name", name);
    if (row.find("grid") != row.end())
      ACCESS_CSV_ROW(row, "grid", type);
    else
      type = "Normalnetz";  // default type if not specified

    if (geoBuses.empty()) continue;
    auto line = createLine(name, from, geoBuses, data, points);
    if (type == "Sondernetz") {
      linesSonder.push_back(line);
    } else {
      lines.push_back(line);
    }
  }

  return std::make_pair<vector<Lines>, vector<grid::ConnectionDataList>>(
      {lines, linesSonder}, {additionalData, additionalDataSonder});
}

std::pair<std::unique_ptr<grid::Indices>, std::unique_ptr<grid::ConnectionDataList>>
EnergyPlugin::getPowerGridIndicesAndOptionalData(COVERUtils::read::CSVStream &stream,
                                                 const size_t &numPoints) {
  using Indices = grid::Indices;
  using CDL = grid::ConnectionDataList;
  Indices indices(numPoints);
  CDL additionalData(numPoints);
  CSVStream::CSVRow line;
  int from = 0, to = 0;
  std::string geoBuses = "";
  auto header = stream.getHeader();
  //   while (stream >> line) {
  while (stream.readNextRow(line)) {
    grid::Data data;

    for (auto colName : header) {
      fs::path filename(stream.getFilename());
      auto filename_without_ext = filename.stem().string();
      if (!checkBoxSelection_powergrid(filename_without_ext, colName)) continue;
      std::string value;
      ACCESS_CSV_ROW(line, colName, value);
      data[colName] = value;
    }

    ACCESS_CSV_ROW(line, "geo_buses", geoBuses);
    ACCESS_CSV_ROW(line, "from_bus", from);

    if (geoBuses.empty()) {
      ACCESS_CSV_ROW(line, "to_bus", to);
      indices[from].push_back(to);
      additionalData[from].push_back(data);
    } else {
      processGeoBuses(indices, from, geoBuses, additionalData, data);
    }
  }
  return std::make_pair(std::make_unique<Indices>(indices),
                        std::make_unique<CDL>(additionalData));
}

void EnergyPlugin::buildPowerGrid() {
  using grid::Point;
  if (m_powerGridStreams.empty()) return;

  constexpr float connectionsRadius(0.5f);
  constexpr float sphereRadius(1.0f);
  size_t numPoints(0);

  // fetch bus names
  auto busData = m_powerGridStreams.find("bus");
  std::vector<IDLookupTable> busNames;
  if (busData != m_powerGridStreams.end()) {
    auto &[name, busStream] = *busData;
    busNames = retrieveBusNameIdMapping(busStream);
  }

  if (busNames.empty()) return;

  // create points
  auto pointsData = m_powerGridStreams.find("bus_geodata");
  std::vector<grid::PointsMap> points;
  if (pointsData != m_powerGridStreams.end()) {
    auto &[name, pointStream] = *pointsData;
    points = createPowerGridPoints(pointStream, numPoints, sphereRadius, busNames);
  }

  // create line
  auto lineData = m_powerGridStreams.find("line");
  std::vector<grid::Lines> lines;
  std::vector<grid::ConnectionDataList> optData;
  if (lineData != m_powerGridStreams.end()) {
    auto &[name, lineStream] = *lineData;
    std::tie(lines, optData) = getPowerGridLines(lineStream, points);
  }

  // create grid
  if (lines[0].empty() || lines[1].empty() || points.empty()) return;

  grid::PointsMap mergedPoints = points[0];
  mergedPoints.insert(points[1].begin(), points[1].end());
  // TODO: workaround for merging => PLS REFACTOR LATER
  grid::Lines mergedLines = lines[0];
  mergedLines.insert(mergedLines.end(), lines[1].begin(), lines[1].end());

  grid::ConnectionDataList mergedOptData = optData[0];
  mergedOptData.insert(mergedOptData.end(), optData[1].begin(), optData[1].end());

  auto idx = getEnergyGridTypeIndex(EnergyGridType::PowerGrid);
  //   auto idxSonder = getEnergyGridTypeIndex(EnergyGridType::PowerGridSonder);
  auto &egrid = m_energyGrids[idx];
  //   auto &egridSonder = m_energyGrids[idxSonder];
  auto &powerGroup = egrid.group;
  //   auto &powerGroupSonder = egridSonder.group;
  powerGroup = new osg::MatrixTransform;
  //   powerGroupSonder = new osg::MatrixTransform;
  auto font = configString("Billboard", "font", "default")->value();
  TxtBoxAttributes infoboardAttributes = TxtBoxAttributes(
      osg::Vec3(0, 0, 0), "EnergyGridText", font, 50, 50, 2.0f, 0.1, 2);
  powerGroup->setName("PowerGrid");
  //   powerGroupSonder->setName("PowerGridSonder");

  EnergyGridConfig econfig("POWER", {}, grid::Indices(), mergedPoints, powerGroup,
                           connectionsRadius, mergedOptData, infoboardAttributes,
                           EnergyGridConnectionType::Line, mergedLines);
  //   EnergyGridConfig econfig("POWER", {}, grid::Indices(), points[0], powerGroup,
  //                            connectionsRadius, optData[0], infoboardAttributes,
  //                            EnergyGridConnectionType::Line, lines[0]);
  //   EnergyGridConfig econfigSonder("POWERSonder", {}, grid::Indices(), points[1],
  //                                  powerGroupSonder, connectionsRadius,
  //                                  optData[1], infoboardAttributes,
  //                                  EnergyGridConnectionType::Line, lines[1]);

  auto powerGrid = std::make_unique<EnergyGrid>(econfig, false);
  powerGrid->initDrawables();
  //   powerGrid->updateColor(
  //       osg::Vec4(255.0f / 255.0f, 222.0f / 255.0f, 33.0f / 255.0f, 1.0f));
  egrid.grid = std::move(powerGrid);
  addEnergyGridToGridSwitch(egrid.group);

  //   auto powerGridSonder = std::make_unique<EnergyGrid>(econfigSonder, false);
  //   powerGridSonder->initDrawables();
  //   //   powerGridSonder->updateColor(
  //   //       osg::Vec4(0.0f / 255.0f, 200.0f / 255.0f, 33.0f / 255.0f, 1.0f));
  //   egridSonder.grid = std::move(powerGridSonder);
  //   addEnergyGridToGridSwitch(egridSonder.group);

  // TODO:
  //  [ ] set trafo as 3d model or block

  // how to implement this generically?
  // - fixed grid structure for discussion in AK Software
  // - look into Energy ADE
}

void EnergyPlugin::initHeatingGridStreams() {
  auto heatingGridDir =
      configString("Simulation", "heatingGridDir", "default")->value();
  fs::path dir_path(heatingGridDir);
  if (!fs::exists(dir_path)) return;
  m_heatingGridStreams = getCSVStreams(dir_path);
  if (m_heatingGridStreams.empty()) {
    std::cout << "No csv files found in " << heatingGridDir << std::endl;
    return;
  }
}

void EnergyPlugin::initHeatingGrid() {
  initHeatingGridStreams();
  buildHeatingGrid();
  applySimulationDataToHeatingGrid();
  m_heatingGridStreams.clear();
}

std::vector<int> EnergyPlugin::createHeatingGridIndices(
    const std::string &pointName,
    const std::string &connectionsStrWithCommaDelimiter,
    grid::ConnectionDataList &additionalConnectionData) {
  std::vector<int> connectivityList{};
  std::stringstream ss(connectionsStrWithCommaDelimiter);
  std::string connection("");

  while (std::getline(ss, connection, ' ')) {
    if (connection.empty() || connection == INVALID_CELL_VALUE) continue;
    grid::Data connectionData{{"name", pointName + "_" + connection}};
    additionalConnectionData.emplace_back(std::vector{connectionData});
    connectivityList.push_back(std::stoi(connection));
  }
  return connectivityList;
}

osg::ref_ptr<grid::Point> EnergyPlugin::searchHeatingGridPointById(
    const grid::Points &points, int id) {
  auto pointIt = std::find_if(points.begin(), points.end(), [id](const auto &p) {
    return std::stoi(p->getName()) == id;
  });
  if (pointIt == points.end()) {
    std::cerr << "Point with id " << id << " not found in points." << std::endl;
  }
  return *pointIt;  // returns nullptr if not found
}

// int getId(const std::string &connectionsStrWithCommaDelimiter,
//     grid::ConnectionDataList &additionalData)
// {
//   std::stringstream ss(connectionsStrWithCommaDelimiter);
//   std::string connection("");
//   grid::Connections connections;
//   auto pointName = from->getName();
//   std::string lineName{pointName};
//   while (std::getline(ss, connection, ' ')) {
//     if (connection.empty() || connection == INVALID_CELL_VALUE) continue;
//     grid::Data connectionData{{"name", pointName + "_" + connection}};
//     additionalData.emplace_back(std::vector{connectionData});
//     int toID(-1);
//     try {
//       toID = std::stoi(connection);
//     } catch (...) {
//       continue;
//     }
//   }
// }

osg::ref_ptr<grid::Line> EnergyPlugin::createHeatingGridLine(
    const grid::Points &points, osg::ref_ptr<grid::Point> from,
    const std::string &connectionsStrWithCommaDelimiter,
    grid::ConnectionDataList &additionalData) {
  std::string connection("");
  grid::Connections gridConnections;
  auto pointName = from->getName();
  std::string lineName{pointName};
  auto connections = split(connectionsStrWithCommaDelimiter, ' ');
  for (const auto &connection : connections) {
    if (connection.empty() || connection == INVALID_CELL_VALUE) continue;
    grid::Data connectionData{{"name", pointName + "_" + connection}};
    additionalData.emplace_back(std::vector{connectionData});
    int toID(-1);
    try {
      toID = std::stoi(connection);
    } catch (...) {
      continue;
    }
    lineName +=
        std::string(" ") + UIConstants::RIGHT_ARROW_UNICODE_HEX + " " + connection;

    // TODO: Really bad solution to find the point by id, but the id is not
    // necessarily the index in the points vector, so we need to find it by name =>
    // refactor the Points structure to use std::map later
    auto to = searchHeatingGridPointById(points, toID);
    if (to == nullptr) {
      std::cerr << "Point with id " << toID << " not found in points." << std::endl;
      continue;
    }
    grid::ConnectionData connData{
        pointName + "_" + connection, from, to, 0.5f, true, nullptr, connectionData};
    grid::DirectedConnection directed(connData,
                                      grid::ConnectionType::LineWithShader);
    gridConnections.push_back(new grid::DirectedConnection(directed));
  }

  return new grid::Line(lineName, gridConnections);
}

void EnergyPlugin::readSimulationDataStream(
    COVERUtils::read::CSVStream &heatingSimStream) {
  auto idx = getEnergyGridTypeIndex(EnergyGridType::HeatingGrid);
  if (m_energyGrids[idx].grid == nullptr) return;
  std::regex consumer_value_split_regex("Consumer_(\\d+)_(.+)");
  std::regex producer_value_split_regex("Producer_(\\d+)_(.+)");
  std::smatch match;

  CSVStream::CSVRow row;
  auto sim = std::make_shared<heating::HeatingSimulation>();
  const auto &header = heatingSimStream.getHeader();
  auto &consumers = sim->Consumers();
  auto &producers = sim->Producers();
  double val = 0.0f;
  std::string name(""), valName("");
  while (heatingSimStream.readNextRow(row)) {
    for (const auto &col : header) {
      ACCESS_CSV_ROW(row, col, val);
      if (std::regex_search(col, match, consumer_value_split_regex)) {
        name = match[1];
        valName = match[2];
        consumers.add(name);
        consumers.addDataToContainerObject(name, valName, val);
      } else if (std::regex_search(col, match, producer_value_split_regex)) {
        name = match[1];
        valName = match[2];
        producers.add(name);
        producers.addDataToContainerObject(name, valName, val);
      } else {
        if (val == 0) continue;
        sim->addData(col, val);
      }
    }
  }
  auto &heatingGrid = m_energyGrids[idx];

  heatingGrid.simUI = std::make_unique<HeatingSimUI>(sim, heatingGrid.grid);
  heatingGrid.sim = std::move(sim);

  auto timesteps = heatingGrid.sim->getTimesteps("mass_flow");
  std::cout << "Number of timesteps: " << timesteps << std::endl;
  setAnimationTimesteps(timesteps, heatingGrid.group);
}

void EnergyPlugin::applySimulationDataToHeatingGrid() {
  if (m_heatingGridStreams.empty()) return;
  auto simulationData = m_heatingGridStreams.find("results");
  if (simulationData == m_heatingGridStreams.end()) return;

  auto &[_, stream] = *simulationData;
  readSimulationDataStream(stream);
}

grid::Lines EnergyPlugin::createHeatingGridLines(
    const grid::Points &points, const std::map<int, std::string> &connectionStrings,
    grid::ConnectionDataList &additionalData) {
  grid::Lines lines;
  for (auto it = connectionStrings.begin(); it != connectionStrings.end(); ++it) {
    int id = it->first;
    const std::string &connectionsStr = it->second;
    if (connectionsStr.empty() || connectionsStr == INVALID_CELL_VALUE) continue;
    // TODO: Really bad solution to find the point by id, but the id is not
    // necessarily the index in the points vector, so we need to find it by name =>
    // refactor the Points structure to use std::map later
    auto from = searchHeatingGridPointById(points, id);
    if (from == nullptr) {
      std::cerr << "Point with id " << id << " not found in points." << std::endl;
      continue;
    }
    auto line = createHeatingGridLine(points, from, connectionsStr, additionalData);
    if (line == nullptr) {
      std::cerr << "Failed to create line for point: " << from->getName()
                << std::endl;
      continue;
    }
    lines.push_back(line);
  }
  return lines;
}

std::pair<grid::Points, grid::Data> EnergyPlugin::createHeatingGridPointsAndData(
    COVERUtils::read::CSVStream &heatingStream,
    std::map<int, std::string> &connectionStrings) {
  grid::Points points{};
  grid::Data pointData{};
  CSVStream::CSVRow row;
  std::string name = "", connections = "", label = "", type = "";
  float lat = 0.0f, lon = 0.0f;
  auto checkForInvalidValue = [](const std::string &value) {
    return value == INVALID_CELL_VALUE;
  };

  auto addToPointData = [&checkForInvalidValue](grid::Data &pointData,
                                                const std::string &key,
                                                const std::string &value) {
    if (!checkForInvalidValue(value)) pointData[key] = value;
  };
  //   while (heatingStream >> row) {
  while (heatingStream.readNextRow(row)) {
    ACCESS_CSV_ROW(row, "connections", connections);
    ACCESS_CSV_ROW(row, "id", name);
    ACCESS_CSV_ROW(row, "Latitude", lat);
    ACCESS_CSV_ROW(row, "Longitude", lon);
    ACCESS_CSV_ROW(row, "Label", label);
    ACCESS_CSV_ROW(row, "Type", type);

    addToPointData(pointData, "name", name);
    addToPointData(pointData, "label", label);
    addToPointData(pointData, "type", type);

    projTransLatLon(lat, lon);

    int strangeId = std::stoi(name);

    // create a point
    osg::ref_ptr<grid::Point> point =
        new grid::Point(name, lon, lat, m_offset[2], 1.0f, pointData);
    points.push_back(point);

    // needs cleanup because dataset is not final and has empty cells => no need to
    // display them
    pointData.clear();
    row.clear();
    if (connections.empty() || connections == INVALID_CELL_VALUE) {
      std::cerr << "No connections for point: " << name << std::endl;
      continue;
    }
    connectionStrings[strangeId] = connections;
  }

  return std::make_pair(points, pointData);
}

void EnergyPlugin::readHeatingGridStream(CSVStream &heatingStream) {
  CSVStream::CSVRow row;
  grid::ConnectionDataList additionalConnectionData{};
  auto egridIdx = getEnergyGridTypeIndex(EnergyGridType::HeatingGrid);
  m_energyGrids[egridIdx].group = new osg::MatrixTransform;
  auto font = configString("Billboard", "font", "default")->value();
  TxtBoxAttributes infoboardAttributes = TxtBoxAttributes(
      osg::Vec3(0, 0, 0), "EnergyGridText", font, 50, 50, 2.0f, 0.1, 2);

  std::map<int, std::string> connectionStrings;
  auto [points, pointData] =
      createHeatingGridPointsAndData(heatingStream, connectionStrings);
  auto lines =
      createHeatingGridLines(points, connectionStrings, additionalConnectionData);

  auto &heatingGrid = m_energyGrids[egridIdx];
  heatingGrid.group->setName(heatingGrid.name);
  heatingGrid.grid =
      std::make_unique<EnergyGrid>(EnergyGridConfig{"HEATING",
                                                    points,
                                                    {},
                                                    {},
                                                    heatingGrid.group,
                                                    0.5f,
                                                    additionalConnectionData,
                                                    infoboardAttributes,
                                                    EnergyGridConnectionType::Line,
                                                    lines});
  heatingGrid.grid->initDrawables();
  addEnergyGridToGridSwitch(heatingGrid.group);
  switchEnergyGrid(EnergyGridType::HeatingGrid);
}

void EnergyPlugin::addEnergyGridToGridSwitch(
    osg::ref_ptr<osg::Group> energyGridGroup) {
  assert(energyGridGroup && "EnergyGridGroup is nullptr");
  m_grid->addChild(energyGridGroup);
  switchTo(energyGridGroup, m_grid);
}

void EnergyPlugin::buildHeatingGrid() {
  if (m_heatingGridStreams.empty()) return;

  // find correct csv
  auto heatingIt = m_heatingGridStreams.find("heating_network_simple");
  if (heatingIt == m_heatingGridStreams.end()) return;

  auto &[_, heatingStream] = *heatingIt;
  readHeatingGridStream(heatingStream);
}

void EnergyPlugin::buildCoolingGrid() {
  // NOTE: implement when data is available
}

/* #endregion */
COVERPLUGIN(EnergyPlugin)
