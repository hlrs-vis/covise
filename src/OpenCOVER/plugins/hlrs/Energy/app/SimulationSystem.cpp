// NEEDS TO BE INCLUDED FIRST
// apache arrow
// colliding with qt signals => compiler sees this as parameter name => QT replaces
// signals with QT_SIGNALS which expands to public
// QT_ANNOTATE_ACCESS_SPECIFIER(qt_signal)
// By simply put the includes for arrow before qt
// includes. In our case before everything else we can resolve the issue.
#include <lib/apache/arrow.h>
#include "SimulationSystem.h"

// cover
#include <cover/coVRAnimationManager.h>

// std
#include <algorithm>
#include <iostream>
#include <regex>

// ui
#include "ui/simulation/HeatingSimulationUI.h"
#include "ui/simulation/PowerSimulationUI.h"

// PluginUtil
#include <PluginUtil/coShaderUtil.h>
#include <util/string_util.h>

// presentation
#include <app/presentation/EnergyGrid.h>
#include <app/presentation/TxtInfoboard.h>

// core
#include <lib/core/constants.h>
#include <lib/core/simulation/heating.h>
#include <lib/core/simulation/object.h>
#include <lib/core/simulation/object_factory.h>
#include <lib/core/simulation/object_type.h>
#include <lib/core/simulation/power.h>
#include <lib/core/simulation/simulation.h>
#include <lib/core/utils/color.h>
#include <lib/core/utils/osgUtils.h>

// other
#include <proj.h>

// app
#include "CityGMLSystem.h"

using namespace opencover;
using namespace core::interface;
namespace fs = boost::filesystem;

namespace {
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

template <typename T>
void printObjectContainerDistribution(
    const ObjectMap &map, float min, float max, int numBins = 20,
    const std::string &species = "loading_percent") {
  static_assert(std::is_base_of_v<Object, T>,
                "T must be derived from core::simulation::Object");
  std::vector<int> histogram(numBins, 0);
  int total = 0;

  for (const auto &object : map) {
    auto it = object.second->getData().find(species);
    if (it == object.second->getData().end()) continue;
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

float computeDistributionCenter(const std::vector<float> &values) {
  float sum = 0;
  for (auto &value : values) sum += value;
  return sum / values.size();
}
};  // namespace

SimulationSystem::SimulationSystem(opencover::coVRPlugin *plugin,
                                   opencover::ui::Menu *parentMenu,
                                   CityGMLSystem *cityGMLSystem,
                                   osg::ref_ptr<osg::Switch> parent)
    : m_plugin(plugin),
      m_gridSwitch(parent),
      m_cityGMLSystem(cityGMLSystem),
      m_offset(3),
      m_enabled(false),
      m_simulationMenu(nullptr),
      m_energygridGroup(nullptr),
      m_energygridBtnGroup(nullptr),
      m_scenarios(nullptr),
      m_liftGrids(nullptr),
      m_status_quo(nullptr),
      m_future_ev(nullptr),
      m_future_ev_pv(nullptr),
      m_rule_base_bigger_hp(nullptr),
      m_rule_based(nullptr),
      m_optimized_bigger_awz(nullptr),
      m_optimized(nullptr),
      m_powerGridMenu(nullptr),
      m_updatePowerGridSelection(nullptr),
      m_powerGridSelectionPtr(nullptr),
      m_powerGridCheckboxes(),
      m_powerGridStreams(),
      m_heatingGridStreams(),
      m_powerGridDir(""),
      m_energyGrids({EnergySimulation{"PowerGrid", EnergyGridType::PowerGrid},
                     EnergySimulation{"HeatingGrid", EnergyGridType::HeatingGrid}}) {
  m_offset =
      m_plugin->configFloatArray("General", "offset", std::vector<double>{0, 0, 0})
          ->value();
  initSimUI(parentMenu);
}

SimulationSystem::~SimulationSystem() {}

void SimulationSystem::init() { initGrid(); }

void SimulationSystem::enable(bool on) {}

void SimulationSystem::update() {
  for (auto &energyGrid : m_energyGrids) {
    if (!energyGrid.grid) continue;
    energyGrid.grid->update();
  }
}

void SimulationSystem::updateTime(int timestep) {
  auto &energyGrid = m_energyGrids[m_energygridBtnGroup->value()];
  if (energyGrid.grid) energyGrid.grid->updateTime(timestep);
}

void SimulationSystem::preFrame() {
  opencover::ColorBar::HudPosition hudPos;
  auto numHuds = 0;
  for (auto &energyGrid : m_energyGrids) {
    if (!energyGrid.scalarSelector) continue;
    const auto &selectedScalar = energyGrid.scalarSelector->selectedItem();
    auto &colorMapMenu = energyGrid.colorMapRegistry[selectedScalar];
    if (colorMapMenu.selector && colorMapMenu.selector->hudVisible()) {
      hudPos.setNumHuds(numHuds++);
      colorMapMenu.selector->setHudPosition(hudPos);
    }
  }
}

void SimulationSystem::updateEnergyGridColorMapInShader(
    const opencover::ColorMap &map, EnergyGridType type) {
  auto gridTypeIndex = getEnergyGridTypeIndex(type);
  auto &grid = m_energyGrids[gridTypeIndex];
  if (grid.group && core::utils::osgUtils::isActive(m_gridSwitch, grid.group) &&
      grid.simUI) {
    // TODO: remove this later
    // HACK: this is a workaround
    grid.grid->setColorMap(map, m_vmPuColorMap->colorMap());
  }
}

std::string SimulationSystem::getScenarioName(Scenario scenario) {
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

void SimulationSystem::setAnimationTimesteps(size_t maxTimesteps, const void *who) {
  if (maxTimesteps > opencover::coVRAnimationManager::instance()->getNumTimesteps())
    opencover::coVRAnimationManager::instance()->setNumTimesteps(maxTimesteps, who);
}
/* #endregion */

void SimulationSystem::initSimMenu(opencover::ui::Menu *parent) {
  if (m_simulationMenu != nullptr) return;
  if (parent == nullptr) {
    std::cerr << "Parent menu is null, cannot initialize simulation menu."
              << std::endl;
    return;
  }

  m_simulationMenu = new ui::Menu(parent, "Simulation");
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
    core::utils::osgUtils::switchTo(
        m_energyGrids[getEnergyGridTypeIndex(EnergyGridType::PowerGrid)].group,
        m_gridSwitch);
    // auto scenarioIndex = getScenarioIndex(Scenario(value));
    std::string scenarioName = getScenarioName(Scenario(value));
    auto simPath =
        m_plugin->configString("Simulation", "powerSimDir", "default")->value();
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

void SimulationSystem::switchEnergyGrid(EnergyGridType grid) {
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
  core::utils::osgUtils::switchTo(switch_to, m_gridSwitch);
}

void SimulationSystem::initEnergyGridUI() {
  if (m_simulationMenu == nullptr) {
    std::cerr << "Simulation menu not initialized before calling function "
                 "initEnergyGridUI"
              << std::endl;
    return;
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

void SimulationSystem::initSimUI(opencover::ui::Menu *parent) {
  initSimMenu(parent);
  initEnergyGridUI();
  initPowerGridUI(parent,
                  {"trafo3w_stdtypes", "trafo_std_types", "trafo", "parameters",
                   "dtypes", "bus_geodata", "fuse_std_types", "line_std_types"});
}

void SimulationSystem::initPowerGridStreams() {
  auto powerGridDir =
      m_plugin->configString("Simulation", "powerGridDir", "default")->value();
  fs::path dir_path(powerGridDir);
  if (!fs::exists(dir_path)) return;
  m_powerGridStreams = getCSVStreams(dir_path);
  if (m_powerGridStreams.empty()) {
    std::cout << "No csv files found in " << powerGridDir << std::endl;
    return;
  }
}

bool SimulationSystem::checkBoxSelection_powergrid(const std::string &tableName,
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

void SimulationSystem::rebuildPowerGrid() {
  auto idx = getEnergyGridTypeIndex(EnergyGridType::PowerGrid);
  //   auto idxSonder = getEnergyGridTypeIndex(EnergyGridType::PowerGridSonder);
  m_gridSwitch->removeChild(m_energyGrids[idx].group);
  //   m_grid->removeChild(m_energyGrids[idxSonder].group);
  initPowerGridStreams();
  buildPowerGrid();
}

void SimulationSystem::updatePowerGridConfig(const std::string &tableName,
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

void SimulationSystem::updatePowerGridSelection(bool on) {
  if (!on) return;
  m_updatePowerGridSelection->setState(false);
  rebuildPowerGrid();
}

void SimulationSystem::initPowerGridUI(
    opencover::ui::Menu *parent, const std::vector<std::string> &tablesToSkip) {
  if (m_powerGridStreams.empty()) initPowerGridStreams();
  m_powerGridMenu = new opencover::ui::Menu("PowerGridData", parent);

  m_updatePowerGridSelection = new opencover::ui::Button(m_powerGridMenu, "Update");
  m_updatePowerGridSelection->setState(false);
  m_updatePowerGridSelection->setCallback([this](bool enable) {
    updatePowerGridSelection(enable);
    auto idx = getEnergyGridTypeIndex(EnergyGridType::PowerGrid);
    core::utils::osgUtils::switchTo(m_energyGrids[idx].group, m_gridSwitch);
  });

  m_powerGridSelectionPtr = m_plugin->configBoolArray(
      "Simulation", "powerGridDataSelection", std::vector<bool>{});
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

void SimulationSystem::applySimulationDataToPowerGrid(const std::string &simPath) {
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
  auto processColumns = [&](const std::shared_ptr<arrow::Table> &tbl, ObjectMap &map,
                            const std::string &dataKey) {
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
          // create entry if it doesn't exist
          if (map.find(columnName) == map.end()) {
            Data data = {{dataKey, std::vector<double>(column->length())}};
            auto object = std::make_unique<Object>(columnName, std::move(data));
            map.emplace(columnName, std::move(object));
          }
          auto &vec = map[columnName]->getData()[dataKey];
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
  powerGrid.simUI =
      std::make_unique<PowerSimulationUI<IEnergyGrid>>(sim, powerGrid.grid);
  powerGrid.sim = std::move(sim);

  if (m_cityGMLSystem) {
    const auto &[min, max] = powerGrid.sim->getMinMax("res_mw");
    const auto &preferredColorMap = powerGrid.sim->getPreferredColorMap("res_mw");
    m_cityGMLSystem->updateInfluxColorMaps(min, max, powerGrid.sim,
                                           preferredColorMap, "res_mw", "MW");
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

void SimulationSystem::initPowerGrid() {
  initPowerGridStreams();
  buildPowerGrid();
  m_powerGridStreams.clear();
  auto simPath =
      m_plugin->configString("Simulation", "powerSimDir", "default")->value();
  simPath += "/status_quo";
  applySimulationDataToPowerGrid(simPath);
}

void SimulationSystem::initEnergyGridColorMaps() {
  if (m_simulationMenu == nullptr) {
    std::cerr << "Simulation menu not initialized before calling function "
                 "initEnergyGridColorMaps"
              << std::endl;
    return;
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

void SimulationSystem::updateEnergyGridShaderData(EnergySimulation &energyGrid) {
  switch (energyGrid.type) {
    case EnergyGridType::PowerGrid: {
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

void SimulationSystem::initGrid() {
  initPowerGrid();
  initHeatingGrid();
  //   buildCoolingGrid();
  initEnergyGridColorMaps();
}

std::vector<SimulationSystem::IDLookupTable>
SimulationSystem::retrieveBusNameIdMapping(CSVStream &stream) {
  auto busNames = IDLookupTable();
  auto busNamesSonder = IDLookupTable();
  CSVStream::CSVRow bus;
  std::string busName(""), type("");
  int id = 0;
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

void SimulationSystem::helper_getAdditionalPowerGridPointData_addData(
    int busId, grid::PointDataList &additionalData, const grid::Data &data) {
  if (busId == -1) return;
  auto &existingDataMap = additionalData[busId];
  if (existingDataMap.empty())
    additionalData[busId] = data;
  else
    existingDataMap.insert(data.begin(), data.end());
}

void SimulationSystem::helper_getAdditionalPowerGridPointData_handleDuplicate(
    std::string &name, std::map<std::string, uint> &duplicateMap) {
  if (auto it = duplicateMap.find(name); it != duplicateMap.end())
    // if there is a similar entity, add the id to the name
    name = name + "_" + std::to_string(++it->second);
  else
    duplicateMap.insert({name, 0});
}

std::unique_ptr<grid::PointDataList>
SimulationSystem::getAdditionalPowerGridPointData(const std::size_t &numOfBus) {
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

std::vector<grid::PointsMap> SimulationSystem::createPowerGridPoints(
    CSVStream &stream, size_t &numPoints, const float &sphereRadius,
    const std::vector<IDLookupTable> &busNames) {
  using PointsMap = grid::PointsMap;

  CSVStream::CSVRow point;
  float lat = 0, lon = 0;
  PointsMap points;
  PointsMap pointsSonder;
  std::string busName = "", type = "";
  int busID = 0;

  // TODO: need to be adjusted
  auto additionalData = getAdditionalPowerGridPointData(numPoints);

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

void SimulationSystem::processGeoBuses(grid::Indices &indices, int &from,
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

osg::ref_ptr<grid::Line> SimulationSystem::createLine(
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
SimulationSystem::getPowerGridLines(CSVStream &stream,
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
SimulationSystem::getPowerGridIndicesAndOptionalData(CSVStream &stream,
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

void SimulationSystem::buildPowerGrid() {
  using grid::Point;
  if (m_powerGridStreams.empty()) return;

  constexpr float connectionsRadius(1.0f);
  constexpr float sphereRadius(2.0f);
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
  auto &egrid = m_energyGrids[idx];
  auto &powerGroup = egrid.group;
  powerGroup = new osg::MatrixTransform;
  auto font = m_plugin->configString("Billboard", "font", "default")->value();
  TxtBoxAttributes infoboardAttributes = TxtBoxAttributes(
      osg::Vec3(0, 0, 0), "EnergyGridText", font, 50, 50, 2.0f, 0.1, 2);
  powerGroup->setName("PowerGrid");

  EnergyGridConfig econfig("POWER", {}, grid::Indices(), mergedPoints, powerGroup,
                           connectionsRadius, mergedOptData, infoboardAttributes,
                           EnergyGridConnectionType::Line, mergedLines);

  auto powerGrid = std::make_unique<EnergyGrid>(econfig, false);
  powerGrid->initDrawables();
  egrid.grid = std::move(powerGrid);
  addEnergyGridToGridSwitch(egrid.group);

  // TODO:
  //  [ ] set trafo as 3d model or block

  // how to implement this generically?
  // - fixed grid structure for discussion in AK Software
  // - look into Energy ADE
}

void SimulationSystem::initHeatingGridStreams() {
  auto heatingGridDir =
      m_plugin->configString("Simulation", "heatingGridDir", "default")->value();
  fs::path dir_path(heatingGridDir);
  if (!fs::exists(dir_path)) return;
  m_heatingGridStreams = getCSVStreams(dir_path);
  if (m_heatingGridStreams.empty()) {
    std::cout << "No csv files found in " << heatingGridDir << std::endl;
    return;
  }
}

void SimulationSystem::initHeatingGrid() {
  initHeatingGridStreams();
  buildHeatingGrid();
  applySimulationDataToHeatingGrid();
  m_heatingGridStreams.clear();
}

std::vector<int> SimulationSystem::createHeatingGridIndices(
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

osg::ref_ptr<grid::Point> SimulationSystem::searchHeatingGridPointById(
    const grid::Points &points, int id) {
  auto pointIt = std::find_if(points.begin(), points.end(), [id](const auto &p) {
    return std::stoi(p->getName()) == id;
  });
  if (pointIt == points.end()) {
    std::cerr << "Point with id " << id << " not found in points." << std::endl;
  }
  return *pointIt;  // returns nullptr if not found
}

osg::ref_ptr<grid::Line> SimulationSystem::createHeatingGridLine(
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

void SimulationSystem::readSimulationDataStream(CSVStream &heatingSimStream) {
  auto idx = getEnergyGridTypeIndex(EnergyGridType::HeatingGrid);
  if (m_energyGrids[idx].grid == nullptr) return;
  std::regex consumer_value_split_regex("Consumer_(\\d+)_(.+)");
  std::regex producer_value_split_regex("Producer_(\\d+)_(.+)");
  std::smatch match;

  CSVStream::CSVRow row;
  auto sim = std::make_shared<heating::HeatingSimulation>();
  const auto &header = heatingSimStream.getHeader();
  double val = 0.0f;
  std::string name(""), valName("");

  auto getObjMapByType = [&](core::simulation::ObjectType type) -> ObjectMap * {
    if (type == core::simulation::ObjectType::Consumer)
      return &sim->Consumers();
    else if (type == core::simulation::ObjectType::Producer)
      return &sim->Producers();
    return nullptr;
  };

  auto createObjAndAddToMap = [&](core::simulation::ObjectType type,
                                  const std::string &name) {
    auto obj =
        core::simulation::createObject(type, name, {{std::string("value"), {}}});
    auto map = getObjMapByType(type);
    if (map == nullptr) return;
    map->emplace(name, std::move(obj));
  };

  auto getObjPtr = [&](core::simulation::ObjectType type, const std::string &name) {
    auto map = getObjMapByType(type);
    auto it = map->find(name);
    if (it != map->end()) return it->second.get();
    createObjAndAddToMap(type, name);
    return map->at(name).get();
  };

  auto addDataToMap = [&](core::simulation::ObjectType type, const std::string &name,
                          const std::string &valName, double value) {
    auto objPtr = getObjPtr(type, name);
    objPtr->addData(valName, value);
  };

  while (heatingSimStream.readNextRow(row)) {
    for (const auto &col : header) {
      ACCESS_CSV_ROW(row, col, val);
      if (std::regex_search(col, match, consumer_value_split_regex)) {
        name = match[1];
        valName = match[2];
        addDataToMap(core::simulation::ObjectType::Consumer, name, valName, val);
      } else if (std::regex_search(col, match, producer_value_split_regex)) {
        name = match[1];
        valName = match[2];
        addDataToMap(core::simulation::ObjectType::Producer, name, valName, val);
      } else {
        if (val == 0) continue;
        sim->addData(col, val);
      }
    }
  }
  auto &heatingGrid = m_energyGrids[idx];

  heatingGrid.simUI =
      std::make_unique<HeatingSimulationUI<IEnergyGrid>>(sim, heatingGrid.grid);
  heatingGrid.sim = std::move(sim);

  auto timesteps = heatingGrid.sim->getTimesteps("mass_flow");
  std::cout << "Number of timesteps: " << timesteps << std::endl;
  setAnimationTimesteps(timesteps, heatingGrid.group);
}

void SimulationSystem::applySimulationDataToHeatingGrid() {
  if (m_heatingGridStreams.empty()) return;
  auto simulationData = m_heatingGridStreams.find("results");
  if (simulationData == m_heatingGridStreams.end()) return;

  auto &[_, stream] = *simulationData;
  readSimulationDataStream(stream);
}

grid::Lines SimulationSystem::createHeatingGridLines(
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

std::pair<grid::Points, grid::Data> SimulationSystem::createHeatingGridPointsAndData(
    CSVStream &heatingStream, std::map<int, std::string> &connectionStrings) {
  grid::Points points{};
  grid::Data pointData{};
  CSVStream::CSVRow row;
  std::string name = "", connections = "", label = "", type = "";
  float lat = 0.0f, lon = 0.0f;
  auto checkForInvalidValue = [](const std::string &value) {
    return value == INVALID_CELL_VALUE;
  };

  std::string projFrom =
      m_plugin->configString("General", "projFrom", "default")->value();
  std::string projTo =
      m_plugin->configString("General", "projTo", "default")->value();
  auto P =
      proj_create_crs_to_crs(PJ_DEFAULT_CTX, projFrom.c_str(), projTo.c_str(), NULL);
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

  auto addToPointData = [&checkForInvalidValue](grid::Data &pointData,
                                                const std::string &key,
                                                const std::string &value) {
    if (!checkForInvalidValue(value)) pointData[key] = value;
  };
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

    coord.lpzt.lam = lon;
    coord.lpzt.phi = lat;

    coord = proj_trans(P, PJ_FWD, coord);

    lon = coord.xy.x + m_offset[0];
    lat = coord.xy.y + m_offset[1];

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

void SimulationSystem::readHeatingGridStream(CSVStream &heatingStream) {
  CSVStream::CSVRow row;
  grid::ConnectionDataList additionalConnectionData{};
  auto egridIdx = getEnergyGridTypeIndex(EnergyGridType::HeatingGrid);
  m_energyGrids[egridIdx].group = new osg::MatrixTransform;
  auto font = m_plugin->configString("Billboard", "font", "default")->value();
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

void SimulationSystem::addEnergyGridToGridSwitch(
    osg::ref_ptr<osg::Group> energyGridGroup) {
  assert(energyGridGroup && "EnergyGridGroup is nullptr");
  m_gridSwitch->addChild(energyGridGroup);
  core::utils::osgUtils::switchTo(energyGridGroup, m_gridSwitch);
}

void SimulationSystem::buildHeatingGrid() {
  if (m_heatingGridStreams.empty()) return;

  // find correct csv
  auto heatingIt = m_heatingGridStreams.find("heating_network_simple");
  if (heatingIt == m_heatingGridStreams.end()) return;

  auto &[_, heatingStream] = *heatingIt;
  readHeatingGridStream(heatingStream);
}

void SimulationSystem::buildCoolingGrid() {
  // NOTE: implement when data is available
  throw std::runtime_error(
      "SimulationSystem::buildCoolingGrid() is not implemented yet.");
}
