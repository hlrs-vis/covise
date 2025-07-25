#pragma once

// core
#include <lib/core/interfaces/IEnergyGrid.h>
#include <lib/core/interfaces/ISystem.h>
#include <lib/core/simulation/simulation.h>

// COVER
#include <PluginUtil/colors/ColorBar.h>
#include <PluginUtil/colors/coColorMap.h>
#include <cover/coVRPlugin.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Group.h>
#include <cover/ui/Menu.h>
#include <cover/ui/SelectionList.h>
#include <utils/read/csv/csv.h>

// std
#include <string>
#include <unordered_map>

// osg
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Switch>

// ui
#include "ui/simulation/BaseSimulationUI.h"

using namespace opencover::utils::read;

class SimulationSystem final : public core::interface::ISystem {
 public:
  SimulationSystem(opencover::coVRPlugin *plugin, opencover::ui::Menu *parentMenu,
                   osg::ref_ptr<osg::Switch> parent);
  ~SimulationSystem() override;

  void init() override;
  void enable(bool on) override;
  void update() override;
  void updateTime(int timestep) override;
  bool isEnabled() const override { return m_enabled; }
  void preFrame();

 private:
  typedef std::unordered_map<int, std::string> IDLookupTable;

  enum class Scenario {
    status_quo,
    future_ev,
    future_ev_pv,
    optimized_bigger_awz,
    optimized,
    NUM_SCENARIOS
  };

  enum class EnergyGridType { PowerGrid, HeatingGrid, NUM_ENERGY_TYPES };

  struct ColorMapMenu {
    opencover::ui::Menu *menu;
    std::unique_ptr<opencover::CoverColorBar> selector;
  };

  struct EnergySimulation {
    const std::string name;
    const EnergyGridType type;
    opencover::ui::Button *simulationUIBtn = nullptr;
    opencover::ui::SelectionList *scalarSelector = nullptr;
    osg::ref_ptr<osg::MatrixTransform> group = nullptr;
    std::shared_ptr<core::interface::IEnergyGrid> grid;
    std::shared_ptr<core::simulation::Simulation> sim;
    std::unique_ptr<BaseSimulationUI<core::interface::IEnergyGrid>> simUI;
    std::map<std::string, ColorMapMenu> colorMapRegistry;
  };

  static constexpr int const getEnergyGridTypeIndex(EnergyGridType type) {
    return static_cast<int>(type);
  }
  static constexpr int const getScenarioIndex(Scenario scenario) {
    return static_cast<int>(scenario);
  }

  std::string getScenarioName(Scenario scenario);

  void setAnimationTimesteps(size_t maxTimesteps, const void *who);
  void initSimMenu(opencover::ui::Menu *parent);
  void initSimUI(opencover::ui::Menu *parent);
  void initPowerGridUI(opencover::ui::Menu *parent,
                       const std::vector<std::string> &tablesToSkip = {});
  void initEnergyGridUI();
  void initEnergyGridColorMaps();
  void switchEnergyGrid(EnergyGridType grid);
  void updateEnergyGridColorMapInShader(const opencover::ColorMap &map,
                                        EnergyGridType grid);
  void initColorMap();
  void updateEnergyGridShaderData(EnergySimulation &energyGrid);
  void initGrid();
  void addEnergyGridToGridSwitch(osg::ref_ptr<osg::Group> energyGridGroup);

  // TODO: use another interface here
  /* #region POWERGRID */
  void initPowerGridStreams();

  std::vector<grid::PointsMap> createPowerGridPoints(
      opencover::utils::read::CSVStream &stream, size_t &numPoints,
      const float &sphereRadius, const std::vector<IDLookupTable> &busNames);
  osg::ref_ptr<grid::Line> createLine(const std::string &name, int &from,
                                      const std::string &geoBuses, grid::Data &data,
                                      const std::vector<grid::PointsMap> &points);
  void processGeoBuses(grid::Indices &indices, int &from,
                       const std::string &geoBuses,
                       grid::ConnectionDataList &additionalData, grid::Data &data);

  std::pair<std::unique_ptr<grid::Indices>,
            std::unique_ptr<grid::ConnectionDataList>>
  getPowerGridIndicesAndOptionalData(CSVStream &stream, const size_t &numPoints);

  std::pair<std::vector<grid::Lines>, std::vector<grid::ConnectionDataList>>
  getPowerGridLines(CSVStream &stream, const std::vector<grid::PointsMap> &points);

  std::vector<IDLookupTable> retrieveBusNameIdMapping(CSVStream &stream);

  bool checkBoxSelection_powergrid(const std::string &tableName,
                                   const std::string &paramName);
  void helper_getAdditionalPowerGridPointData_addData(
      int busId, grid::PointDataList &additionalData, const grid::Data &data);
  void helper_getAdditionalPowerGridPointData_handleDuplicate(
      std::string &name, std::map<std::string, uint> &duplicateMap);
  std::unique_ptr<grid::PointDataList> getAdditionalPowerGridPointData(
      const std::size_t &numOfBus);
  void applySimulationDataToPowerGrid(const std::string &simPath);
  void updatePowerGridSelection(bool on);
  void updatePowerGridConfig(const std::string &tableName, const std::string &name,
                             bool on);
  void rebuildPowerGrid();
  void initPowerGrid();
  void buildPowerGrid();
  /* #endregion*/

  /* #region HEATINGGRID */
  void initHeatingGridStreams();
  void initHeatingGrid();
  void buildHeatingGrid();
  void readSimulationDataStream(CSVStream &heatingSimStream);
  void applySimulationDataToHeatingGrid();
  void readHeatingGridStream(CSVStream &heatingStream);
  std::vector<int> createHeatingGridIndices(
      const std::string &pointName,
      const std::string &connectionsStrWithCommaDelimiter,
      grid::ConnectionDataList &additionalData);

  osg::ref_ptr<grid::Line> createHeatingGridLine(
      const grid::Points &points, osg::ref_ptr<grid::Point> from,
      const std::string &connectionsStrWithCommaDelimiter,
      grid::ConnectionDataList &additionalData);
  std::pair<grid::Points, grid::Data> createHeatingGridPointsAndData(
      CSVStream &heatingStream, std::map<int, std::string> &connectionStrings);
  grid::Lines createHeatingGridLines(
      const grid::Points &points,
      const std::map<int, std::string> &connectionStrings,
      grid::ConnectionDataList &additionalData);
  osg::ref_ptr<grid::Point> searchHeatingGridPointById(const grid::Points &points,
                                                       int id);
  /* #endregion */

  /* #region COOLINGGRID */
  void buildCoolingGrid();
  /* #endregion */
  /* #endregion*/

  std::unique_ptr<opencover::CoverColorBar> m_vmPuColorMap;

  // Simulation UI
  opencover::coVRPlugin *m_plugin;
  opencover::ui::Menu *m_simulationMenu;
  opencover::ui::Group *m_energygridGroup;
  opencover::ui::ButtonGroup *m_energygridBtnGroup;
  opencover::ui::ButtonGroup *m_scenarios;
  opencover::ui::Button *m_liftGrids;
  opencover::ui::Button *m_status_quo;
  opencover::ui::Button *m_future_ev;
  opencover::ui::Button *m_future_ev_pv;
  opencover::ui::Button *m_rule_base_bigger_hp;
  opencover::ui::Button *m_rule_based;
  opencover::ui::Button *m_optimized_bigger_awz;
  opencover::ui::Button *m_optimized;

  // Powergrid UI
  opencover::ui::Menu *m_powerGridMenu;
  opencover::ui::Button *m_updatePowerGridSelection;
  std::map<opencover::ui::Menu *, std::vector<opencover::ui::Button *>>
      m_powerGridCheckboxes;
  std::unique_ptr<opencover::config::Array<bool>> m_powerGridSelectionPtr;

  // Heatgrid UI
  // opencover::ui::Menu *m_heatGridMenu = nullptr;

  // // Coolinggrid UI
  // opencover::ui::Menu *m_coolingGridMenu = nullptr;
  //
  //
  osg::ref_ptr<osg::Switch> m_gridSwitch;

  std::array<EnergySimulation,
             static_cast<std::size_t>(EnergyGridType::NUM_ENERGY_TYPES)>
      m_energyGrids;
  StreamMap m_powerGridStreams;
  StreamMap m_heatingGridStreams;
  std::vector<double> m_offset;
  std::string m_powerGridDir;

  bool m_enabled;
};
