/****************************************************************************\
 **                                                          (C)2024 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading building energy data        **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern, Marko Djuric                                       **
 **                                                                        **
 ** History:                                                               **
 **  2024  v1                                                              **
 **  Marko Djuric 02.2024: add ennovatis client                            **
 **                                                                        **
\****************************************************************************/
#pragma once
// presentation
#include "app/CityGMLSystem.h"
#include "app/EnnovatisSystem.h"
#include "presentation/grid.h"

// ui
#include "lib/core/simulation/simulation.h"
#include "ui/simulation/HeatingSimulationUI.h"
#include "ui/simulation/PowerSimulationUI.h"

// core
#include <cover/coTUIListener.h>
#include <lib/core/simulation/heating.h>
#include <lib/core/interfaces/IEnergyGrid.h>
#include <lib/core/utils/color.h>
#include <lib/core/utils/osgUtils.h>
#include <lib/core/interfaces/ISystem.h>

// cover
#include <OpenConfig/array.h>
#include <PluginUtil/coSensor.h>
#include <PluginUtil/colors/coColorMap.h>
#include <PluginUtil/colors/coColorBar.h>
#include <PluginUtil/colors/ColorBar.h>

#include <cover/VRViewer.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Group.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <OpenConfig/covconfig/array.h>

#include <gdal_priv.h>
#include <proj.h>
#include <util/coTypes.h>
#include <util/common.h>
#include <utils/read/csv/csv.h>

// boost
#include <boost/filesystem.hpp>

// osg
#include <cstddef>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Sequence>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osg/ref_ptr>

// std
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>

using namespace core::simulation;
namespace COVERUtils = opencover::utils;
namespace CoreUtils = core::utils;

class EnergyPlugin : public opencover::coVRPlugin,
                     public opencover::ui::Owner,
                     public opencover::coTUIListener {
  enum class Scenario {
    status_quo,
    future_ev,
    future_ev_pv,
    optimized_bigger_awz,
    optimized,
    NUM_SCENARIOS
  };
  enum class EnergyGridType { PowerGrid, HeatingGrid, NUM_ENERGY_TYPES };
  enum class System { CityGML, Ennovatis, NUM_SYSTEMS };
  struct ProjTrans {
    std::string projFrom;
    std::string projTo;
  };

 public:
  EnergyPlugin();
  ~EnergyPlugin();
  EnergyPlugin(const EnergyPlugin &) = delete;
  void operator=(const EnergyPlugin &) = delete;

  bool init() override;
  bool update() override;
  void setTimestep(int t) override;

 private:
  /* #region using */
  using Geodes = CoreUtils::osgUtils::Geodes;
  using CSVStream = COVERUtils::read::CSVStream;
  /* #endregion */

  /* #region typedef */
  typedef std::unordered_map<int, std::string> IDLookupTable;
  typedef BaseSimulationUI<core::interface::IEnergyGrid> BaseSimUI;
  typedef HeatingSimulationUI<core::interface::IEnergyGrid> HeatingSimUI;
  typedef PowerSimulationUI<core::interface::IEnergyGrid> PowerSimUI;
  /* #endregion */

  struct ColorMapMenu {
    opencover::ui::Menu *menu;
    std::unique_ptr<opencover::CoverColorBar> selector;
  };

  static constexpr int const getEnergyGridTypeIndex(EnergyGridType type) {
    return static_cast<int>(type);
  }
  static constexpr int const getScenarioIndex(Scenario scenario) {
    return static_cast<int>(scenario);
  }
  static constexpr int const getSystemIndex(System system) {
    return static_cast<int>(system);
  }

  template <typename T>
  T *getSystem(System system) {
    auto it = m_systems.find(system);
    if (it != m_systems.end()) {
      return dynamic_cast<T *>(it->second.get());
    }
    return nullptr;
  }

  CityGMLSystem *getCityGMLSystem() {
    return getSystem<CityGMLSystem>(System::CityGML);
  }

  EnnovatisSystem *getEnnovatisSystem() {
    return getSystem<EnnovatisSystem>(System::Ennovatis);
  }

  std::string getScenarioName(Scenario scenario);
  void preFrame() override;  // update colormaps

  /* #region GENERAL */
  inline void checkEnergyTab() {
    assert(m_EnergyTab && "EnergyTab must be initialized before");
  }
  std::pair<PJ *, PJ_COORD> initProj();
  void projTransLatLon(float &lat, float &lon);
  void setAnimationTimesteps(size_t maxTimesteps, const void *who);
  void initOverview();
  void initUI();
  void initSystems();
  /* #endregion */

  /* #region SIMULATION */

  struct EnergySimulation {
    const std::string name;
    const EnergyGridType type;
    opencover::ui::Button *simulationUIBtn = nullptr;
    opencover::ui::SelectionList *scalarSelector = nullptr;
    osg::ref_ptr<osg::MatrixTransform> group = nullptr;
    std::shared_ptr<core::interface::IEnergyGrid> grid;
    std::shared_ptr<Simulation> sim;
    std::unique_ptr<BaseSimUI> simUI;
    std::map<std::string, ColorMapMenu> colorMapRegistry;
  };

  void initSimUI();
  void initEnergyGridUI();
  void initEnergyGridColorMaps();
  void switchEnergyGrid(EnergyGridType grid);
  void initSimMenu();
  void updateEnergyGridColorMapInShader(const opencover::ColorMap &map,
                                        EnergyGridType grid);
  void initColorMap();
  void updateEnergyGridShaderData(EnergySimulation &energyGrid);
  void initGrid();
  void addEnergyGridToGridSwitch(osg::ref_ptr<osg::Group> energyGridGroup);

  /* #region POWERGRID */
  void initPowerGridStreams();

  std::vector<grid::PointsMap> createPowerGridPoints(
      CSVStream &stream, size_t &numPoints, const float &sphereRadius,
      const std::vector<IDLookupTable> &busNames);
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
  void initPowerGridUI(const std::vector<std::string> &tablesToSkip = {});
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
      COVERUtils::read::CSVStream &heatingStream,
      std::map<int, std::string> &connectionStrings);
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

  // general
  static EnergyPlugin *m_plugin;
  opencover::ui::Menu *m_EnergyTab = nullptr;
  opencover::ui::Menu *m_controlPanel = nullptr;
  opencover::coTUITab *m_coEnergyTabPanel = nullptr;
  opencover::ui::Button *m_gridControlButton = nullptr;
  opencover::ui::Button *m_energySwitchControlButton = nullptr;

  std::array<EnergySimulation,
             static_cast<std::size_t>(EnergyGridType::NUM_ENERGY_TYPES)>
      m_energyGrids;
  // TODO: remove this later
  std::unique_ptr<opencover::CoverColorBar> m_vmPuColorMap;

  // Simulation UI
  opencover::ui::Menu *m_simulationMenu = nullptr;
  opencover::ui::Group *m_energygridGroup = nullptr;
  opencover::ui::ButtonGroup *m_energygridBtnGroup = nullptr;
  opencover::ui::ButtonGroup *m_scenarios = nullptr;
  opencover::ui::Button *m_liftGrids = nullptr;
  opencover::ui::Button *m_status_quo = nullptr;
  opencover::ui::Button *m_future_ev = nullptr;
  opencover::ui::Button *m_future_ev_pv = nullptr;
  opencover::ui::Button *m_rule_base_bigger_hp = nullptr;
  opencover::ui::Button *m_rule_based = nullptr;
  opencover::ui::Button *m_optimized_bigger_awz = nullptr;
  opencover::ui::Button *m_optimized = nullptr;

  // Powergrid UI
  opencover::ui::Menu *m_powerGridMenu = nullptr;
  opencover::ui::Button *m_updatePowerGridSelection = nullptr;
  std::map<opencover::ui::Menu *, std::vector<opencover::ui::Button *>>
      m_powerGridCheckboxes;
  std::unique_ptr<opencover::config::Array<bool>> m_powerGridSelectionPtr = nullptr;

  // Heatgrid UI
  // opencover::ui::Menu *m_heatGridMenu = nullptr;

  // // Coolinggrid UI
  // opencover::ui::Menu *m_coolingGridMenu = nullptr;

  // current selected channel group
  //   std::shared_ptr<ennovatis::ChannelGroup> m_channelGrp;

  // switch used to toggle between ennovatis, db and citygml data
  osg::ref_ptr<osg::Switch> m_switch;
  osg::ref_ptr<osg::Switch> m_grid;
  osg::ref_ptr<osg::MatrixTransform> m_Energy;

  opencover::utils::read::StreamMap m_powerGridStreams;
  opencover::utils::read::StreamMap m_heatingGridStreams;

  std::map<System, std::unique_ptr<core::interface::ISystem>> m_systems;
  std::vector<double> m_offset;
  std::string m_powerGridDir;
  float rad, scaleH;
  int m_selectedComp = 0;
};
