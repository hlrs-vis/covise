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

#ifndef _Energy_PLUGIN_H
#define _Energy_PLUGIN_H

// presentation
#include "app/CityGMLSystem.h"
#include "presentation/grid.h"

// ui
#include "lib/core/simulation/simulation.h"
#include "ui/simulation/HeatingSimulationUI.h"
#include "ui/simulation/PowerSimulationUI.h"

// presentation
#include "presentation/PrototypeBuilding.h"

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

// ennovatis
#include <lib/ennovatis/building.h>
#include <lib/ennovatis/rest.h>

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
  enum class System { CityGML, NUM_SYSTEMS };
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

  template <typename T>
  using NameMap = std::map<std::string, T>;
  template <typename T>
  using NameMapPtr = NameMap<std::unique_ptr<T>>;
  template <typename T>
  using NameMapVector = NameMap<std::vector<T>>;
  template <typename T>
  using NameMapVectorPtr = NameMapPtr<std::vector<T>>;

  /* #endregion */

  /* #region typedef */
  typedef std::unordered_map<int, std::string> IDLookupTable;
  typedef BaseSimulationUI<core::interface::IEnergyGrid> BaseSimUI;
  typedef HeatingSimulationUI<core::interface::IEnergyGrid> HeatingSimUI;
  typedef PowerSimulationUI<core::interface::IEnergyGrid> PowerSimUI;
  typedef const ennovatis::Building *building_const_ptr;
  typedef const ennovatis::Buildings *buildings_const_Ptr;
  typedef std::vector<building_const_ptr> const_buildings;
  typedef std::map<energy::Device::ptr, building_const_ptr> DeviceBuildingMap;

//   typedef NameMapVector<float> FloatMap;
  typedef NameMapVector<energy::DeviceSensor::ptr> DeviceList;

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

  CityGMLSystem *CityGML() {
    if (m_systems[getSystemIndex(System::CityGML)]) {
      return dynamic_cast<CityGMLSystem *>(
          m_systems[getSystemIndex(System::CityGML)].get());
    }
    return nullptr;
  }

  std::string getScenarioName(Scenario scenario);
  void preFrame() override;  // update colormaps

  /* #region GENERAL */
  inline void checkEnergyTab() {
    assert(m_EnergyTab && "EnergyTab must be initialized before");
  }
  bool isActiv(osg::ref_ptr<osg::Switch> switchToCheck,
               osg::ref_ptr<osg::Group> group);
  void switchTo(const osg::ref_ptr<osg::Node> child,
                osg::ref_ptr<osg::Switch> parent);
  std::pair<PJ *, PJ_COORD> initProj();
  void projTransLatLon(float &lat, float &lon);
  void setAnimationTimesteps(size_t maxTimesteps, const void *who);
  void initOverview();
  void initUI();
  void initSystems();
  /* #endregion */

  /* #region ENNOVATIS */
  void initRESTRequest();
  void initEnnovatisUI();
  void selectEnabledDevice();
  void setEnnovatisChannelGrp(ennovatis::ChannelGroup group);
  void setRESTDate(const std::string &toSet, bool isFrom);
  void updateEnnovatis();
  void updateEnnovatisChannelGrp();
  void initEnnovatisDevices();
  bool updateChannelIDsFromCSV(const std::string &pathToCSV);
  CylinderAttributes getCylinderAttributes();

  //   /**
  //    * Initializes the Ennovatis buildings.
  //    *
  //    * This function takes a `DeviceList` object as a parameter and returns a
  //    * `std::unique_ptr` to a `const_buildings` object. The `const_buildings`
  //    * object represents the initialized Ennovatis buildings.
  //    *
  //    * TODO: apply this while parsing the JSON file
  //    * @param deviceList The list of devices. Make sure map is sorted.
  //    * @return A unique pointer to buildings which have ne matching device.
  //    */
  //   std::unique_ptr<const_buildings> updateEnnovatisBuildings(
  //       const DeviceList &deviceList);

  //   /**
  //    * Loads Ennovatis channelids from the specified JSON file into cache.
  //    *
  //    * @param pathToJSON The path to the JSON file which contains the channelids
  //    * for REST-calls.
  //    * @return True if the data was successfully loaded, false otherwise.
  //    */
  //   bool loadChannelIDs(const std::string &pathToJSON, const std::string
  //   &pathToCSV);
  /* #endregion */

  /* #region SIMULATION */

  struct EnergySimulation {
    const std::string name;
    const EnergyGridType type;
    opencover::ui::Button *simulationUIBtn = nullptr;
    // opencover::ui::Menu *menu = nullptr;
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
  opencover::coTUITab *m_coEnergyTabPanel = nullptr;
  opencover::ui::Menu *m_EnergyTab = nullptr;
  opencover::ui::Menu *m_controlPanel = nullptr;
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

  ennovatis::BuildingsPtr m_buildings;
  DeviceList m_SDlist;
  std::shared_ptr<ennovatis::rest_request> m_req;

  // current selected channel group
  std::shared_ptr<ennovatis::ChannelGroup> m_channelGrp;

  // not necessary but better for debugging
  DeviceBuildingMap m_devBuildMap;
  std::vector<std::unique_ptr<EnnovatisDeviceSensor>> m_ennovatisDevicesSensors;
  osg::ref_ptr<osg::Group> m_ennovatis;

  // switch used to toggle between ennovatis, db and citygml data
  osg::ref_ptr<osg::Switch> m_switch;
  osg::ref_ptr<osg::Switch> m_grid;
  osg::ref_ptr<osg::MatrixTransform> m_Energy;

  opencover::utils::read::StreamMap m_powerGridStreams;
  opencover::utils::read::StreamMap m_heatingGridStreams;

  static constexpr std::size_t NUM_SYSTEMS =
      static_cast<std::size_t>(System::NUM_SYSTEMS);
  std::array<std::unique_ptr<core::interface::ISystem>, NUM_SYSTEMS> m_systems;
  std::vector<double> m_offset;
  std::string m_powerGridDir;
  float rad, scaleH;
  int m_selectedComp = 0;
};

#endif
