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
#include "presentation/grid.h"

// ui
#include "lib/core/interfaces/ISolarPanel.h"
#include "lib/core/simulation/object.h"
#include "lib/core/simulation/power.h"
#include "lib/core/simulation/simulation.h"
#include "ui/citygml/CityGMLDeviceSensor.h"
#include "ui/historic/Device.h"
#include "ui/historic/DeviceSensor.h"
#include "ui/ennovatis/EnnovatisDeviceSensor.h"
#include "ui/simulation/HeatingSimulationUI.h"
#include "ui/simulation/PowerSimulationUI.h"

// presentation
#include "presentation/PrototypeBuilding.h"
#include "presentation/SolarPanel.h"

// core
#include <cover/coTUIListener.h>
// #include <lib/core/simulation/grid.h>
#include <lib/core/simulation/heating.h>
#include <lib/core/interfaces/IEnergyGrid.h>
#include <lib/core/utils/color.h>
#include <lib/core/utils/osgUtils.h>

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
  enum Components { Strom, Waerme, Kaelte };
  enum class Scenario {
    status_quo,
    future_ev,
    future_ev_pv,
    optimized_bigger_awz,
    optimized,
    NUM_SCENARIOS
  };
  enum class EnergyGridType {
    PowerGrid,
    HeatingGrid,
    // PowerGridSonder,
    NUM_ENERGY_TYPES
  };
  // enum EnergyGrids { PowerGrid, HeatingGrid, CoolingGrid, NUM_ENERGY_GRIDS };

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
  void setComponent(Components c);
  static EnergyPlugin *instance() {
    if (!m_plugin) m_plugin = new EnergyPlugin;
    return m_plugin;
  };

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

  typedef NameMapVector<float> FloatMap;
  typedef NameMapVector<energy::DeviceSensor::ptr> DeviceList;
  typedef NameMap<CSVStream> CSVStreamMap;
  typedef std::unique_ptr<CSVStreamMap> CSVStreamMapPtr;

  typedef std::vector<std::unique_ptr<core::interface::ISolarPanel>> SolarPanelList;
  /* #endregion */

  struct ColorMapMenu {
    opencover::ui::Menu *menu;
    std::unique_ptr<opencover::CoverColorBar> selector;
  };

  auto getEnergyGridTypeIndex(EnergyGridType type) { return static_cast<int>(type); }
  auto getScenarioIndex(Scenario scenario) { return static_cast<int>(scenario); }

//   std::string getScenarioName(int value);
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
  CSVStreamMap getCSVStreams(const boost::filesystem::path &dirPath);
  void setAnimationTimesteps(size_t maxTimesteps, const void *who);
  void initOverview();
  void initUI();
  /* #endregion */

  /* #region HISTORIC */
  void helper_initTimestepGrp(size_t maxTimesteps,
                              osg::ref_ptr<osg::Group> &timestepGroup);
  void helper_initTimestepsAndMinYear(size_t &maxTimesteps, int &minYear,
                                      const std::vector<std::string> &header);
  void helper_projTransformation(bool mapdrape, PJ *P, PJ_COORD &coord,
                                 energy::DeviceInfo &deviceInfoPtr,
                                 const double &lat, const double &lon);
  void helper_handleEnergyInfo(size_t maxTimesteps, int minYear,
                               const CSVStream::CSVRow &row,
                               energy::DeviceInfo &deviceInfoPtr);
  bool loadDBFile(const std::string &fileName, const ProjTrans &projTrans);
  bool loadDB(const std::string &path, const ProjTrans &projTrans);
  void reinitDevices(int comp);
  void initHistoricUI();
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
  /**
   * Initializes the Ennovatis buildings.
   *
   * This function takes a `DeviceList` object as a parameter and returns a
   * `std::unique_ptr` to a `const_buildings` object. The `const_buildings`
   * object represents the initialized Ennovatis buildings.
   *
   * TODO: apply this while parsing the JSON file
   * @param deviceList The list of devices. Make sure map is sorted.
   * @return A unique pointer to buildings which have ne matching device.
   */
  std::unique_ptr<const_buildings> updateEnnovatisBuildings(
      const DeviceList &deviceList);
  /**
   * Loads Ennovatis channelids from the specified JSON file into cache.
   *
   * @param pathToJSON The path to the JSON file which contains the channelids
   * for REST-calls.
   * @return True if the data was successfully loaded, false otherwise.
   */
  bool loadChannelIDs(const std::string &pathToJSON, const std::string &pathToCSV);
  /* #endregion */

  /* #region CITYGML */
  void initCityGMLUI();
  void initCityGMLColorMap();
  void addSolarPanelsToCityGML(const boost::filesystem::path &dirPath);
  //   void enableCityGML(bool on);
  void enableCityGML(bool on, bool updateColorMap = true);
  void addCityGMLObjects(osg::ref_ptr<osg::Group> citygmlGroup);
  void addCityGMLObject(const std::string &name,
                        osg::ref_ptr<osg::Group> citygmlObjGroup);
  void saveCityGMLObjectDefaultStateSet(const std::string &name,
                                        const Geodes &citygmlGeodes);
  void restoreCityGMLDefaultStatesets();
  void restoreGeodesStatesets(CityGMLDeviceSensor &sensor, const std::string &name,
                              const Geodes &citygmlGeodes);
  void transformCityGML(const osg::Vec3 &translation, const osg::Quat &rotation,
                        const osg::Vec3 &scale = osg::Vec3(1.0, 1.0, 1.0));
  osg::Vec3 getCityGMLTranslation() const;
  /* #endregion */

  /* #region SIMULATION */
  //   struct EnergySimulation {
  //     const std::string name;
  //     const std::string species;
  //     const std::string unit;
  //     const EnergyGridType type;
  //     opencover::ui::Button *simulationUIBtn = nullptr;
  //     opencover::ui::Menu *menu = nullptr;
  //     opencover::ui::SelectionList *scalarSelector = nullptr;
  //     osg::ref_ptr<osg::MatrixTransform> group = nullptr;
  //     std::shared_ptr<core::interface::IEnergyGrid> grid;
  //     std::shared_ptr<Simulation> sim;
  //     std::unique_ptr<BaseSimUI> simUI;
  //     std::map<std::string, ColorMapMenu> colorMapRegistry;
  //   };

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

  //   struct InfluxData {
  //     sys_time<minutes> time;  // RFC3339 format
  //     double value;
  //     std::string field;
  //     std::string measurement;
  //     std::string district;
  //     bool hkw;
  //     bool new_building;
  //     bool pv_penetration;
  //   };

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
  std::unique_ptr<FloatMap> getInlfuxDataFromCSV(CSVStream &stream, float &max,
                                                 float &min, float &sum,
                                                 int &timesteps);

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

  auto readStaticCampusData(CSVStream &stream, float &max, float &min, float &sum);
  auto readStaticPowerData(CSVStream &stream, float &max, float &min, float &sum);
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
  void applyInfluxCSVToCityGML(const std::string &filePath,
                               bool updateColorMap = true);
  void applyInfluxArrowToCityGML();
  void applyStaticDataToCityGML(const std::string &filePath,
                                bool updateColorMap = true);
  void applyStaticDataCampusToCityGML(const std::string &filePath,
                                      bool updateColorMap = true);
  void applySimulationDataToPowerGrid(const std::string &simPath);
  void updatePowerGridSelection(bool on);
  void updatePowerGridConfig(const std::string &tableName, const std::string &name,
                             bool on);
  void rebuildPowerGrid();
  void initPowerGrid();
  void initPowerGridUI(const std::vector<std::string> &tablesToSkip = {});
  void buildPowerGrid();
  /* #region PV*/
  std::pair<std::map<std::string, PVData>, float> loadPVData(CSVStream &pvStream);
  void processPVRow(const CSVStream::CSVRow &row,
                    std::map<std::string, PVData> &pvDataMap, float &maxPVIntensity);
  osg::ref_ptr<osg::Node> readPVModel(const boost::filesystem::path &modelDir,
                                      const std::string &nameInModelDir);
  void initPV(osg::ref_ptr<osg::Node> masterPanel,
              const std::map<std::string, PVData> &pvDataMap, float maxPVIntensity);
  void processPVDataMap(
      const std::vector<CoreUtils::osgUtils::instancing::GeometryData>
          &masterGeometryData,
      const std::map<std::string, PVData> &pvDataMap, float maxPVIntensity);
  void processSolarPanelDrawable();
  SolarPanel createSolarPanel(
      const std::string &name, osg::ref_ptr<osg::Group> parent,
      const std::vector<CoreUtils::osgUtils::instancing::GeometryData>
          &masterGeometryData,
      const osg::Matrix &matrix, const osg::Vec4 &colorIntensity);

  struct SolarPanelConfig {
    std::string name;
    float zOffset;
    float numMaxPanels;
    float panelWidth;
    float panelHeight;
    osg::Vec4 colorIntensity;
    osg::Matrixd rotation;
    osg::ref_ptr<osg::Group> parent;
    osg::ref_ptr<osg::Geode> geode;
    std::vector<CoreUtils::osgUtils::instancing::GeometryData> masterGeometryData;
    bool valid() const { return parent && geode && !masterGeometryData.empty(); }
  };

  void processSolarPanelDrawable(SolarPanelList &solarPanels,
                                 const SolarPanelConfig &config);
  void processSolarPanelDrawables(
      const PVData &data, const std::vector<osg::ref_ptr<osg::Node>> drawables,
      SolarPanelList &solarPanels, SolarPanelConfig &config);
  /* #endregion*/
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
  std::unique_ptr<opencover::CoverColorBar> m_cityGmlColorMap;
  // TODO: remove this later
  std::unique_ptr<opencover::CoverColorBar> m_vmPuColorMap;

  // historical
  opencover::ui::Button *ShowGraph = nullptr;
  opencover::ui::ButtonGroup *componentGroup = nullptr;
  opencover::ui::Menu *componentList = nullptr;
  opencover::ui::Button *StromBt = nullptr;
  opencover::ui::Button *WaermeBt = nullptr;
  opencover::ui::Button *KaelteBt = nullptr;

  // ennovatis UI
  opencover::ui::SelectionList *m_ennovatisSelectionsList = nullptr;
  opencover::ui::Menu *m_ennovatisMenu = nullptr;
  opencover::ui::EditField *m_ennovatisFrom = nullptr;
  opencover::ui::EditField *m_ennovatisTo = nullptr;
  opencover::ui::Button *m_ennovatisUpdate = nullptr;
  opencover::ui::SelectionList *m_ennovatisChannelList = nullptr;
  opencover::ui::SelectionList *m_enabledEnnovatisDevices = nullptr;

  // citygml UI
  opencover::ui::Menu *m_cityGMLMenu = nullptr;
  opencover::ui::Button *m_cityGMLEnableInfluxCSV = nullptr;
  opencover::ui::Button *m_cityGMLEnableInfluxArrow = nullptr;
  opencover::ui::Button *m_PVEnable = nullptr;
  //   opencover::ui::Button *m_cityGMLDisableBuildings = nullptr;
  opencover::ui::EditField *m_cityGMLX = nullptr, *m_cityGMLY = nullptr,
                           *m_cityGMLZ = nullptr;

  // Simulation UI
  opencover::ui::Menu *m_simulationMenu = nullptr;
  opencover::ui::Group *m_energygridGroup = nullptr;
  opencover::ui::ButtonGroup *m_energygridBtnGroup = nullptr;
  opencover::ui::Button *m_liftGrids = nullptr;
  opencover::ui::Button *m_staticPower = nullptr;
  opencover::ui::Button *m_staticCampusPower = nullptr;
  opencover::ui::ButtonGroup *m_scenarios = nullptr;
  opencover::ui::Button *m_status_quo = nullptr;
  opencover::ui::Button *m_future_ev = nullptr;
  opencover::ui::Button *m_future_ev_pv = nullptr;
  opencover::ui::Button *m_rule_base_bigger_hp = nullptr;
  opencover::ui::Button *m_rule_based = nullptr;
  opencover::ui::Button *m_optimized_bigger_awz = nullptr;
  opencover::ui::Button *m_optimized = nullptr;

  // opencover::ui::Button *m_powerGridBtn = nullptr;
  // opencover::ui::Button *m_heatingGridBtn = nullptr;
  // opencover::ui::Button *m_coolingGridBtn = nullptr;

  // Powergrid UI
  opencover::ui::Menu *m_powerGridMenu = nullptr;
  opencover::ui::Button *m_updatePowerGridSelection = nullptr;
  std::map<opencover::ui::Menu *, std::vector<opencover::ui::Button *>>
      m_powerGridCheckboxes;
  std::unique_ptr<config::Array<bool>> m_powerGridSelectionPtr = nullptr;

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
  osg::ref_ptr<osg::Sequence> m_sequenceList;
  osg::ref_ptr<osg::MatrixTransform> m_Energy;
  osg::ref_ptr<osg::Group> m_cityGML;
  osg::ref_ptr<osg::Group> m_pvGroup;
  std::map<std::string, Geodes> m_cityGMLDefaultStatesets;
  std::map<std::string, std::unique_ptr<CityGMLDeviceSensor>> m_cityGMLObjs;

  CSVStreamMap m_powerGridStreams;
  CSVStreamMap m_heatingGridStreams;

  // std::array<std::unique_ptr<BaseSimUI>, NUM_ENERGY_GRIDS> m_simUIs;
  //   std::unique_ptr<SolarPanelList> m_solarPanels;
  SolarPanelList m_solarPanels;
  //   std::unique_ptr<SolarPanel> m_solarPanel;
  //
  std::vector<double> m_offset;
  std::string m_powerGridDir;
  float rad, scaleH;
  int m_selectedComp = 0;
};

#endif
