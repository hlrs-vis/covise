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

#include <CityGMLDeviceSensor.h>
#include <Device.h>
#include <DeviceSensor.h>
#include <EnnovatisDeviceSensor.h>

// core
#include <core/PrototypeBuilding.h>
#include <core/grid.h>
#include <core/interfaces/IEnergyGrid.h>
#include <core/utils/color.h>
#include <core/utils/osgUtils.h>

// cover
#include <OpenConfig/array.h>
#include <PluginUtil/coSensor.h>
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
#include <ennovatis/building.h>
#include <ennovatis/rest.h>
#include <gdal_priv.h>
#include <proj.h>
#include <util/coTypes.h>
#include <util/common.h>
#include <utils/read/csv/csv.h>

// boost
#include <boost/filesystem.hpp>

// osg
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
#include <vector>

class EnergyPlugin : public opencover::coVRPlugin,
                     public opencover::ui::Owner,
                     public opencover::coTUIListener {
  enum Components { Strom, Waerme, Kaelte };
  struct ProjTrans {
    std::string projFrom;
    std::string projTo;
  };

 public:
  EnergyPlugin();
  ~EnergyPlugin();
  EnergyPlugin(const EnergyPlugin &) = delete;
  void operator=(const EnergyPlugin &) = delete;

  bool init();
  bool update();
  void setTimestep(int t);
  void setComponent(Components c);
  static EnergyPlugin *instance() {
    if (!m_plugin) m_plugin = new EnergyPlugin;
    return m_plugin;
  };

 private:
  /* #region using */
  using Geodes = core::utils::osgUtils::Geodes;

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
  typedef const ennovatis::Building *building_const_ptr;
  typedef const ennovatis::Buildings *buildings_const_Ptr;
  typedef std::vector<building_const_ptr> const_buildings;
  typedef std::map<energy::Device::ptr, building_const_ptr> DeviceBuildingMap;

  typedef NameMapVector<float> FloatMap;
  typedef NameMapVector<energy::DeviceSensor::ptr> DeviceList;
  typedef NameMapPtr<utils::read::CSVStream> CSVStreamMap;
  typedef std::unique_ptr<CSVStreamMap> CSVStreamMapPtr;
  /* #endregion */

  /* #region GENERAL */
  void switchTo(const osg::ref_ptr<osg::Node> child);
  void updateColorMap(const covise::ColorMap &map);
  void initColorMap();
  std::pair<PJ *, PJ_COORD> initProj();
  CSVStreamMapPtr getCSVStreams(const boost::filesystem::path &dirPath);
  /* #endregion */

  /* #region HISTORIC */
  void helper_initTimestepGrp(size_t maxTimesteps,
                              osg::ref_ptr<osg::Group> &timestepGroup);
  void helper_initTimestepsAndMinYear(size_t &maxTimesteps, int &minYear,
                                      const std::vector<std::string> &header);
  void helper_projTransformation(bool mapdrape, PJ *P, PJ_COORD &coord,
                                 energy::DeviceInfo::ptr deviceInfoPtr,
                                 const double &lat, const double &lon);
  void helper_handleEnergyInfo(size_t maxTimesteps, int minYear,
                               const opencover::utils::read::CSVStream::CSVRow &row,
                               energy::DeviceInfo::ptr deviceInfoPtr);
  bool loadDBFile(const std::string &fileName, const ProjTrans &projTrans);
  bool loadDB(const std::string &path, const ProjTrans &projTrans);
  void reinitDevices(int comp);
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
  core::CylinderAttributes getCylinderAttributes();
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
  void enableCityGML(bool on);
  void addCityGMLObjects(osg::ref_ptr<osg::Group> citygmlGroup);
  void addCityGMLObject(const std::string &name,
                        osg::ref_ptr<osg::Group> citygmlObjGroup);
  void saveCityGMLObjectDefaultStateSet(const std::string &name,
                                        const Geodes &citygmlGeodes);
  void restoreCityGMLDefaultStatesets();
  void restoreGeodesStatesets(CityGMLDeviceSensor &sensor, const std::string &name,
                              const Geodes &citygmlGeodes);
  /* #endregion */

  /* #region SIMULATION */
  void initPowerGridStreams();

  std::unique_ptr<FloatMap> getInlfuxDataFromCSV(utils::read::CSVStream &stream,
                                                 float &max, float &min, float &sum,
                                                 int &timesteps);
  std::unique_ptr<core::grid::Points> createPowerGridPoints(
      utils::read::CSVStream &stream, const float &sphereRadius,
      const std::vector<std::string> &busNames);
  std::pair<std::unique_ptr<core::grid::Indices>,
            std::unique_ptr<core::grid::DataList>>
  getPowerGridIndicesAndOptionalData(utils::read::CSVStream &stream,
                                     const size_t &numBus);
  std::unique_ptr<std::vector<std::string>> getBusNames(
      utils::read::CSVStream &stream);

  bool checkBoxSelection_powergrid(const std::string &tableName,
                                   const std::string &paramName);
  void helper_getAdditionalPowerGridPointData_addData(
      int busId, core::grid::DataList &additionalData, const core::grid::Data &data);
  void helper_getAdditionalPowerGridPointData_handleDuplicate(
      std::string &name, std::map<std::string, uint> &duplicateMap);
  std::unique_ptr<core::grid::DataList> getAdditionalPowerGridPointData(
      const std::size_t &numOfBus);

  void initSimUI();
  void initGrid();
  void applyStaticInfluxToCityGML(const std::string &filePath);
  void updatePowerGridSelection(bool on);
  void updatePowerGridConfig(const std::string &tableName, const std::string &name,
                             bool on);
  void rebuildPowerGrid();
  void initPowerGrid();
  void initPowerGridUI(const std::vector<std::string> &tablesToSkip = {});
  void buildPowerGrid();
  void buildHeatingGrid();
  void buildCoolingGrid();

  /* #endregion*/

  // general
  static EnergyPlugin *m_plugin;
  opencover::coTUITab *coEnergyTab = nullptr;
  opencover::ui::Menu *EnergyTab = nullptr;
  opencover::ui::Group *m_colorMapGroup = nullptr;
  opencover::ui::Slider *m_minAttribute = nullptr;
  opencover::ui::Slider *m_maxAttribute = nullptr;
  opencover::ui::Slider *m_numSteps = nullptr;
  std::unique_ptr<covise::ColorMapSelector> m_colorMapSelector = nullptr;

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
  opencover::ui::Button *m_cityGMLEnable = nullptr;

  // Simulation UI
  opencover::ui::Menu *m_simulationMenu = nullptr;

  // Powergrid UI
  opencover::ui::Menu *m_powerGridMenu = nullptr;
  opencover::ui::Button *m_updatePowerGridSelection = nullptr;
  std::map<opencover::ui::Menu *, std::vector<opencover::ui::Button *>>
      m_powerGridCheckboxes;
  std::unique_ptr<config::Array<bool>> m_powerGridSelectionPtr = nullptr;

  // Heatgrid UI
  opencover::ui::Menu *m_heatGridMenu = nullptr;

  // Coolinggrid UI
  opencover::ui::Menu *m_coolingGridMenu = nullptr;

  float rad, scaleH;
  int m_selectedComp = 0;
  std::vector<double> m_offset;

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
  osg::ref_ptr<osg::Sequence> m_sequenceList;
  osg::ref_ptr<osg::MatrixTransform> m_Energy;
  osg::ref_ptr<osg::Group> m_cityGML;
  std::map<std::string, Geodes> m_cityGMLDefaultStatesets;
  std::map<std::string, std::unique_ptr<CityGMLDeviceSensor>> m_cityGMLObjs;

  std::shared_ptr<core::utils::color::ColorMapExtended> m_colorMap;
  std::unique_ptr<core::interface::IEnergyGrid> m_powerGrid;
  CSVStreamMapPtr m_powerGridStreams;
};

#endif
