#pragma once

#include <OpenConfig/file.h>
#include <cover/coVRPlugin.h>

#include <lib/core/interfaces/ISystem.h>
#include <lib/core/simulation/simulation.h>
#include <lib/core/simulation/unitmap.h>
#include <lib/core/simulation/power.h>
#include <lib/core/utils/osgUtils.h>
#include <utils/read/csv/csv.h>

#include <boost/filesystem.hpp>
#include <memory>
#include <osg/ClipNode>
#include <osg/Group>
#include <osg/Switch>
#include <string>
#include <vector>

#include "app/osg/presentation/SolarPanel.h"
#include "app/osg/ui/citygml/CityGMLDeviceSensor.h"
#include "app/cover/ui/CityGMLUI.h"

/**
 * @brief A list of unique pointers to ISolarPanel interfaces.
 *
 * This typedef defines a container for managing multiple solar panel objects,
 * ensuring unique ownership semantics for each ISolarPanel instance.
 */
typedef std::vector<std::unique_ptr<core::interface::ISolarPanel>> SolarPanelList;

/**
 * @class CityGMLSystem
 * @brief Manages the CityGML system integration, visualization, and simulation within OpenCOVER.
 *
 * This class provides functionality to initialize, enable, update, and manage CityGML objects,
 * solar panels, and related UI components. It handles loading and processing of PV (photovoltaic)
 * data, influx data from CSV files, and static power data for campus and city objects.
 * The class also manages color maps for visualization, transformation of objects, and state sets
 * for CityGML devices and sensors.
 *
 * @note Instances of this class are non-copyable and non-movable.
 *
 * @see core::interface::ISystem
 */
class CityGMLSystem final : public core::interface::ISystem {
 public:
  CityGMLSystem(opencover::coVRPlugin *plugin, opencover::ui::Menu *parentMenu,
                osg::ref_ptr<osg::ClipNode> rootGroup, osg::ref_ptr<osg::Switch> parent);
  virtual ~CityGMLSystem();
  CityGMLSystem(const CityGMLSystem &) = delete;
  CityGMLSystem &operator=(const CityGMLSystem &) = delete;
  CityGMLSystem &operator=(CityGMLSystem &&) = delete;
  CityGMLSystem(CityGMLSystem &&) = delete;

  virtual void init() override;
  virtual void enable(bool on) override;
  virtual bool isEnabled() const override;
  virtual void update() override;
  virtual void updateTime(int timestep) override;

  void updateInfluxColorMaps(
      float min, float max,
      std::shared_ptr<core::simulation::Simulation> powerSimulation,
      const std::string &colormapName, const std::string &species = "Residuallast",
      const std::string &unit = "MW");

 private:
  void initUICallbacks();
  void initPV(
      osg::ref_ptr<osg::Node> masterPanel,
      const std::map<std::string, core::simulation::power::PVData> &pvDataMap,
      float maxPVIntensity);

  std::pair<std::map<std::string, core::simulation::power::PVData>, float>
  loadPVData(opencover::utils::read::CSVStream &pvStream);

  std::unique_ptr<SolarPanel> createSolarPanel(
      const std::string &name, osg::ref_ptr<osg::Group> parent,
      const std::vector<core::utils::osgUtils::instancing::GeometryData>
          &masterGeometryData,
      const osg::Matrix &matrix, const Color &colorIntensity);

  void processPVRow(
      const opencover::utils::read::CSVStream::CSVRow &row,
      std::map<std::string, core::simulation::power::PVData> &pvDataMap,
      float &maxPVIntensity);
  void processSolarPanelDrawable(SolarPanelList &solarPanels,
                                 const SolarPanelConfig &config);
  void processSolarPanelDrawables(
      const core::simulation::power::PVData &data,
      const std::vector<osg::ref_ptr<osg::Node>> drawables,
      SolarPanelList &solarPanels, SolarPanelConfig &config);
  void processPVDataMap(
      const std::vector<core::utils::osgUtils::instancing::GeometryData>
          &masterGeometryData,
      const std::map<std::string, core::simulation::power::PVData> &pvDataMap,
      float maxPVIntensity);

  void transform(const osg::Vec3 &translation, const osg::Quat &rotation,
                 const osg::Vec3 &scale = osg::Vec3(1.0, 1.0, 1.0));

  auto getTranslation() const;

  std::unique_ptr<std::map<std::string, std::vector<float>>> getInfluxDataFromCSV(
      opencover::utils::read::CSVStream &stream, float &max, float &min, float &sum,
      int &timesteps);

  auto readStaticCampusData(opencover::utils::read::CSVStream &stream, float &max,
                            float &min, float &sum);

  void applyStaticDataCampusToCityGML(const std::string &filePath,
                                      bool updateColorMap);
  void applyStaticDataToCityGML(const std::string &filePath,
                                bool updateColorMap = true);
  void applyInfluxCSVToCityGML(const std::string &filePath,
                               bool updateColorMap = true);

  void enableCityGML(bool on, bool updateColorMap = true);

  void addCityGMLObject(const std::string &name,
                        osg::ref_ptr<osg::Group> citygmlObjGroup);
  void addCityGMLObjects(osg::ref_ptr<osg::Group> citygmlGroup);
  void addSolarPanels(const boost::filesystem::path &dirPath);

  auto readStaticPowerData(opencover::utils::read::CSVStream &stream, float &max,
                           float &min, float &sum);
  osg::ref_ptr<osg::Node> readPVModel(const boost::filesystem::path &modelDir,
                                      const std::string &nameInModelDir);

  void saveCityGMLObjectDefaultStateSet(
      const std::string &name, const core::utils::osgUtils::Geodes &citygmlGeodes);

  void restoreGeodesStatesets(CityGMLDeviceSensor &sensor, const std::string &name,
                              const core::utils::osgUtils::Geodes &citygmlGeodes);
  void restoreCityGMLDefaultStatesets();

  std::map<std::string, std::unique_ptr<CityGMLDeviceSensor>> m_sensorMap;
  std::map<std::string, core::utils::osgUtils::Geodes> m_defaultStatesets;
  SolarPanelList m_panels;

  osg::ref_ptr<osg::Switch> m_parent;
  osg::ref_ptr<osg::Group> m_cityGMLGroup;
  osg::ref_ptr<osg::Group> m_pvGroup;
  osg::ref_ptr<osg::ClipNode> m_coverRootGroup;

  CityGMLUI m_cityGMLUI;
  std::string m_pvDir;
  std::string m_influxPath;
  std::string m_campusPath;
  std::string m_staticPower;
  std::string m_modelDir;
  bool m_enabled;
};
