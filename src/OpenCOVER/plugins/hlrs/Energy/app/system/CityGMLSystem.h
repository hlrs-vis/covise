#pragma once

#include <OpenConfig/file.h>
#include <cover/coVRPlugin.h>

#include <lib/core/interfaces/ISystem.h>
#include <lib/core/interfaces/ILogger.h>
#include <lib/core/simulation/simulation.h>
#include <lib/core/simulation/unitmap.h>
#include <lib/core/simulation/power.h>
#include <lib/core/utils/osgUtils.h>
#include <utils/read/csv/csv.h>

#include <memory>
#include <string>
#include <vector>

#include "app/cover/ui/CityGMLUI.h"
#include "app/osg/CityGMLSceneObject.h"
#include "app/osg/SolarPanelSceneObject.h"

struct CityGMLConfig {
  std::string pvDir;
  std::string influxPath;
  std::string campusPath;
  std::string staticPower;
  std::string modelDir;
};

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
                osg::ref_ptr<osg::ClipNode> rootGroup, osg::ref_ptr<osg::Switch> parent, core::interface::ILogger& logger);
  virtual ~CityGMLSystem() = default;
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

  std::pair<std::map<std::string, core::simulation::power::PVData>, float>
  loadPVData(opencover::utils::read::CSVStream &pvStream);

  void processPVRow(
      const opencover::utils::read::CSVStream::CSVRow &row,
      std::map<std::string, core::simulation::power::PVData> &pvDataMap,
      float &maxPVIntensity);

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

  void enableScene(bool on, bool updateColorMap = true);

  void addSolarPanels(const boost::filesystem::path &dirPath);

  auto readStaticPowerData(opencover::utils::read::CSVStream &stream, float &max,
                           float &min, float &sum);
  osg::ref_ptr<osg::Node> readPVModel(const boost::filesystem::path &modelDir,
                                      const std::string &nameInModelDir);

  CityGMLUI m_cityGMLUI;
  CityGMLSceneObject m_gmlSceneObject;
  std::unique_ptr<SolarPanelSceneObject> m_pvSceneObject;
  CityGMLConfig m_config;
  core::interface::ILogger& m_logger;
  bool m_enabled;
};
