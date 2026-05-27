#pragma once

#include <OpenConfig/file.h>
#include <cover/coVRPlugin.h>

#include <lib/core/interfaces/ui/IComponent.h>
#include <lib/core/simulation/simulationresult.h>
#include <lib/core/Logger.h>
#include <lib/core/simulation/unitmap.h>
#include <lib/core/simulation/powerresult.h>
#include <lib/core/utils/osgUtils.h>
#include <utils/read/csv/csv.h>

#include <memory>
#include <string>
#include <vector>

#include "app/ui/CityGMLUI.h"
#include "app/osg/CityGMLSceneObject.h"
#include "app/osg/SolarPanelSceneObject.h"
#include "lib/core/interfaces/ui/IGUIFactory.h"


struct CityGMLConfig {
  std::string pvDir;
  std::string influxPath;
  std::string campusPath;
  std::string staticPower;
  std::string modelDir;
};

class CityGMLSystem {
 public:
  explicit CityGMLSystem(opencover::coVRPlugin *plugin, core::interface::ui::IComponent *parentMenu, const core::interface::ui::IGUIFactory &factory,
                osg::ref_ptr<osg::ClipNode> rootGroup, osg::ref_ptr<osg::Switch> parent, Logger logger);

  void init();
  void enable(bool on);
  bool isEnabled() const ;
  void update();
  void updateTime(int timestep);

  void updateInfluxColorMaps(
      float min, float max,
      std::shared_ptr<core::simulation::SimulationResult> powerSimulation,
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
  Logger m_logger;
  bool m_enabled;
};
