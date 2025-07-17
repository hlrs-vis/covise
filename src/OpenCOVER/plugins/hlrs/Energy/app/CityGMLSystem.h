#pragma once

#include <OpenConfig/file.h>
#include <PluginUtil/colors/ColorBar.h>
#include <cover/coVRPlugin.h>
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>
#include <lib/core/interfaces/ISystem.h>
#include <lib/core/simulation/simulation.h>
#include <lib/core/simulation/power.h>
#include <lib/core/utils/osgUtils.h>
#include <utils/read/csv/csv.h>

#include <boost/filesystem.hpp>
#include <memory>
#include <osg/ClipNode>
#include <osg/Group>
#include <string>
#include <vector>

#include "presentation/SolarPanel.h"
#include "ui/citygml/CityGMLDeviceSensor.h"

typedef std::vector<std::unique_ptr<core::interface::ISolarPanel>> SolarPanelList;

class CityGMLSystem final : public core::interface::ISystem {
 public:
  CityGMLSystem(opencover::coVRPlugin *plugin, opencover::ui::Menu *parentMenu,
                osg::ref_ptr<osg::ClipNode> rootGroup, osg::ref_ptr<osg::Group> parent);
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
  void initPV(
      osg::ref_ptr<osg::Node> masterPanel,
      const std::map<std::string, core::simulation::power::PVData> &pvDataMap,
      float maxPVIntensity);
  void initCityGMLUI();
  void initCityGMLColorMap();

  std::pair<std::map<std::string, core::simulation::power::PVData>, float>
  loadPVData(opencover::utils::read::CSVStream &pvStream);

  SolarPanel createSolarPanel(
      const std::string &name, osg::ref_ptr<osg::Group> parent,
      const std::vector<core::utils::osgUtils::instancing::GeometryData>
          &masterGeometryData,
      const osg::Matrix &matrix, const osg::Vec4 &colorIntensity);

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

  osg::Vec3 getTranslation() const;

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

  osg::ref_ptr<osg::Group> m_cityGMLGroup;
  osg::ref_ptr<osg::Group> m_pvGroup;
  osg::ref_ptr<osg::ClipNode> m_coverRootGroup;

  std::unique_ptr<opencover::CoverColorBar> m_colorMap;

  // will be deleted by opencover::ui
  opencover::coVRPlugin *m_plugin;
  opencover::ui::Menu *m_parentMenu;
  opencover::ui::Menu *m_menu;
  opencover::ui::EditField *m_X;
  opencover::ui::EditField *m_Y;
  opencover::ui::EditField *m_Z;
  opencover::ui::Button *m_enableInfluxCSV;
  opencover::ui::Button *m_enableInfluxArrow;
  opencover::ui::Button *m_PVEnable;
  opencover::ui::Button *m_staticCampusPower;
  opencover::ui::Button *m_staticPower;

  bool m_enabled;
};
