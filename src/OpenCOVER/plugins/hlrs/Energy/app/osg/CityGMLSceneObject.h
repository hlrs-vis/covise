#pragma once
#include "ui/citygml/CityGMLDeviceSensor.h"
#include "presentation/SolarPanel.h"
#include <osg/Switch>
#include <osg/ClipNode>
#include <osg/Group>
#include <osg/Quat>
#include <osg/Vec3>

#include <utils/read/csv/csv.h>

#include <lib/core/utils/osgUtils.h>
#include <lib/core/interfaces/ISolarPanel.h>

#include <boost/filesystem.hpp>

typedef core::utils::osgUtils::Geodes Geodes;
typedef std::vector<std::unique_ptr<core::interface::ISolarPanel>> SolarPanelList;

class CityGMLSceneObject
{
public:
    CityGMLSceneObject(osg::ref_ptr<osg::ClipNode> rootGroup,
        osg::ref_ptr<osg::Switch> parent);
    ~CityGMLSceneObject();
    void enable(const osg::Vec3 &translation = { 0.0f, 0.0f, 0.0f });
    bool enabled() const { return m_enabled; }
    void update();
    void updateTime(int timestep);
    void move(const Pos &pos);
    auto begin() { return m_sensorMap.begin(); }
    auto end() { return m_sensorMap.end(); }

    auto getRoot() { return m_root; };
    auto contains(const std::string &name) { return m_sensorMap.find(name) != m_sensorMap.end(); }
    auto find(const std::string &name)
    {
        auto it = m_sensorMap.find(name);
        return (it != m_sensorMap.end()) ? it->second.get() : nullptr;
    }

    // PV
    void enablePV(bool on);
    void initPV(
        const boost::filesystem::path &modelDir,
        const std::map<std::string, PVData> &pvDataMap,
        float maxPVIntensity);
    const auto &getPanels() const { return m_panels; }

private:
    void init();
    void transform(const osg::Vec3 &translation, const osg::Quat &rotation = {},
        const osg::Vec3 &scale = osg::Vec3(1.0, 1.0, 1.0));
    void addCityGMLObject(const std::string &name,
        osg::ref_ptr<osg::Group> citygmlObjGroup);
    void addCityGMLObjects(osg::ref_ptr<osg::Group> citygmlGroup);
    void restoreGeodesStatesets(CityGMLDeviceSensor &sensor,
        const std::string &name,
        const Geodes &citygmlGeodes);
    void restoreCityGMLDefaultStatesets();
    void saveCityGMLObjectDefaultStateSet(
        const std::string &name, const core::utils::osgUtils::Geodes &citygmlGeodes);

    // PV
    osg::ref_ptr<osg::Node> readPVModel(const boost::filesystem::path &modelDir,
        const std::string &nameInModelDir);
    void processPVDataMap(
        const std::vector<core::utils::osgUtils::instancing::GeometryData>
            &masterGeometryData,
        const std::map<std::string, core::simulation::power::PVData> &pvDataMap,
        float maxPVIntensity);
    void processSolarPanelDrawable(SolarPanelList &solarPanels,
        const SolarPanelConfig &config);
    void processSolarPanelDrawables(
        const core::simulation::power::PVData &data,
        const std::vector<osg::ref_ptr<osg::Node>> drawables,
        SolarPanelList &solarPanels, SolarPanelConfig &config);
    std::unique_ptr<SolarPanel> createSolarPanel(
        const std::string &name, osg::ref_ptr<osg::Group> parent,
        const std::vector<core::utils::osgUtils::instancing::GeometryData>
            &masterGeometryData,
        const osg::Matrix &matrix, const Color &colorIntensity);

    bool m_enabled;
    SolarPanelList m_panels;
    std::map<std::string, std::unique_ptr<CityGMLDeviceSensor>> m_sensorMap;
    std::map<std::string, core::utils::osgUtils::Geodes> m_defaultStateSets;

    osg::ref_ptr<osg::Switch> m_parent;
    osg::ref_ptr<osg::Group> m_root;
    osg::ref_ptr<osg::Group> m_pv;
    osg::ref_ptr<osg::ClipNode> m_coverRootGroup;
};
