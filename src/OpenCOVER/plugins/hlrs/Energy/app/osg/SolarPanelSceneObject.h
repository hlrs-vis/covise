#pragma once
#include "CityGMLSceneObject.h"
#include "presentation/SolarPanel.h"

#include <vector>
#include <memory>
#include <boost/filesystem.hpp>

#include <lib/core/interfaces/ISolarPanel.h>
#include <lib/core/ClassLogger.h>
#include <lib/core/utils/osgUtils.h>
#include <app/typedefs.h>

typedef std::vector<std::unique_ptr<core::interface::ISolarPanel>> SolarPanelList;
typedef std::map<std::string, PVData> PVDataMap;

class SolarPanelSceneObject : core::ClassLogger
{
public:
    SolarPanelSceneObject(CityGMLSceneObject *gmlObj, osg::ref_ptr<osg::Group> parent, const boost::filesystem::path &modelDir, const PVDataMap &PV, float maxPVIntensity, core::interface::ILogger &logger);
    ~SolarPanelSceneObject();
    SolarPanelSceneObject(const SolarPanelSceneObject&) = delete;
    SolarPanelSceneObject& operator=(const SolarPanelSceneObject&) = delete;
    void enable();
    bool empty() { return m_panels.empty(); }

private:
    void init(
        CityGMLSceneObject *gmlObj,
        const boost::filesystem::path &modelDir,
        const std::map<std::string, PVData> &pvDataMap,
        float maxPVIntensity);
    osg::ref_ptr<osg::Node> readPVModel(const boost::filesystem::path &modelDir,
        const std::string &nameInModelDir);
    void processPVDataMap(
        CityGMLSceneObject *gmlObj,
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

    SolarPanelList m_panels;
    osg::ref_ptr<osg::Group> m_root;
    osg::ref_ptr<osg::Group> m_parent;
    bool m_enabled;
};
