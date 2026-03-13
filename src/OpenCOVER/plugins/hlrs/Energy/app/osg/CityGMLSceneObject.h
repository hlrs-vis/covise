#pragma once
#include "ui/citygml/CityGMLDeviceSensor.h"
#include <osg/Switch>
#include <osg/ClipNode>
#include <osg/Group>
#include <osg/Quat>
#include <osg/Vec3>

#include <utils/read/csv/csv.h>

#include <lib/core/utils/osgUtils.h>
#include <lib/core/interfaces/ISolarPanel.h>
#include <lib/core/ClassLogger.h>

#include <boost/filesystem.hpp>

typedef core::utils::osgUtils::Geodes Geodes;
typedef std::vector<std::unique_ptr<core::interface::ISolarPanel>> SolarPanelList;

class CityGMLSceneObject : core::ClassLogger
{
public:
    CityGMLSceneObject(osg::ref_ptr<osg::ClipNode> rootGroup,
        osg::ref_ptr<osg::Switch> parent, core::interface::ILogger& logger);
    ~CityGMLSceneObject();
    void enable(const osg::Vec3 &translation = { 0.0f, 0.0f, 0.0f });
    bool enabled() const { return m_enabled && core::utils::osgUtils::isActive(m_parent, m_root); }
    void update();
    void updateTime(int timestep);
    void move(const Pos &pos);
    auto begin() { return m_sensorMap.begin(); }
    auto end() { return m_sensorMap.end(); }

    auto getRoot() const { return m_root; };
    auto getParent() const { return m_parent;}
    auto contains(const std::string &name) { return m_sensorMap.find(name) != m_sensorMap.end(); }
    auto find(const std::string &name)
    {
        auto it = m_sensorMap.find(name);
        return (it != m_sensorMap.end()) ? it->second.get() : nullptr;
    }

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

    bool m_enabled;
    std::map<std::string, std::unique_ptr<CityGMLDeviceSensor>> m_sensorMap;
    std::map<std::string, core::utils::osgUtils::Geodes> m_defaultStateSets;

    osg::ref_ptr<osg::Switch> m_parent;
    osg::ref_ptr<osg::Group> m_root;
    osg::ref_ptr<osg::ClipNode> m_coverRootGroup;
};
