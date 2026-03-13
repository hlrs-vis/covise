#include "CityGMLSceneObject.h"
#include "app/osg/ui/citygml/CityGMLDeviceSensor.h"
#include "presentation/CityGMLBuilding.h"
#include "presentation/OsgTxtInfoboard.h"

#include <osg/MatrixTransform>

#include <cassert>

using namespace core::utils::osgUtils;

CityGMLSceneObject::CityGMLSceneObject(osg::ref_ptr<osg::ClipNode> rootGroup,
    osg::ref_ptr<osg::Switch> parent, core::interface::ILogger &logger)
    : core::ClassLogger(logger, "CityGMLSceneObject")
    , m_coverRootGroup(rootGroup)
    , m_parent(parent)
    , m_root(new osg::Group())
{
    if (!parent) {
        error("Parent must not be null");
        return;
    }
    init();
    info("Create citygml sceneobject.");
}

CityGMLSceneObject::~CityGMLSceneObject()
{
    if (m_root)
    {
        restoreCityGMLDefaultStatesets();
        for (unsigned int i = 0; i < m_root->getNumChildren(); ++i)
        {
            auto child = m_root->getChild(i);
            m_coverRootGroup->addChild(child);
        }
        core::utils::osgUtils::deleteChildrenFromOtherGroup(m_root,
            m_coverRootGroup);
    }
}

void CityGMLSceneObject::init()
{
    m_root->setName("CityGML");
    m_parent->addChild(m_root);
}

void CityGMLSceneObject::update()
{
    for (auto &[name, sensor] : m_sensorMap)
        sensor->update();
}

void CityGMLSceneObject::updateTime(int timestep)
{
    for (auto &[name, sensor] : m_sensorMap)
        sensor->updateTime(timestep);
}

void CityGMLSceneObject::addCityGMLObject(const std::string &name,
    osg::ref_ptr<osg::Group> citygmlObjGroup)
{
    if (!citygmlObjGroup->getNumChildren())
        return;

    if (m_sensorMap.find(name) != m_sensorMap.end())
        return;

    auto geodes = core::utils::osgUtils::getGeodes(citygmlObjGroup);
    if (geodes->empty())
        return;

    // store default stateset
    saveCityGMLObjectDefaultStateSet(name, *geodes);

    auto boundingbox = core::utils::osgUtils::getBoundingBox(*geodes);
    auto infoboardPos = Pos(boundingbox.center().x(), boundingbox.center().y(), boundingbox.center().z());
    infoboardPos.z += (boundingbox.zMax() - boundingbox.zMin()) / 2 + boundingbox.zMin();
    auto infoboard = std::make_unique<OsgTxtInfoboard>(
        infoboardPos, name, "DroidSans-Bold.ttf", 50, 50, 2.0f, 0.1, 2);
    auto building = std::make_unique<CityGMLBuilding>(*geodes);
    auto sensor = std::make_unique<CityGMLDeviceSensor>(
        citygmlObjGroup, std::move(infoboard), std::move(building));
    m_sensorMap.insert({ name, std::move(sensor) });
}

void CityGMLSceneObject::addCityGMLObjects(osg::ref_ptr<osg::Group> citygmlGroup)
{
    for (unsigned int i = 0; i < citygmlGroup->getNumChildren(); ++i)
    {
        osg::ref_ptr<osg::Group> child = dynamic_cast<osg::Group *>(citygmlGroup->getChild(i));
        if (!child)
            continue;
        const auto &name = child->getName();

        // handle quad tree optimized scenegraph
        if (name == "GROUP" || name == "")
        {
            addCityGMLObjects(child);
            continue;
        }

        addCityGMLObject(name, child);
    }
}

void CityGMLSceneObject::move(const Pos &pos)
{
    if (!isActive(m_parent, m_root))
        return;
    transform(osg::Vec3(pos.x, pos.y, pos.z), {});
}

void CityGMLSceneObject::enable(const osg::Vec3 &translation)
{
    if (m_sensorMap.empty())
    {
        for (unsigned int i = 0; i < m_coverRootGroup->getNumChildren(); ++i)
        {
            osg::ref_ptr<osg::MatrixTransform> child = dynamic_cast<osg::MatrixTransform *>(m_coverRootGroup->getChild(i));
            if (!child)
                continue;

            auto name = child->getName();
            if (name.find(".gml") == std::string::npos)
                continue;

            addCityGMLObjects(child);
            m_root->addChild(child);
            child->setMatrix(osg::Matrix::translate(translation));
            transform(translation, {});
        }
        core::utils::osgUtils::deleteChildrenFromOtherGroup(m_coverRootGroup,
            m_root);
    }
    m_enabled = !m_enabled;
    if (m_enabled)
        switchTo(m_root, m_parent);
}

void CityGMLSceneObject::transform(const osg::Vec3 &translation,
    const osg::Quat &rotation, const osg::Vec3 &scale)
{
    assert(m_root && "CityGML group is not initialized.");
    if (m_root->getNumChildren() == 0)
    {
        warn("No CityGML objects to transform.");
        return;
    }
    for (unsigned int i = 0; i < m_root->getNumChildren(); ++i)
    {
        osg::ref_ptr<osg::Node> child = m_root->getChild(i);
        if (auto mt = dynamic_cast<osg::MatrixTransform *>(child.get()))
        {
            osg::Matrix matrix = osg::Matrix::translate(translation) * osg::Matrix::rotate(rotation) * osg::Matrix::scale(scale);
            mt->setMatrix(matrix);
        }
        else
        {
            error("Child is not a MatrixTransform.");
        }
    }
}

void CityGMLSceneObject::saveCityGMLObjectDefaultStateSet(const std::string &name,
    const Geodes &citygmlGeodes)
{
    Geodes geodesCopy(citygmlGeodes.size());
    for (auto i = 0; i < citygmlGeodes.size(); ++i)
    {
        auto geode = citygmlGeodes[i];
        geodesCopy[i] = dynamic_cast<osg::Geode *>(geode->clone(osg::CopyOp::DEEP_COPY_STATESETS));
    }
    m_defaultStateSets.insert({ name, std::move(geodesCopy) });
}

void CityGMLSceneObject::restoreGeodesStatesets(CityGMLDeviceSensor &sensor,
    const std::string &name,
    const Geodes &citygmlGeodes)
{
    if (m_defaultStateSets.find(name) == m_defaultStateSets.end())
        return;

    if (citygmlGeodes.empty())
        return;

    for (auto i = 0; i < citygmlGeodes.size(); ++i)
    {
        auto gmlDefault = citygmlGeodes[i];
        osg::ref_ptr<osg::Geode> toRestore = sensor.getDrawable(i)->asGeode();
        if (toRestore)
        {
            toRestore->setStateSet(gmlDefault->getStateSet());
        }
    }
}

void CityGMLSceneObject::restoreCityGMLDefaultStatesets()
{
    for (auto &[name, sensor] : m_sensorMap)
    {
        osg::ref_ptr<osg::Group> sensorParent = sensor->getParent();
        if (!sensorParent)
            continue;

        restoreGeodesStatesets(*sensor, name, m_defaultStateSets[name]);
    }
    m_defaultStateSets.clear();
}
