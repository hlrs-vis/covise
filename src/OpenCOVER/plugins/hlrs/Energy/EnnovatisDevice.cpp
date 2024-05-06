#include "EnnovatisDevice.h"
#include "build_options.h"

// core
#include "core/interfaces/IInfoboard.h"

// ennovatis
#include "ennovatis/json.h"
#include <ennovatis/building.h>
#include <ennovatis/rest.h>

// cover
#include "cover/ui/SelectionList.h"
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRAnimationManager.h>

// std
#include <string>
#include <memory>
#include <algorithm>

// osg
#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Material>
#include <osg/Quat>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/ref_ptr>

using namespace opencover;

namespace {
float h = 1.f;
float w = 2.f;
EnnovatisDevice *m_selectedDevice = nullptr;
constexpr bool debug = build_options.debug_ennovatis;

auto createMaterial(const osg::Vec4 &color)
{
    osg::ref_ptr<osg::Material> mat = new osg::Material;
    mat->setDiffuse(osg::Material::FRONT, color);
    return mat;
}

void overrideGeodeColor(osg::Geode *geode, const osg::Vec4 &color)
{
    auto mat = createMaterial(color);
    geode->getOrCreateStateSet()->setAttribute(mat, osg::StateAttribute::OVERRIDE);
}

/**
 * @brief Adds a cylinder between two points.
 * Source: http://www.thjsmith.com/40/cylinder-between-two-points-opengl-c
 * 
 * @param start The starting point of the cylinder.
 * @param end The ending point of the cylinder.
 * @param radius The radius of the cylinder.
 * @param cylinderColor The color of the cylinder.
 * @param group The group to which the cylinder will be added.
 */
void addCylinderBetweenPoints(osg::Vec3 start, osg::Vec3 end, float radius, osg::Vec4 cylinderColor, osg::Group *group)
{
    osg::ref_ptr geode = new osg::Geode;
    osg::Vec3 center;
    float height;
    osg::ref_ptr<osg::Cylinder> cylinder;
    osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
    osg::ref_ptr<osg::Material> pMaterial;

    height = (start - end).length();
    center = osg::Vec3((start.x() + end.x()) / 2, (start.y() + end.y()) / 2, (start.z() + end.z()) / 2);

    // This is the default direction for the cylinders to face in OpenGL
    osg::Vec3 z = osg::Vec3(0, 0, 1);

    // Get diff between two points you want cylinder along
    osg::Vec3 p = start - end;

    // Get CROSS product (the axis of rotation)
    osg::Vec3 t = z ^ p;

    // Get angle. length is magnitude of the vector
    double angle = acos((z * p) / p.length());

    // Create a cylinder between the two points with the given radius
    cylinder = new osg::Cylinder(center, radius, height);
    cylinder->setRotation(osg::Quat(angle, osg::Vec3(t.x(), t.y(), t.z())));

    cylinderDrawable = new osg::ShapeDrawable(cylinder);
    geode->addDrawable(cylinderDrawable);

    // Set the color of the cylinder that extends between the two points.
    overrideGeodeColor(geode, cylinderColor);

    // Add the cylinder between the two points to an existing group
    group->addChild(geode);
}
} // namespace

EnnovatisDevice::EnnovatisDevice(const ennovatis::Building &building,
                                 std::shared_ptr<opencover::ui::SelectionList> channelList,
                                 std::shared_ptr<ennovatis::rest_request> req,
                                 std::shared_ptr<ennovatis::ChannelGroup> channelGroup,
                                 std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
                                 const CylinderAttributes &cylinderAttributes)
: m_deviceGroup(new osg::Group())
, m_infoBoard(std::move(infoBoard))
, m_request(req)
, m_channelGroup(channelGroup)
, m_channelSelectionList(channelList)
, m_buildingInfo(BuildingInfo(&building))
, m_opncvr_ctrl(opencover::coVRMSController::instance())
, m_cylinderAttributes(cylinderAttributes)
{
    init();
}

void EnnovatisDevice::setChannel(int idx)
{
    m_channelSelectionList.lock()->select(idx);
    if (!m_buildingInfo.channelResponse.empty() && !m_rest_worker.isRunning())
        /* showInfo(); */
        m_infoBoard->showInfo();
}

void EnnovatisDevice::setChannelGroup(std::shared_ptr<ennovatis::ChannelGroup> group)
{
    m_channelGroup = group;
    if (m_InfoVisible)
        fetchData();
    if (m_selectedDevice == this)
        updateChannelSelectionList();
}

void EnnovatisDevice::updateChannelSelectionList()
{
    auto channels = m_buildingInfo.building->getChannels(*m_channelGroup.lock());
    std::vector<std::string> channelNames(channels.size());
    auto channelsIt = channels.begin();
    std::generate(channelNames.begin(), channelNames.end(), [&channelsIt]() mutable {
        auto channel = *channelsIt;
        ++channelsIt;
        return channel.name;
    });
    auto cSLi = m_channelSelectionList.lock();
    cSLi->setList(channelNames);
    cSLi->setCallback([this](int idx) { setChannel(idx); });
    cSLi->select(0);
}

osg::Vec4 EnnovatisDevice::getColor(float val, float max) const
{
    const auto &colMax = m_cylinderAttributes.colorMap.max;
    const auto &colMin = m_cylinderAttributes.colorMap.min;
    max = std::max(max, 1.f);
    float valN = val / max;

    osg::Vec4 col(colMax.r() * valN + colMin.r() * (1 - valN), colMax.g() * valN + colMin.g() * (1 - valN),
                  colMax.b() * valN + colMin.b() * (1 - valN), colMax.a() * valN + colMin.a() * (1 - valN));
    return col;
}

void EnnovatisDevice::fetchData()
{
    if (!m_InfoVisible || !m_rest_worker.checkStatus())
        return;

    // make sure only master node fetches data from Ennovatis => sync with slave in update()
    if (m_opncvr_ctrl->isMaster())
        m_rest_worker.fetchChannels(*m_channelGroup.lock(), *m_buildingInfo.building, *m_request.lock());
}

void EnnovatisDevice::init()
{
    // infoboard
    m_infoBoard->initInfoboard();
    m_infoBoard->initDrawable();

    // cylinder
    m_deviceGroup->setName(m_buildingInfo.building->getId() + ".");

    osg::MatrixTransform *matTrans = new osg::MatrixTransform();
    osg::Matrix mat;
    mat.makeTranslate(osg::Vec3(m_buildingInfo.building->getLat(), m_buildingInfo.building->getLon(),
                                m_cylinderAttributes.height + h));
    matTrans->setMatrix(mat);
    matTrans->addChild(m_infoBoard->getDrawable());
    m_deviceGroup->addChild(matTrans);

    w = m_cylinderAttributes.radius * 20;
    h = m_cylinderAttributes.radius * 21;

    // RGB Colors 1,1,1 = white, 0,0,0 = black
    const osg::Vec3f bottom(m_buildingInfo.building->getLat(), m_buildingInfo.building->getLon(),
                            m_buildingInfo.building->getHeight());
    osg::Vec3f top(bottom);
    top.z() += m_cylinderAttributes.height;

    addCylinderBetweenPoints(bottom, top, m_cylinderAttributes.radius, m_cylinderAttributes.colorMap.defaultColor,
                             m_deviceGroup.get());
}

auto EnnovatisDevice::getCylinderGeode()
{
    osg::Geode *cyl = nullptr;
    for (auto i = 0; i < m_deviceGroup->getNumChildren(); ++i)
        if (auto geode = dynamic_cast<osg::Geode *>(m_deviceGroup->getChild(i)))
            return geode;

    return cyl;
}


void EnnovatisDevice::updateColorByTime(int timestep)
{
    if (m_timestepColors.empty())
        return;
    auto numTimesteps = m_timestepColors.size();
    auto geode = getCylinderGeode();
    if (geode) {
        osg::Vec4 cylinderColor = m_timestepColors[timestep < numTimesteps ? timestep : numTimesteps - 1];
        overrideGeodeColor(geode, cylinderColor);
    }
}

void EnnovatisDevice::update()
{
    if (!m_InfoVisible)
        return;
    auto results = m_rest_worker.getResult();

    bool finished_master = m_opncvr_ctrl->isMaster() && results != nullptr;
    finished_master = m_opncvr_ctrl->syncBool(finished_master);

    if (finished_master) {
        std::vector<std::string> results_vec;
        if (m_opncvr_ctrl->isMaster())
            results_vec = *results;

        results_vec = m_opncvr_ctrl->syncVector(results_vec);

        m_buildingInfo.channelResponse.clear();
        m_opncvr_ctrl->waitForSlaves();
        m_buildingInfo.channelResponse = std::move(results_vec);

        // building info
        std::string textvalues =
            "ID: " + m_buildingInfo.building->getId() + "\n" + "Street: " + m_buildingInfo.building->getStreet() + "\n";

        // channel info
        auto currentSelectedChannel = getSelectedChannelIdx();
        auto channels = m_buildingInfo.building->getChannels(*m_channelGroup.lock());
        auto channelIt = std::next(channels.begin(), currentSelectedChannel);

        // channel response
        auto channel = *channelIt;
        std::string response = m_buildingInfo.channelResponse[currentSelectedChannel];
        textvalues += channel.to_string() + "\n";
        auto resp_obj = ennovatis::json_parser()(response);
        std::string resp_str = "Error parsing response";
        if (resp_obj) {
            resp_str = *resp_obj;
            createTimestepColorList(*resp_obj);
        }
        textvalues += "Response:\n" + resp_str + "\n";
        m_infoBoard->updateInfo(textvalues);
        m_infoBoard->showInfo();
    }
}


void EnnovatisDevice::activate()
{
    m_InfoVisible = true;
    m_selectedDevice = this;
    updateChannelSelectionList();
    fetchData();
}

void EnnovatisDevice::disactivate()
{
    if (m_infoBoard->enabled()) {
        m_InfoVisible = false;
        auto geode = getCylinderGeode();
        overrideGeodeColor(geode, m_cylinderAttributes.colorMap.defaultColor);
        m_timestepColors.clear();
        m_infoBoard->hideInfo();
    }
}

int EnnovatisDevice::getSelectedChannelIdx() const
{
    auto selectedChannel = m_channelSelectionList.lock()->selectedIndex();
    return (selectedChannel < m_buildingInfo.channelResponse.size()) ? selectedChannel : 0;
}

void EnnovatisDevice::createTimestepColorList(const ennovatis::json_response_object &j_resp_obj)
{
    auto numTimesteps = j_resp_obj.Times.size();
    auto &respValues = j_resp_obj.Values;
    auto maxValue = *std::max_element(respValues.begin(), respValues.end());
    m_timestepColors.clear();
    m_timestepColors.resize(numTimesteps);
    if (numTimesteps > opencover::coVRAnimationManager::instance()->getNumTimesteps())
        opencover::coVRAnimationManager::instance()->setNumTimesteps(numTimesteps);

    for (auto i = 0; i < m_timestepColors.size(); ++i)
        m_timestepColors[i] = getColor(respValues[i], maxValue);
}