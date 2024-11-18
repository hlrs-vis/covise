#include "EnnovatisDevice.h"

#include "build_options.h"

// core
#include "core/interfaces/IInfoboard.h"

// ennovatis
#include <ennovatis/building.h>
#include <ennovatis/json.h>
#include <ennovatis/rest.h>

// cover
#include <cover/coVRAnimationManager.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/ui/SelectionList.h>

// std
#include <algorithm>
#include <memory>
#include <string>

// osg
#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Quat>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/ref_ptr>

using namespace opencover;

namespace {
EnnovatisDevice *m_selectedDevice = nullptr;
constexpr bool debug = build_options.debug_ennovatis;
}  // namespace

EnnovatisDevice::EnnovatisDevice(
    const ennovatis::Building &building, opencover::ui::SelectionList *channelList,
    std::shared_ptr<ennovatis::rest_request> req,
    std::shared_ptr<ennovatis::ChannelGroup> channelGroup,
    std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
    std::unique_ptr<core::interface::IBuilding> &&drawableBuilding)
    : m_deviceGroup(new osg::Group()),
      m_infoBoard(std::move(infoBoard)),
      m_drawableBuilding(std::move(drawableBuilding)),
      m_request(req),
      m_channelGroup(channelGroup),
      m_channelSelectionList(channelList),
      m_buildingInfo(BuildingInfo(&building)),
      m_opncvrCtrl(opencover::coVRMSController::instance()) {
  init();
}

auto EnnovatisDevice::createBillboardTxt() {
  if (m_buildingInfo.channelResponse.empty()) return std::string();

  // building info
  std::string billboardTxt = "ID: " + m_buildingInfo.building->getId() + "\n" +
                             "Street: " + m_buildingInfo.building->getStreet() +
                             "\n";

  // channel info
  auto currentSelectedChannel = getSelectedChannelIdx();
  auto channels = m_buildingInfo.building->getChannels(*m_channelGroup.lock());
  auto channelIt = std::next(channels.begin(), currentSelectedChannel);

  // channel response
  auto channel = *channelIt;
  std::string response = m_buildingInfo.channelResponse[currentSelectedChannel];
  billboardTxt += channel.to_string() + "\n";
  auto resp_obj = ennovatis::json_parser()(response);
  std::string resp_str = "Error parsing response";
  if (resp_obj) {
    resp_str = *resp_obj;
    createTimestepColorList(*resp_obj);
  }
  billboardTxt += "Response:\n" + resp_str + "\n";
  return billboardTxt;
}

void EnnovatisDevice::setChannel(int idx) {
  m_channelSelectionList->select(idx);
  if (!m_buildingInfo.channelResponse.empty() && !m_restWorker.isRunning())
    m_infoBoard->updateInfo(createBillboardTxt());
}

void EnnovatisDevice::setChannelGroup(
    std::shared_ptr<ennovatis::ChannelGroup> group) {
  m_channelGroup = group;
  if (m_InfoVisible) fetchData();
  if (m_selectedDevice == this) updateChannelSelectionList();
}

void EnnovatisDevice::updateChannelSelectionList() {
  auto channels = m_buildingInfo.building->getChannels(*m_channelGroup.lock());
  std::vector<std::string> channelNames(channels.size());
  auto channelsIt = channels.begin();
  std::generate(channelNames.begin(), channelNames.end(), [&channelsIt]() mutable {
    auto channel = *channelsIt;
    ++channelsIt;
    return channel.name;
  });
  m_channelSelectionList->setList(channelNames);
  m_channelSelectionList->setCallback([this](int idx) { setChannel(idx); });
  m_channelSelectionList->select(0);
}

void EnnovatisDevice::fetchData() {
  if (!m_InfoVisible || !m_restWorker.checkStatus()) return;

  // make sure only master node fetches data from Ennovatis => sync with slave
  // in update()
  if (m_opncvrCtrl->isMaster())
    m_restWorker.fetchChannels(*m_channelGroup.lock(), *m_buildingInfo.building,
                               *m_request.lock());
}

void EnnovatisDevice::init() {
  m_deviceGroup->setName(m_buildingInfo.building->getId() + ".");

  // infoboard
  m_infoBoard->initInfoboard();
  m_infoBoard->initDrawable();
  m_deviceGroup->addChild(m_infoBoard->getDrawable());

  // cylinder / building representation
  m_drawableBuilding->initDrawables();
  for (auto drawable : m_drawableBuilding->getDrawables())
    m_deviceGroup->addChild(drawable);
}

void EnnovatisDevice::updateColorByTime(int timestep) {
  if (m_timestepColors.empty()) return;
  auto numTimesteps = m_timestepColors.size();
  const auto &cylinderColor =
      m_timestepColors[timestep < numTimesteps ? timestep : numTimesteps - 1];
  m_drawableBuilding->updateColor(*cylinderColor);
}

void EnnovatisDevice::update() {
  if (!m_InfoVisible) return;
  auto results = m_restWorker.getResult();

  bool finished_master = m_opncvrCtrl->isMaster() && results != nullptr;
  finished_master = m_opncvrCtrl->syncBool(finished_master);

  if (finished_master) {
    std::vector<std::string> results_vec;
    if (m_opncvrCtrl->isMaster()) results_vec = *results;

    results_vec = m_opncvrCtrl->syncVector(results_vec);

    m_buildingInfo.channelResponse.clear();
    m_opncvrCtrl->waitForSlaves();
    m_buildingInfo.channelResponse = std::move(results_vec);

    // building info
    auto billboardTxt = createBillboardTxt();
    m_infoBoard->updateInfo(billboardTxt);
    m_infoBoard->showInfo();
  }
}

void EnnovatisDevice::activate() {
  m_InfoVisible = true;
  m_selectedDevice = this;
  updateChannelSelectionList();
  fetchData();
}

void EnnovatisDevice::disactivate() {
  if (m_infoBoard->enabled()) {
    m_InfoVisible = false;
    m_infoBoard->hideInfo();
    // reset to default
    for (auto drawable : m_drawableBuilding->getDrawables())
      m_deviceGroup->removeChild(drawable);

    m_drawableBuilding->initDrawables();
    m_timestepColors.clear();
  }
}

int EnnovatisDevice::getSelectedChannelIdx() const {
  auto selectedChannel = m_channelSelectionList->selectedIndex();
  return (selectedChannel < m_buildingInfo.channelResponse.size()) ? selectedChannel
                                                                   : 0;
}

void EnnovatisDevice::createTimestepColorList(
    const ennovatis::json_response_object &j_resp_obj) {
  auto numTimesteps = j_resp_obj.Times.size();
  auto &respValues = j_resp_obj.Values;
  auto maxValue = *std::max_element(respValues.begin(), respValues.end());
  m_timestepColors.clear();
  m_timestepColors.resize(numTimesteps);
  if (numTimesteps > opencover::coVRAnimationManager::instance()->getNumTimesteps())
    opencover::coVRAnimationManager::instance()->setNumTimesteps(numTimesteps);

  for (auto t = 0; t < numTimesteps; ++t)
    m_timestepColors[t] =
        m_drawableBuilding->getColorInRange(respValues[t], maxValue);
}
