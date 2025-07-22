#include "EnnovatisDevice.h"

#include "build_options.h"

// core
#include <lib/core/interfaces/IInfoboard.h>

// ennovatis
#include <lib/ennovatis/building.h>
#include <lib/ennovatis/channel.h>
#include <lib/ennovatis/json.h>
#include <lib/ennovatis/rest.h>

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

auto roundToString(float value, int precision = 2) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}
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

auto EnnovatisDevice::getSelectedChannelIterator() const {
  auto channels = m_buildingInfo.building->getChannels(*m_channelGroup.lock());
  auto currentSelectedChannel = getSelectedChannelIdx();
  return std::next(channels.begin(), currentSelectedChannel);
}

auto EnnovatisDevice::getResponseObjectForSelectedChannel() const {
  auto currentSelectedChannel = getSelectedChannelIdx();
  auto response = m_buildingInfo.channelResponse[currentSelectedChannel];
  return ennovatis::json_parser()(response);
}

auto EnnovatisDevice::createBillboardTxt(
    const ennovatis::json_response_object &j_resp_obj) {
  if (m_buildingInfo.channelResponse.empty()) return std::string();

  constexpr auto ENDLINE = "\n\n";

  // building info
  std::string billboardTxt = "> ID: " + m_buildingInfo.building->getId() + ENDLINE +
                             "> Street: " + m_buildingInfo.building->getStreet() +
                             ENDLINE;
  auto channelIt = getSelectedChannelIterator();

  // channel response
  auto channel = *channelIt;
  billboardTxt += "> Channel: " + channel.name + ENDLINE;
  billboardTxt +=
      "> Group: " + ennovatis::ChannelGroupToString(channel.group) + ENDLINE;
  billboardTxt += "> Type: " + channel.type + ENDLINE;

  const auto &values = j_resp_obj.Values;
  if (channel.unit.find("/m²") != std::string::npos) {
    // NOTE: compute it when area is available => ennovatis data looks wrong (at
    // least in the rest api; in Cofely it looks fine)
    billboardTxt += "Ennovatis REST-API is not working correctly!\n\n";
    billboardTxt +=
        "> Total Consumption per m²: " + roundToString(m_consumptionPerArea) +
        ENDLINE;
  } else {
    // it seems that the first value at 0 is always the same as the second value
    auto sum = std::accumulate(values.begin() + 1, values.end(), 0.0f);
    auto max = *std::max_element(values.begin(), values.end());
    m_consumptionPerArea = sum / m_buildingInfo.building->getArea();
    auto mean = sum / (values.size() - 1);
    billboardTxt += "> Average Consumption per day: " + roundToString(mean) + " " +
                    channel.unit + ENDLINE;

    billboardTxt +=
        "> Total Consumption: " + roundToString(sum) + " " + channel.unit + ENDLINE;

    billboardTxt +=
        "> Total Consumption per m²: " + roundToString(m_consumptionPerArea) + " " +
        channel.unit + "/m²" + ENDLINE;

    billboardTxt +=
        "> Peak load: " + roundToString(max) + " " + channel.unit + ENDLINE;
  }
  return billboardTxt;
}

void EnnovatisDevice::setChannel(int idx) {
  m_channelSelectionList->select(idx);
  if (!m_buildingInfo.channelResponse.empty() && !m_restWorker.isRunning()) {
    auto resp_obj = getResponseObjectForSelectedChannel();
    m_infoBoard->updateInfo(createBillboardTxt(*resp_obj));
  }
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
  auto drawables = m_drawableBuilding->getDrawables();
  for (auto drawable : drawables) {
    m_deviceGroup->addChild(drawable);
    if (osg::ref_ptr<osg::Geode> geode = drawable->asGeode()) {
      //   m_defaultStateSets.push_back(
      //       new osg::Geode(*geode, osg::CopyOp::DEEP_COPY_NODES));
      m_defaultStateSets.push_back(
          new osg::Geode(*geode, osg::CopyOp::DEEP_COPY_STATESETS));

      //   osg::ref_ptr<osg::Geode> clonedGeode =
      //       new osg::Geode(*geode, osg::CopyOp::DEEP_COPY_NODES);
      //   for (auto i = 0; i < clonedGeode->getNumParents(); ++i) {
      //     auto parent = clonedGeode->getParent(i);
      //     parent->removeChild(clonedGeode);
      //   }

      //   for (auto i = 0; i < clonedGeode->getNumChildren(); ++i) {
      //     auto child = clonedGeode->getChild(i);
      //     for (auto j = 0; j < child->getNumParents(); ++j) {
      //       auto parent = child->getParent(j);
      //     //   if (parent == clonedGeode) continue;
      //     //   parent->removeChild(child);
      //     }
      //   }

      //   m_defaultStateSets.push_back(clonedGeode);
      //   clonedGeode->removeDrawables(0, clonedGeode->getNumDrawables());
      //   for (auto i = 0; i < clonedGeode->getNumDrawables(); ++i) {
      //     auto drawable = clonedGeode->getDrawable(i);
      //     d
      //   }

      //   m_defaultStateSets.emplace_back(
      //     //   new osg::Geode(*geode, osg::CopyOp::DEEP_COPY_STATESETS));
      //       new osg::Geode(*geode, osg::CopyOp::DEEP_COPY_NODES));
      //   osg::clone(geode.get(), osg::CopyOp::DEEP_COPY_NODES));
      //   *geode, osg::CopyOp::DEEP_COPY_NODES);
    }
  }
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
    if (!handleResponse(*results)) {
      std::cout << "Error parsing response: \n";
      for (auto &res : *results) std::cout << res << "\n";
    }
  }
}

bool EnnovatisDevice::handleResponse(const std::vector<std::string> &results) {
  std::vector<std::string> results_vec;
  if (m_opncvrCtrl->isMaster()) results_vec = results;

  results_vec = m_opncvrCtrl->syncVector(results_vec);

  m_buildingInfo.channelResponse.clear();
  m_opncvrCtrl->waitForSlaves();
  m_buildingInfo.channelResponse = std::move(results_vec);

  // building info
  auto resp_obj = getResponseObjectForSelectedChannel();
  if (!resp_obj) return false;

  auto billboardTxt = createBillboardTxt(*resp_obj);
  m_infoBoard->updateInfo(billboardTxt);
  m_infoBoard->showInfo();

  createTimestepColorList(*resp_obj);
  m_sensorData = resp_obj->Values;
  return true;
}

void EnnovatisDevice::activate() {
  m_InfoVisible = true;
  m_selectedDevice = this;
  updateChannelSelectionList();
  fetchData();
}

void EnnovatisDevice::setTimestep(int timestep) {
  updateColorByTime(timestep);
  updateHeightByTime(timestep);
}

void EnnovatisDevice::updateHeightByTime(int timestep) {
  if (m_sensorData.empty()) return;
  auto numTimesteps = m_sensorData.size();
  auto height = m_sensorData[timestep < numTimesteps ? timestep : numTimesteps - 1];
  if (height <= 0) return;
  switch (getSelectedChannelIdx()) {
    case ennovatis::ChannelGroup::Strom:
    case ennovatis::ChannelGroup::Waerme:
    case ennovatis::ChannelGroup::Kaelte:
      height = height * 0.1;
      break;
    case ennovatis::ChannelGroup::Wasser:
      height = height * 10;
      break;
    default:
      break;
  }

  for (auto drawable : m_drawableBuilding->getDrawables()) {
    osg::ref_ptr<osg::Geode> geo = drawable->asGeode();
    if (!geo) continue;
    osg::ref_ptr<osg::ShapeDrawable> shape =
        dynamic_cast<osg::ShapeDrawable *>(geo->getDrawable(0));
    if (!shape) continue;
    osg::ref_ptr<osg::Cylinder> cylinder =
        dynamic_cast<osg::Cylinder *>(shape->getShape());
    if (!cylinder) continue;
    auto old_height = cylinder->getHeight();
    if (old_height == height)
      return;
    else {
      auto tmp_height = osg::Vec3(0, 0, std::abs(height - old_height) / 2);
      if (old_height < height)
        cylinder->setCenter(cylinder->getCenter() + tmp_height);
      else
        cylinder->setCenter(cylinder->getCenter() - tmp_height);
    }
    cylinder->setHeight(height);
    // TODO: Shader implementation later?!
    // NOTE: The new ShapeDrawable is written completely differently, and is now
    // subclassed from osg::Geometry and has ShapeDrawable::build() function that
    // computes all the appropriate vertex arrays etc, So try just calling
    // shapedrawabl->build() when you update the shape.

    // I should however suggest that perhaps ShapeDrawable isn't the tool for the
    // job for this type of interactive updating, it's written as a create once,
    // use many times features. If you are updating the height interactively then
    // it may be best to just use a shader to set the height within the vertex
    // shader. For instance if you had a 1000 cyclinders that all had
    // independently varying heights then I'd write this as an instanced geometry
    // with a uniform arrays or 1D texture that stores the position and sizes then
    // have the vertex shader positing and scale the geometry accordingly. This
    // way you'd just update the uniform/1D texture in a super lightweigth way and
    // everything would just render smoothly and efficiently.
    shape->build();
  }
}

void EnnovatisDevice::disactivate() {
  if (m_infoBoard->enabled()) {
    m_InfoVisible = false;
    m_infoBoard->hideInfo();
    // reset to default
    for (auto drawable : m_drawableBuilding->getDrawables())
      m_deviceGroup->removeChild(drawable);

    for (auto i = 0; i < m_defaultStateSets.size(); ++i) {
      auto drawable = m_drawableBuilding->getDrawable(i);
      //   m_deviceGroup->removeChild(drawable);
      //   osg::ref_ptr<osg::Geode> geode =
      //       osg::clone(m_defaultStateSets[i].get(), osg::CopyOp::DEEP_COPY_NODES);

      if (osg::ref_ptr<osg::Geode> geode = drawable->asGeode()) {
        geode->setStateSet(m_defaultStateSets[i]->getStateSet());
        m_deviceGroup->addChild(geode);
      }

      //   drawable->setStateSet(m_defaultStateSets[i]->getStateSet());
      //   m_deviceGroup->addChild(geode);
      //   if (osg::ref_ptr<osg::Geode> geode = drawable->asGeode()) {
      //     // osg::ref_ptr<osg::Geode> default_geode = new
      //     // osg::Geode(*m_defaultStateSets[i]->asGeode(),
      //     // osg::CopyOp::DEEP_COPY_NODES); geode = default_geode;
      //     m_deviceGroup->removeChild(drawable);
      //     geode = dynamic_cast<osg::Geode *>(
      //         osg::clone(m_defaultStateSets[i].get(),
      //         osg::CopyOp::DEEP_COPY_NODES));
      //     m_deviceGroup->addChild(geode);
      //     // geode->setStateSet(default_geode->getStateSet());
      //     // osg::ref_ptr<osg::Geode> default_geode =
      //     m_defaultStateSets[i]->asGeode();
      //     // geode->setStateSet(default_geode->getStateSet());

      //     // osg::ref_ptr<osg::ShapeDrawable> shape =
      //     //     dynamic_cast<osg::ShapeDrawable *>(geode->getDrawable(0));
      //     // if (!shape) continue;
      //     // shape->build();
      //   }
    }
    // m_drawableBuilding->initDrawables();
    // for (auto drawable : m_drawableBuilding->getDrawables())
    //   m_deviceGroup->addChild(drawable);
    m_timestepColors.clear();
    m_sensorData.clear();
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
