#pragma once
// core
#include <lib/core/interfaces/IBuilding.h>
#include <lib/core/interfaces/IInfoboard.h>

// ennovatis
#include <lib/core/utils/osgUtils.h>
#include <lib/ennovatis/building.h>
#include <lib/ennovatis/channel.h>
#include <lib/ennovatis/json.h>
#include <lib/ennovatis/rest.h>

// cover
#include <cover/coBillboard.h>
#include <cover/coVRMSController.h>
#include <cover/ui/SelectionList.h>

// osg
#include <osg/Geode>
#include <osg/Group>
#include <osg/NodeVisitor>
#include <osg/Shape>
#include <osg/StateSet>
#include <osg/Vec4>
#include <osg/ref_ptr>
#include <osgText/Text>

// std
#include <memory>

/**
 * @class EnnovatisDevice
 * @brief Represents a device in the Ennovatis energy management system, providing visualization and interaction capabilities.
 *
 * This class encapsulates the logic for managing and visualizing an Ennovatis device, including its building information,
 * associated channels, sensor data, and UI elements. It interacts with REST APIs to fetch data, updates visual representations
 * based on time steps, and manages user interactions through selection lists and info boards.
 *
 * Dependencies:
 * - Ennovatis core and REST interfaces
 * - OpenCOVER UI and visualization components
 * - OpenSceneGraph (OSG) for 3D graphics
 *
 * Key Responsibilities:
 * - Initialize and manage device visualization
 * - Fetch and handle data from REST endpoints
 * - Update UI elements such as selection lists and info boards
 * - Visualize sensor data through color and height changes
 * - Synchronize state across multiple VR clients
 *
 * @constructor
 * @param building Reference to the Ennovatis building object
 * @param channelList Pointer to the UI selection list for channels
 * @param req Shared pointer to the REST request object
 * @param channelGroup Shared pointer to the channel group
 * @param infoBoard Unique pointer to the info board interface
 * @param drawableBuilding Unique pointer to the drawable building interface
 *
 * @note This class is not copyable or movable.
 */
class EnnovatisDevice {
 public:
  EnnovatisDevice(
      const ennovatis::Building &building, opencover::ui::SelectionList *channelList,
      std::shared_ptr<ennovatis::rest_request> req,
      std::shared_ptr<ennovatis::ChannelGroup> channelGroup,
      std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
      std::unique_ptr<core::interface::IBuilding> &&drawableBuilding);

  void update();
  void activate();
  void disactivate();
  void setChannelGroup(std::shared_ptr<ennovatis::ChannelGroup> group);
  void setTimestep(int timestep);
  [[nodiscard]] const auto &getBuildingInfo() const { return m_buildingInfo; }
  [[nodiscard]] osg::ref_ptr<osg::Group> getDeviceGroup() { return m_deviceGroup; }

 private:
  struct BuildingInfo {
    BuildingInfo(const ennovatis::Building *b) : building(b) {}
    const ennovatis::Building *building;
    std::vector<std::string> channelResponse;
  };
  typedef std::unique_ptr<osg::Vec4> TimestepColor;
  typedef std::vector<TimestepColor> TimestepColorList;
  typedef std::vector<float> SensorData;

  void init();
  void fetchData();
  void updateChannelSelectionList();
  void setChannel(int idx);
  void updateColorByTime(int timestep);
  void updateHeightByTime(int timestep);
  void createTimestepColorList(const ennovatis::json_response_object &j_resp_obj);
  void updateInfoboard(const std::string &info);
  bool handleResponse(const std::vector<std::string> &results);

  [[nodiscard]] int getSelectedChannelIdx() const;
  [[nodiscard]] auto getSelectedChannelIterator() const;
  [[nodiscard]] auto getResponseObjectForSelectedChannel() const;
  [[nodiscard]] auto createBillboardTxt(
      const ennovatis::json_response_object &j_resp_obj);

  osg::ref_ptr<osg::Group> m_deviceGroup = nullptr;
  std::unique_ptr<core::interface::IInfoboard<std::string>> m_infoBoard;
  std::unique_ptr<core::interface::IBuilding> m_drawableBuilding;
  std::weak_ptr<ennovatis::rest_request> m_request;
  std::weak_ptr<ennovatis::ChannelGroup> m_channelGroup;
  opencover::ui::SelectionList *m_channelSelectionList;

  bool m_InfoVisible = false;
  BuildingInfo m_buildingInfo;
  ennovatis::rest_request_handler m_restWorker;
  opencover::coVRMSController *m_opncvrCtrl;  // cannot be const because syncing
                                              // methods are not const correct
  core::utils::osgUtils::Geodes m_defaultStateSets;
  TimestepColorList m_timestepColors;
  SensorData m_sensorData;
  float m_consumptionPerArea = 0.0f;
};
