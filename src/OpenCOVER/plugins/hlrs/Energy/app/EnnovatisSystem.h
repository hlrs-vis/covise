/**
 * @class EnnovatisSystem
 * @brief Manages the integration of Ennovatis energy system within the OpenCOVER plugin framework.
 *
 * This class implements the core::interface::ISystem interface to provide functionality for
 * initializing, enabling, updating, and managing Ennovatis devices and channels in a 3D visualization environment.
 * It handles UI components, REST requests, device sensors, and data loading from CSV/JSON files.
 *
 * @note Copy and move operations are deleted to ensure unique ownership and prevent unintended duplication.
 *
 * @param plugin Pointer to the OpenCOVER plugin instance.
 * @param parentMenu Pointer to the parent UI menu for integration.
 * @param parent OSG Switch node for scene graph management.
 *
 * @section Responsibilities
 * - Initialize Ennovatis UI and devices.
 * - Handle REST requests and channel group selection.
 * - Manage device sensors and their attributes.
 * - Load and update channel IDs from external files.
 * - Provide time-based updates and enable/disable functionality.
 *
 * @section Member Variables
 * - m_deviceSensors: List of managed device sensors.
 * - m_plugin, m_menu: References to plugin and UI menu.
 * - m_selectionsList, m_enabledDeviceList, m_channelList: UI selection lists.
 * - m_from, m_to: UI edit fields for date selection.
 * - m_update: UI button for triggering updates.
 * - m_ennovatis, m_parent: OSG nodes for scene management.
 * - m_req: REST request handler.
 * - m_channelGrp: Current channel group.
 * - m_buildings: Building data.
 * - m_enabled: System enabled state.
 */
#pragma once

#include <cover/coVRPlugin.h>
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>
#include <cover/ui/SelectionList.h>
#include <lib/core/interfaces/ISystem.h>
#include <lib/ennovatis/building.h>
#include <lib/ennovatis/rest.h>

#include <memory>
#include <osg/Switch>
#include <string>

#include "presentation/PrototypeBuilding.h"
#include "ui/ennovatis/EnnovatisDeviceSensor.h"

class EnnovatisSystem final : public core::interface::ISystem {
 public:
  EnnovatisSystem(opencover::coVRPlugin *plugin, opencover::ui::Menu *parentMenu,
                  osg::ref_ptr<osg::Switch> parent);
  virtual ~EnnovatisSystem();

  EnnovatisSystem(const EnnovatisSystem &) = delete;
  EnnovatisSystem &operator=(const EnnovatisSystem &) = delete;
  EnnovatisSystem(EnnovatisSystem &&) = delete;
  EnnovatisSystem &operator=(EnnovatisSystem &&) = delete;

  void init() override;
  void enable(bool on) override;
  void update() override;
  void updateTime(int timestep) override;
  bool isEnabled() const override { return m_enabled; }

 private:
  void initEnnovatisUI(opencover::ui::Menu *parentMenu);
  void initEnnovatisDevices();
  void initRESTRequest();
  void selectEnabledDevice();
  void setEnnovatisChannelGrp(ennovatis::ChannelGroup group);
  void setRESTDate(const std::string &toSet, bool isFrom = false);
  void updateEnnovatis();
  void updateEnnovatisChannelGrp();
  bool updateChannelIDsFromCSV(const std::string &pathToCSV);
  bool loadChannelIDs(const std::string &pathToJSON, const std::string &pathToCSV);
  CylinderAttributes getCylinderAttributes();

  std::vector<std::unique_ptr<EnnovatisDeviceSensor>> m_deviceSensors;

  opencover::coVRPlugin *m_plugin;
  opencover::ui::Menu *m_menu;
  opencover::ui::SelectionList *m_selectionsList;
  opencover::ui::SelectionList *m_enabledDeviceList;
  opencover::ui::SelectionList *m_channelList;
  opencover::ui::EditField *m_from;
  opencover::ui::EditField *m_to;
  opencover::ui::Button *m_update;

  osg::ref_ptr<osg::Group> m_ennovatis;
  osg::ref_ptr<osg::Switch> m_parent;

  std::shared_ptr<ennovatis::rest_request> m_req;
  std::shared_ptr<ennovatis::ChannelGroup> m_channelGrp;
  ennovatis::Buildings m_buildings;
  bool m_enabled;
};
