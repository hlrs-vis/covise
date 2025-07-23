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
// #include "ui/legacy/DeviceSensor.h"

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
  void updateEnnovatisChannelGrp();
  void setEnnovatisChannelGrp(ennovatis::ChannelGroup group);
  bool updateChannelIDsFromCSV(const std::string &pathToCSV);
  bool loadChannelIDs(const std::string &pathToJSON, const std::string &pathToCSV);
  void initRESTRequest();
  CylinderAttributes getCylinderAttributes();

  void selectEnabledDevice();
  void updateEnnovatis();
  void setRESTDate(const std::string &toSet, bool isFrom = false);

  std::vector<std::unique_ptr<EnnovatisDeviceSensor>> m_ennovatisDeviceSensors;

  opencover::coVRPlugin *m_plugin;
  opencover::ui::Menu *m_menu;
  opencover::ui::SelectionList *m_ennovatisSelectionsList;
  opencover::ui::SelectionList *m_enabledDeviceList;
  opencover::ui::SelectionList *m_channelList;
  opencover::ui::EditField *m_from;
  opencover::ui::EditField *m_to;
  opencover::ui::Button *m_update;

  osg::ref_ptr<osg::Group> m_ennovatis;
  osg::ref_ptr<osg::Switch> m_switch;

  std::shared_ptr<ennovatis::rest_request> m_req;
  std::shared_ptr<ennovatis::ChannelGroup> m_channelGrp;
  ennovatis::Buildings m_buildings;
  bool m_enabled;
};
