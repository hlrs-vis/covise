#include "EnnovatisSystem.h"

#include <build_options.h>
#include <lib/ennovatis/building.h>
#include <lib/ennovatis/channel.h>
#include <lib/ennovatis/csv.h>
#include <lib/ennovatis/date.h>
#include <lib/ennovatis/rest.h>
#include <lib/ennovatis/sax.h>
#include <proj.h>
#include <utils/read/csv/csv.h>
#include <utils/string/LevenshteinDistane.h>

#include <fstream>
#include <iostream>
#include <regex>

#include "presentation/TxtInfoboard.h"

namespace {
constexpr bool debug = build_options.debug_ennovatis;
ennovatis::rest_request_handler m_debug_worker;
// regex for dd.mm.yyyy
const std::regex dateRgx(
    R"(((0[1-9])|([12][0-9])|(3[01]))\.((0[0-9])|(1[012]))\.((20[012]\d|19\d\d)|(1\d|2[0123])))");

// Compare two string numbers as integer using std::stoi
bool helper_compareStreetNumbers_as_int(const std::string &strtNo,
                                        const std::string &strtNo2) {
  try {
    int intStrtNo = std::stoi(strtNo), intStrtNo2 = std::stoi(strtNo2);
    auto validConversion =
        strtNo == std::to_string(intStrtNo) && strtNo2 == std::to_string(intStrtNo2);
    if (intStrtNo2 == intStrtNo && validConversion) return true;
  } catch (...) {
  }
  return false;
}

/** @brief Compares two string street numbers in the format ("<streetname>
 * <streetnumber>").
 *
 * The function compares the street numbers of the two street names as string
 * and integer. If the street numbers are equal, the function returns true.
 * @param strtName The first string street name.
 * @param strtName2 The second string street name.
 * @return true if the street numbers are equal, otherwise false.
 */
bool compareStreetNumbers(const std::string &strtName,
                          const std::string &strtName2) {
  auto strtNo = strtName.substr(strtName.find_last_of(" ") + 1);
  auto strtNo2 = strtName2.substr(strtName2.find_last_of(" ") + 1);

  // compare in lower case str
  auto lower = [](unsigned char c) { return std::tolower(c); };
  std::transform(strtNo2.begin(), strtNo2.end(), strtNo2.begin(), lower);
  std::transform(strtNo.begin(), strtNo.end(), strtNo.begin(), lower);
  if (strtNo2 == strtNo) return true;

  // compare as integers
  return helper_compareStreetNumbers_as_int(strtNo, strtNo2);
};

};  // namespace

using namespace ennovatis;

EnnovatisSystem::EnnovatisSystem(opencover::coVRPlugin *plugin,
                                 opencover::ui::Menu *parentMenu,
                                 osg::ref_ptr<osg::Switch> parent)
    : m_plugin(plugin),
      m_menu(nullptr),
      m_selectionsList(nullptr),
      m_enabledDeviceList(nullptr),
      m_channelList(nullptr),
      m_from(nullptr),
      m_to(nullptr),
      m_update(nullptr),
      m_ennovatis(new osg::Group()),
      m_parent(parent),
      m_enabled(false) {
  assert(parentMenu && "EnnovatisSystem: parent must not be null");
  assert(plugin && "EnnovatisSystem: plugin must not be null");
  initEnnovatisUI(parentMenu);
  m_parent->addChild(m_ennovatis);
}

EnnovatisSystem::~EnnovatisSystem() {
  if (m_parent) m_parent->removeChild(m_ennovatis);
}

void EnnovatisSystem::init() {
  m_ennovatis->setName("Ennovatis");

  initRESTRequest();

  if (!loadChannelIDs(
          m_plugin->configString("Ennovatis", "jsonChannelIdPath", "default")
              ->value(),
          m_plugin->configString("Ennovatis", "csvChannelIdPath", "default")
              ->value())) {
    std::cerr << "Failed to load channel IDs" << std::endl;
    return;
  }

  initEnnovatisDevices();
}

void EnnovatisSystem::enable(bool on) {
  m_enabled = on;
}

void EnnovatisSystem::update() {
  if (m_deviceSensors.empty()) return;

  // update the sensors
  for (auto &sensor : m_deviceSensors) sensor->update();
}

void EnnovatisSystem::updateTime(int timestep) {
  for (auto &sensor : m_deviceSensors) sensor->setTimestep(timestep);
}

void EnnovatisSystem::initEnnovatisUI(opencover::ui::Menu *parentMenu) {
  m_menu = new opencover::ui::Menu(parentMenu, "Ennovatis");
  m_menu->setText("Ennovatis");

  m_selectionsList =
      new opencover::ui::SelectionList(m_menu, "Ennovatis_ChannelType");
  m_selectionsList->setText("Channel Type: ");
  std::vector<std::string> ennovatisSelections;
  for (int i = 0; i < static_cast<int>(ChannelGroup::None); ++i)
    ennovatisSelections.push_back(
        ChannelGroupToString(static_cast<ChannelGroup>(i)));

  m_selectionsList->setList(ennovatisSelections);
  m_enabledDeviceList = new opencover::ui::SelectionList(m_menu, "Enabled_Devices");
  m_enabledDeviceList->setText("Enabled Devices: ");
  m_enabledDeviceList->setCallback([this](int value) { selectEnabledDevice(); });
  m_channelList = new opencover::ui::SelectionList(m_menu, "Channels");
  m_channelList->setText("Channels: ");

  // TODO: add calender widget instead of txtfields
  m_from = new opencover::ui::EditField(m_menu, "from");
  m_to = new opencover::ui::EditField(m_menu, "to");

  m_update = new opencover::ui::Button(m_menu, "Update");
  m_update->setCallback([this](bool on) { updateEnnovatis(); });

  m_selectionsList->setCallback(
      [this](int value) { setEnnovatisChannelGrp(ennovatis::ChannelGroup(value)); });
  m_from->setCallback(
      [this](const std::string &toSet) { setRESTDate(toSet, true); });
  m_to->setCallback([this](const std::string &toSet) { setRESTDate(toSet, false); });
}

void EnnovatisSystem::selectEnabledDevice() {
  auto selected = m_enabledDeviceList->selectedItem();
  for (auto &sensor : m_deviceSensors) {
    auto building = sensor->getDevice()->getBuildingInfo().building;
    if (building->getName() == selected) {
      sensor->disactivate();
      sensor->activate();
      return;
    }
  }
}

void EnnovatisSystem::updateEnnovatis() { updateEnnovatisChannelGrp(); }

void EnnovatisSystem::setRESTDate(const std::string &toSet, bool isFrom) {
  std::string fromOrTo = (isFrom) ? "From: " : "To: ";
  fromOrTo += toSet;
  if (!std::regex_match(toSet, dateRgx)) {
    std::cout << "Invalid date format for " << fromOrTo
              << " Please use the following format: " << ennovatis::date::dateformat
              << std::endl;
    return;
  }

  auto time = ennovatis::date::str_to_time_point(toSet, ennovatis::date::dateformat);
  bool validTime = (isFrom) ? (time <= m_req->dtt) : (time >= m_req->dtf);
  if (!validTime) {
    std::cout << "Invalid date. (To >= From)" << std::endl;
    if (isFrom)
      m_from->setValue(ennovatis::date::time_point_to_str(
          m_req->dtf, ennovatis::date::dateformat));
    else
      m_to->setValue(ennovatis::date::time_point_to_str(
          m_req->dtt, ennovatis::date::dateformat));
    return;
  }

  if (isFrom)
    m_req->dtf = time;
  else
    m_req->dtt = time;
}

CylinderAttributes EnnovatisSystem::getCylinderAttributes() {
  auto configDefaultColorVec =
      m_plugin
          ->configFloatArray("Ennovatis", "defaultColorCylinder",
                             std::vector<double>{0, 0, 0, 1.f})
          ->value();
  auto configMaxColorVec =
      m_plugin
          ->configFloatArray("Ennovatis", "maxColorCylinder",
                             std::vector<double>{0.0, 0.1, 0.0, 1.f})
          ->value();
  auto configMinColorVec =
      m_plugin
          ->configFloatArray("Ennovatis", "minColorCylinder",
                             std::vector<double>{0.0, 1.0, 0.0, 1.f})
          ->value();
  auto configDefaultHeightCycl =
      m_plugin->configFloat("Ennovatis", "defaultHeightCylinder", 100.0)->value();
  auto configRadiusCycl =
      m_plugin->configFloat("Ennovatis", "radiusCylinder", 3.0)->value();
  auto defaultColor = osg::Vec4(configDefaultColorVec[0], configDefaultColorVec[1],
                                configDefaultColorVec[2], configDefaultColorVec[3]);
  auto maxColor = osg::Vec4(configMaxColorVec[0], configMaxColorVec[1],
                            configMaxColorVec[2], configMaxColorVec[3]);
  auto minColor = osg::Vec4(configMinColorVec[0], configMinColorVec[1],
                            configMinColorVec[2], configMinColorVec[3]);
  return CylinderAttributes(configRadiusCycl, configDefaultHeightCycl, maxColor,
                            minColor, defaultColor);
}

void EnnovatisSystem::initEnnovatisDevices() {
  const auto projFrom =
      m_plugin->configString("General", "projFrom", "default")->value();
  const auto projTo =
      m_plugin->configString("General", "projTo", "default")->value();
  const auto offset =
      m_plugin->configFloatArray("General", "offset", std::vector<double>{0, 0, 0})
          ->value();
  auto P =
      proj_create_crs_to_crs(PJ_DEFAULT_CTX, projFrom.c_str(), projTo.c_str(), NULL);
  PJ_COORD coord;
  coord.lpzt.z = 0.0;
  coord.lpzt.t = HUGE_VAL;

  if (!P)
    fprintf(stderr,
            "EnnovatisSystem: Ignore mapping. No valid projection was "
            "found between given proj string in "
            "config EnergyCampus.toml\n");

  m_ennovatis->removeChildren(0, m_ennovatis->getNumChildren());
  m_deviceSensors.clear();
  auto cylinderAttributes = getCylinderAttributes();
  for (auto &b : m_buildings) {
    auto &lat = b.getY();
    auto &lon = b.getX();
    coord.lpzt.lam = lon;
    coord.lpzt.phi = lat;

    coord = proj_trans(P, PJ_FWD, coord);

    b.setX(coord.xy.x + offset[0]);  // x
    b.setY(coord.xy.y + offset[1]);  // y
    b.setHeight(offset[2]);
    cylinderAttributes.position = osg::Vec3(b.getX(), b.getY(), b.getHeight());
    auto drawableBuilding = std::make_unique<PrototypeBuilding>(cylinderAttributes);
    auto infoboardPos = osg::Vec3(b.getX() + cylinderAttributes.radius + 5,
                                  b.getY() + cylinderAttributes.radius + 5,
                                  b.getHeight() + cylinderAttributes.height);
    auto infoboard = std::make_unique<TxtInfoboard>(
        infoboardPos, b.getName(), "DroidSans-Bold.ttf",
        cylinderAttributes.radius * 20, cylinderAttributes.radius * 21, 2.0f, 0.1,
        2);
    auto enDev = std::make_unique<EnnovatisDevice>(
        b, m_channelList, m_req, m_channelGrp, std::move(infoboard),
        std::move(drawableBuilding));
    m_ennovatis->addChild(enDev->getDeviceGroup());
    m_deviceSensors.push_back(std::make_unique<EnnovatisDeviceSensor>(
        std::move(enDev), enDev->getDeviceGroup(), m_enabledDeviceList));
  }
  proj_destroy(P);
}

void EnnovatisSystem::updateEnnovatisChannelGrp() {
  for (auto &sensor : m_deviceSensors)
    sensor->getDevice()->setChannelGroup(m_channelGrp);
}

void EnnovatisSystem::setEnnovatisChannelGrp(ennovatis::ChannelGroup group) {
  core::utils::osgUtils::switchTo(m_ennovatis, m_parent);
  m_channelGrp = std::make_shared<ennovatis::ChannelGroup>(group);

  if constexpr (debug) {
    auto &b = m_buildings.at(0);
    m_debug_worker.fetchChannels(group, b, *m_req);
  }
  updateEnnovatisChannelGrp();
}

bool EnnovatisSystem::updateChannelIDsFromCSV(const std::string &pathToCSV) {
  auto csvPath = std::filesystem::path(pathToCSV);
  if (csvPath.extension() == ".csv") {
    std::ifstream csvFilestream(pathToCSV);
    if (!csvFilestream.is_open()) {
      std::cout << "File does not exist or cannot be opened: " << pathToCSV
                << std::endl;
      return false;
    }
    ennovatis::csv_channelid_parser csvParser;
    if (!csvParser.update_buildings_by_buildingid(pathToCSV, m_buildings))
      return false;
  }
  return true;
}

bool EnnovatisSystem::loadChannelIDs(const std::string &pathToJSON,
                                     const std::string &pathToCSV) {
  std::ifstream inputFilestream(pathToJSON);
  if (!inputFilestream.is_open()) {
    std::cout << "File does not exist or cannot be opened: " << pathToJSON
              << std::endl;
    return false;
  }
  auto jsonPath = std::filesystem::path(pathToJSON);
  if (jsonPath.extension() == ".json") {
    // init buildings
    ennovatis::sax_channelid_parser sax(&m_buildings);
    if (!sax.parse_filestream(inputFilestream)) return false;

    // update new channelids
    if (!updateChannelIDsFromCSV(pathToCSV)) return false;

    if constexpr (debug)
      for (auto &log : sax.getDebugLogs()) std::cout << log << std::endl;
  }
  return true;
}

void EnnovatisSystem::initRESTRequest() {
  m_req = std::make_shared<rest_request>(rest_request());
  m_req->url = m_plugin->configString("Ennovatis", "restUrl", "default")->value();
  m_req->projEid =
      m_plugin->configString("Ennovatis", "projEid", "default")->value();
  m_req->channelId = "";
  m_req->dtf = std::chrono::system_clock::now() - std::chrono::hours(24);
  m_req->dtt = std::chrono::system_clock::now();
  m_from->setValue(
      ennovatis::date::time_point_to_str(m_req->dtf, ennovatis::date::dateformat));
  m_to->setValue(
      ennovatis::date::time_point_to_str(m_req->dtt, ennovatis::date::dateformat));
}
