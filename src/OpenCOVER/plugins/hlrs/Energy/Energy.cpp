/****************************************************************************\
 **                                                          (C)2024 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading building energy data        **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern, Marko Djuric                                       **
 **                                                                        **
 ** TODO:                                                                  **
 **  [ ] fetch lat lon from googlemaps                                     **
 **  [x] make REST lib independent from ennovatis general use              **
 **  [x] update via REST in background                                     **
 **                                                                        **
 ** History:                                                               **
 **  2024  v1                                                              **
 **  Marko Djuric 02.2024: add ennovatis client                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <Energy.h>

#include <Device.h>
#include <EnnovatisDevice.h>
#include <EnnovatisDeviceSensor.h>
#include <build_options.h>
#include <cover/ui/SelectionList.h>
#include <core/TxtInfoboard.h>
#include <core/PrototypeBuilding.h>
#include <core/CityGMLBuilding.h>
#include <core/utils/osgUtils.h>
#include <ennovatis/building.h>
#include <ennovatis/date.h>
#include <ennovatis/sax.h>
#include <ennovatis/rest.h>
#include <ennovatis/csv.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <filesystem>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Switch>
#include <osg/Vec3>
#include <osg/ref_ptr>
#include <string>
#include <vector>
#include <algorithm>
#include <regex>

#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>

#include <cover/coVRAnimationManager.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <config/CoviseConfig.h>

#include <osg/LineWidth>
#include <osg/Version>

using namespace opencover;
using namespace opencover::utils::read;
using namespace energy;

namespace {

constexpr bool debug = build_options.debug_ennovatis;
// regex for dd.mm.yyyy
const std::regex dateRgx(R"(((0[1-9])|([12][0-9])|(3[01]))\.((0[0-9])|(1[012]))\.((20[012]\d|19\d\d)|(1\d|2[0123])))");
ennovatis::rest_request_handler m_debug_worker;

// Compare two string numbers as integer using std::stoi
bool helper_cmpStrNo_as_int(const std::string &strtNo, const std::string &strtNo2)
{
    try {
        int intStrtNo = std::stoi(strtNo), intStrtNo2 = std::stoi(strtNo2);
        auto validConversion = strtNo == std::to_string(intStrtNo) && strtNo2 == std::to_string(intStrtNo2);
        if (intStrtNo2 == intStrtNo && validConversion)
            return true;
    } catch (...) {
    }
    return false;
}

/**
 * @brief Compares two string street numbers in the format ("<streetname> <streetnumber>").
 *
 * The function compares the street numbers of the two street names as string and integer. If the street numbers are equal, the function returns true.
 * @param strtName The first string street name.
 * @param strtName2 The second string street name.
 * @return true if the street numbers are equal, otherwise false.
 */
bool cmpStrtNo(const std::string &strtName, const std::string &strtName2)
{
    auto strtNo = strtName.substr(strtName.find_last_of(" ") + 1);
    auto strtNo2 = strtName2.substr(strtName2.find_last_of(" ") + 1);

    // compare in lower case str
    auto lower = [](unsigned char c) {
        return std::tolower(c);
    };
    std::transform(strtNo2.begin(), strtNo2.end(), strtNo2.begin(), lower);
    std::transform(strtNo.begin(), strtNo.end(), strtNo.begin(), lower);
    if (strtNo2 == strtNo)
        return true;

    // compare as integers
    return helper_cmpStrNo_as_int(strtNo, strtNo2);
};

// Case-sensitive char comparer
bool helper_cmpChar(char a, char b)
{
    return (a == b);
}

// Case-insensitive char comparer
bool helper_cmpCharIgnoreCase(char a, char b)
{
    return (std::tolower(a) == std::tolower(b));
}

/**
 * Computes the Levenshtein distance between two strings. 0 means the strings are equal.
 * The higher the number, the more different chars are in the strings.
 * e.g. "kitten" and "sitting" have a Levenshtein distance of 3.
 * Source: http://www.blackbeltcoder.com/Articles/algorithms/approximate-string-comparisons-using-levenshtein-distance
 *
 * @param s1 The first string.
 * @param s2 The second string.
 * @param ignoreCase Flag indicating whether to ignore case sensitivity (default: false).
 * @return The Levenshtein distance between the two strings.
 */
size_t computeLevensteinDistance(const std::string &s1, const std::string &s2, bool ignoreCase = false)
{
    const auto &len1 = s1.size(), len2 = s2.size();

    // allocate distance matrix
    std::vector<std::vector<size_t>> d(len1 + 1, std::vector<size_t>(len2 + 1));

    auto isEqual = [&](char a, char b) {
        return (ignoreCase) ? helper_cmpCharIgnoreCase(a, b) : helper_cmpChar(a, b);
    };

    d[0][0] = 0;
    // compute distance
    for (int i = 1; i <= len1; ++i)
        d[i][0] = i;
    for (int j = 1; j <= len2; ++j)
        d[0][j] = j;

    for (int i = 1; i <= len1; ++i) {
        for (int j = 1; j <= len2; ++j) {
            if (isEqual(s1[i - 1], s2[j - 1]))
                d[i][j] = d[i - 1][j - 1];
            else
                d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1});
        }
    }

    return d[len1][len2];
}
} // namespace

EnergyPlugin *EnergyPlugin::m_plugin = nullptr;

EnergyPlugin::EnergyPlugin(): coVRPlugin(COVER_PLUGIN_NAME), ui::Owner("EnergyPlugin", cover->ui)
{
    fprintf(stderr, "Starting Energy Plugin\n");
    m_plugin = this;

    m_buildings = std::make_unique<ennovatis::Buildings>();

    m_EnergyGroup = new osg::Group();
    m_EnergyGroup->setName("Energy");
    cover->getObjectsRoot()->addChild(m_EnergyGroup);

    m_sequenceList = new osg::Sequence();
    m_sequenceList->setName("DB");
    m_ennovatis = new osg::Group();
    m_ennovatis->setName("Ennovatis");
		m_cityGML = new osg::Group();
		m_cityGML->setName("CityGML");

    osg::ref_ptr<osg::MatrixTransform> EnergyGroupMT = new osg::MatrixTransform();

    m_switch = new osg::Switch();
    m_switch->setName("Switch");
    m_switch->addChild(m_sequenceList);
    m_switch->addChild(m_ennovatis);
    m_switch->addChild(m_cityGML);

    EnergyGroupMT->addChild(m_switch);
    m_EnergyGroup->addChild(EnergyGroupMT);

    GDALAllRegister();

    m_SDlist.clear();

    EnergyTab = new ui::Menu("Energy Campus", EnergyPlugin::m_plugin);
    EnergyTab->setText("Energy Campus");

    // db
    componentGroup = new ui::ButtonGroup(EnergyTab, "ComponentGroup");
    componentGroup->setDefaultValue(Strom);
    componentList = new ui::Group(EnergyTab, "Component");
    componentList->setText("Messwerte (jährlich)");
    StromBt = new ui::Button(componentList, "Strom", componentGroup, Strom);
    WaermeBt = new ui::Button(componentList, "Waerme", componentGroup, Waerme);
    KaelteBt = new ui::Button(componentList, "Kaelte", componentGroup, Kaelte);
    componentGroup->setCallback([this](int value) { setComponent(Components(value)); });

    initEnnovatisUI();
    initCityGMLUI();

    m_offset = configFloatArray("General", "offset", std::vector<double>{0, 0, 0})->value();
}

EnergyPlugin::~EnergyPlugin() {
  auto root = cover->getObjectsRoot();

  if (m_cityGML) {
    for (auto i = 0; i < m_cityGML->getNumChildren(); ++i) {
      auto child = m_cityGML->getChild(i);
      root->addChild(child);
    }
    core::utils::osgUtils::deleteChildrenFromOtherGroup(m_cityGML, root);
  }

  if (m_EnergyGroup) {
    root->removeChild(m_EnergyGroup.get());
  }

  m_plugin = nullptr;
}

void EnergyPlugin::initCityGMLUI()
{
    m_cityGMLGroup = new ui::Group(EnergyTab, "CityGML");
    m_cityGMLEnable = new ui::Button(m_cityGMLGroup, "Enable");
    m_cityGMLEnable->setCallback([this](bool on) { enableCityGML(on); });
}

void EnergyPlugin::enableCityGML(bool on) {
  if (on) {
    if (m_cityGMLObjs.empty()) {
      auto root = cover->getObjectsRoot();
      for (auto i = 0; i < root->getNumChildren(); ++i) {
        osg::ref_ptr<osg::MatrixTransform> child =
            dynamic_cast<osg::MatrixTransform *>(root->getChild(i));
        if (child) {
          auto name = child->getName();
          if (name.find(".gml") != std::string::npos) {
            addCityGMLObjects(child);
            m_cityGML->addChild(child);
          }
        }
      }
      core::utils::osgUtils::deleteChildrenFromOtherGroup(root, m_cityGML);
    }
    switchTo(m_cityGML);
  } else {
    switchTo(m_sequenceList);
  }
}

void EnergyPlugin::addCityGMLObjects(osg::MatrixTransform *node) {
  for (auto i = 0; i < node->getNumChildren(); ++i) {
    osg::ref_ptr<osg::Group> child =
        dynamic_cast<osg::Group *>(node->getChild(i));
    if (child) {
      auto name = child->getName();
      if (m_cityGMLObjs.find(name) != m_cityGMLObjs.end())
        continue;

      if (!child->getNumChildren())
        continue;

      if (osg::ref_ptr<osg::Geode> geo =
              dynamic_cast<osg::Geode *>(child->getChild(0))) {
        auto boundingbox = geo->getBoundingBox();
        auto infoboardPos = geo->getBound().center() ;
        infoboardPos.z() +=
            (boundingbox.zMax() - boundingbox.zMin()) / 2 + boundingbox.zMin();
        auto infoboard = std::make_unique<core::TxtInfoboard>(
            infoboardPos, name, "DroidSans-Bold.ttf", 20, 21, 2.0f, 0.1, 2);
        auto building = std::make_unique<core::CityGMLBuilding>(geo);
        auto sensor = std::make_unique<CityGMLDeviceSensor>(
            child, std::move(infoboard), std::move(building));
        m_cityGMLObjs.insert({name, std::move(sensor)});
      }
    }
  }
}

void EnergyPlugin::initEnnovatisUI()
{
    m_ennovatisGroup = new ui::Group(EnergyTab, "Ennovatis");
    m_ennovatisGroup->setText("Ennovatis");

    m_ennovatisSelectionsList = new ui::SelectionList(m_ennovatisGroup, "Ennovatis ChannelType: ");
    std::vector<std::string> ennovatisSelections;
    for (int i = 0; i < static_cast<int>(ennovatis::ChannelGroup::None); ++i)
        ennovatisSelections.push_back(ennovatis::ChannelGroupToString(static_cast<ennovatis::ChannelGroup>(i)));

    m_ennovatisSelectionsList->setList(ennovatisSelections);
    m_enabledEnnovatisDevices = new opencover::ui::SelectionList(EnergyTab, "Enabled Devices: ");
    m_enabledEnnovatisDevices->setCallback([this](int value) { selectEnabledDevice(); });
    m_ennovatisChannelList = new opencover::ui::SelectionList(EnergyTab, "Channels: ");

    // TODO: add calender widget instead of txtfields
    m_ennovatisFrom = new ui::EditField(EnergyTab, "from");
    m_ennovatisTo = new ui::EditField(EnergyTab, "to");

    m_ennovatisUpdate = new ui::Button(m_ennovatisGroup, "Update");
    m_ennovatisUpdate->setCallback([this](bool on) { updateEnnovatis(); });

    m_ennovatisSelectionsList->setCallback(
        [this](int value) { setEnnovatisChannelGrp(ennovatis::ChannelGroup(value)); });
    m_ennovatisFrom->setCallback([this](const std::string &toSet) { setRESTDate(toSet, true); });
    m_ennovatisTo->setCallback([this](const std::string &toSet) { setRESTDate(toSet, false); });
}

void EnergyPlugin::selectEnabledDevice()
{
    auto selected = m_enabledEnnovatisDevices->selectedItem();
    for (auto &sensor: m_ennovatisDevicesSensors) {
        auto building = sensor->getDevice()->getBuildingInfo().building;
        if (building->getName() == selected) {
            sensor->disactivate();
            sensor->activate();
            return;
        }
    }
}

void EnergyPlugin::updateEnnovatis()
{
    updateEnnovatisChannelGrp();
}

void EnergyPlugin::setRESTDate(const std::string &toSet, bool isFrom = false)
{
    std::string fromOrTo = (isFrom) ? "From: " : "To: ";
    fromOrTo += toSet;
    if (!std::regex_match(toSet, dateRgx)) {
        std::cout << "Invalid date format for " << fromOrTo
                  << " Please use the following format: " << ennovatis::date::dateformat << std::endl;
        return;
    }

    auto time = ennovatis::date::str_to_time_point(toSet, ennovatis::date::dateformat);
    bool validTime = (isFrom) ? (time <= m_req->dtt) : (time >= m_req->dtf);
    if (!validTime) {
        std::cout << "Invalid date. (To >= From)" << std::endl;
        if (isFrom)
            m_ennovatisFrom->setValue(ennovatis::date::time_point_to_str(m_req->dtf, ennovatis::date::dateformat));
        else
            m_ennovatisTo->setValue(ennovatis::date::time_point_to_str(m_req->dtt, ennovatis::date::dateformat));
        return;
    }

    if (isFrom)
        m_req->dtf = time;
    else
        m_req->dtt = time;
}

void EnergyPlugin::reinitDevices(int comp)
{
    for (auto s: m_SDlist) {
        if (s.second.empty())
            continue;
        for (auto devSens: s.second) {
            auto t = devSens->getDevice();
            t->init(rad, scaleH, comp);
        }
    }
}

core::CylinderAttributes EnergyPlugin::getCylinderAttributes()
{
    auto configDefaultColorVec =
        configFloatArray("Ennovatis", "defaultColorCylinder", std::vector<double>{0, 0, 0, 1.f})->value();
    auto configMaxColorVec =
        configFloatArray("Ennovatis", "maxColorCylinder", std::vector<double>{0.0, 0.1, 0.0, 1.f})->value();
    auto configMinColorVec =
        configFloatArray("Ennovatis", "minColorCylinder", std::vector<double>{0.0, 1.0, 0.0, 1.f})->value();
    auto configDefaultHeightCycl = configFloat("Ennovatis", "defaultHeightCylinder", 100.0)->value();
    auto configRadiusCycl = configFloat("Ennovatis", "radiusCylinder", 3.0)->value();
    auto defaultColor = osg::Vec4(configDefaultColorVec[0], configDefaultColorVec[1], configDefaultColorVec[2],
                                  configDefaultColorVec[3]);
    auto maxColor = osg::Vec4(configMaxColorVec[0], configMaxColorVec[1], configMaxColorVec[2], configMaxColorVec[3]);
    auto minColor = osg::Vec4(configMinColorVec[0], configMinColorVec[1], configMinColorVec[2], configMinColorVec[3]);
    return core::CylinderAttributes(configRadiusCycl, configDefaultHeightCycl, maxColor, minColor, defaultColor);
}

void EnergyPlugin::initEnnovatisDevices()
{
    m_ennovatis->removeChildren(0, m_ennovatis->getNumChildren());
    m_ennovatisDevicesSensors.clear();
    auto cylinderAttributes = getCylinderAttributes();
    for (auto &b: *m_buildings) {
        cylinderAttributes.position = osg::Vec3(b.getX(), b.getY(), b.getHeight());
        auto drawableBuilding = std::make_unique<core::PrototypeBuilding>(cylinderAttributes);
        auto infoboardPos =
            osg::Vec3(b.getX() + cylinderAttributes.radius + 5, b.getY() + cylinderAttributes.radius + 5,
                      b.getHeight() + cylinderAttributes.height);
        auto infoboard = std::make_unique<core::TxtInfoboard>(infoboardPos, b.getName(), "DroidSans-Bold.ttf",
                                                              cylinderAttributes.radius * 20,
                                                              cylinderAttributes.radius * 21, 2.0f, 0.1, 2);
        auto enDev = std::make_unique<EnnovatisDevice>(b, m_ennovatisChannelList, m_req, m_channelGrp,
                                                       std::move(infoboard), std::move(drawableBuilding));
        m_ennovatis->addChild(enDev->getDeviceGroup());
        m_ennovatisDevicesSensors.push_back(std::make_unique<EnnovatisDeviceSensor>(
            std::move(enDev), enDev->getDeviceGroup(), m_enabledEnnovatisDevices));
    }
}

void EnergyPlugin::updateEnnovatisChannelGrp()
{
    for (auto &sensor: m_ennovatisDevicesSensors)
        sensor->getDevice()->setChannelGroup(m_channelGrp);
}

void EnergyPlugin::setEnnovatisChannelGrp(ennovatis::ChannelGroup group)
{
    switchTo(m_ennovatis);
    m_channelGrp = std::make_shared<ennovatis::ChannelGroup>(group);

    if constexpr (debug) {
        auto &b = m_buildings->at(0);
        m_debug_worker.fetchChannels(group, b, *m_req);
    }
    updateEnnovatisChannelGrp();
}

void EnergyPlugin::switchTo(const osg::ref_ptr<osg::Node> child) {
  m_switch->setAllChildrenOff();
  m_switch->setChildValue(child, true);
}

void EnergyPlugin::setComponent(Components c)
{
    switchTo(m_sequenceList);
    switch (c) {
    case Strom:
        StromBt->setState(true, false);
        break;
    case Waerme:
        WaermeBt->setState(true, false);
        break;
    case Kaelte:
        KaelteBt->setState(true, false);
        break;
    default:
        break;
    }
    m_selectedComp = c;
    reinitDevices(c);
}

bool EnergyPlugin::loadDB(const std::string &path, const ProjTrans &projTrans)
{
    if (!loadDBFile(path, projTrans)) {
        return false;
    }

    if ((int)m_sequenceList->getNumChildren() > coVRAnimationManager::instance()->getNumTimesteps()) {
        coVRAnimationManager::instance()->setNumTimesteps(m_sequenceList->getNumChildren(), m_sequenceList);
    }

    rad = 3.;
    scaleH = 0.1;

    reinitDevices(m_selectedComp);
    return true;
}

bool EnergyPlugin::updateChannelIDsFromCSV(const std::string &pathToCSV)
{
    auto csvPath = std::filesystem::path(pathToCSV);
    if (csvPath.extension() == ".csv") {
        std::ifstream csvFilestream(pathToCSV);
        if (!csvFilestream.is_open()) {
            std::cout << "File does not exist or cannot be opened: " << pathToCSV << std::endl;
            return false;
        }
        ennovatis::csv_channelid_parser csvParser;
        if (!csvParser.update_buildings_by_buildingid(pathToCSV, m_buildings))
            return false;
    }
    return true;
}

bool EnergyPlugin::loadChannelIDs(const std::string &pathToJSON, const std::string &pathToCSV)
{
    std::ifstream inputFilestream(pathToJSON);
    if (!inputFilestream.is_open()) {
        std::cout << "File does not exist or cannot be opened: " << pathToJSON << std::endl;
        return false;
    }
    auto jsonPath = std::filesystem::path(pathToJSON);
    if (jsonPath.extension() == ".json") {
        ennovatis::sax_channelid_parser slp(m_buildings);
        if (!slp.parse_filestream(inputFilestream))
            return false;

        if (!updateChannelIDsFromCSV(pathToCSV))
            return false;

        if constexpr (debug)
            for (auto &log: slp.getDebugLogs())
                std::cout << log << std::endl;
    }
    return true;
}

void EnergyPlugin::initRESTRequest()
{
    m_req = std::make_shared<ennovatis::rest_request>();
    m_req->url = configString("Ennovatis", "restUrl", "default")->value();
    m_req->projEid = configString("Ennovatis", "projEid", "default")->value();
    m_req->channelId = "";
    m_req->dtf = std::chrono::system_clock::now() - std::chrono::hours(24);
    m_req->dtt = std::chrono::system_clock::now();
    m_ennovatisFrom->setValue(ennovatis::date::time_point_to_str(m_req->dtf, ennovatis::date::dateformat));
    m_ennovatisTo->setValue(ennovatis::date::time_point_to_str(m_req->dtt, ennovatis::date::dateformat));
}

std::unique_ptr<EnergyPlugin::const_buildings> EnergyPlugin::updateEnnovatisBuildings(const DeviceList &deviceList)
{
    auto lastDst = 0;
    auto noDeviceMatches = const_buildings();
    Device::ptr devicePickInteractor;
    auto fillLatLon = [&](ennovatis::Building &b) {
        b.setX(devicePickInteractor->getInfo()->lon);
        b.setY(devicePickInteractor->getInfo()->lat);
    };

    auto updateBuildingInfo = [&](ennovatis::Building &b, Device::ptr dev) {
        if (m_devBuildMap.find(dev) != m_devBuildMap.end())
            return;
        m_devBuildMap[dev] = &b;
        // name in ennovatis is the street => first set it for street => then set the name
        b.setStreet(b.getName());
        b.setName(dev->getInfo()->name);
        b.setHeight(dev->getInfo()->height);
        fillLatLon(b);
    };

    for (auto &building: *m_buildings) {
        lastDst = 100;
        devicePickInteractor = nullptr;
        const auto &ennovatis_strt = building.getName();
        for (const auto &[_, devices]: deviceList) {
            const auto &d = devices.front()->getDevice();
            const auto &device_strt = d->getInfo()->strasse;
            auto lvnstnDst = computeLevensteinDistance(ennovatis_strt, device_strt);

            // if the distance is 0, we have a perfect match
            if (!lvnstnDst) {
                lastDst = 0;
                devicePickInteractor = d;
                break;
            }

            // if the distance is less than the last distance, we have a better match
            if (lvnstnDst < lastDst) {
                lastDst = lvnstnDst;
                devicePickInteractor = d;
            }
            // if the distance is the same as the last distance, we have a better match if the street number is the same
            else if (lvnstnDst == lastDst && cmpStrtNo(ennovatis_strt, device_strt)) {
                devicePickInteractor = d;
            }
        }
        if (!lastDst && devicePickInteractor) {
            updateBuildingInfo(building, devicePickInteractor);
            continue;
        }
        if (devicePickInteractor) {
            auto hit = m_devBuildMap.find(devicePickInteractor);
            if (hit == m_devBuildMap.end()) {
                updateBuildingInfo(building, devicePickInteractor);
            } else {
                noDeviceMatches.push_back(&building);
            }
        }
    }
    return std::make_unique<const_buildings>(noDeviceMatches);
}

bool EnergyPlugin::init()
{
    auto dbPath = configString("CSV", "filename", "default")->value();
    auto channelIdJSONPath = configString("Ennovatis", "jsonPath", "default")->value();
    // csv contains only updated buildings
    auto channelIdCSVPath = configString("Ennovatis", "csvPath", "default")->value();
    ProjTrans pjTrans;
    pjTrans.projFrom = configString("General", "projFrom", "default")->value();
    pjTrans.projTo = configString("General", "projTo", "default")->value();

    initRESTRequest();

    if constexpr (debug) {
        std::cout << "Load database: " << dbPath << std::endl;
        std::cout << "Load channelIDs: " << channelIdJSONPath << std::endl;
    }

    if (loadDB(dbPath, pjTrans))
        std::cout << "Database loaded in cache" << std::endl;
    else
        std::cout << "Database not loaded" << std::endl;

    if (loadChannelIDs(channelIdJSONPath, channelIdCSVPath))
        std::cout << "Ennovatis channelIDs loaded in cache" << std::endl;
    else
        std::cout << "Ennovatis channelIDs not loaded" << std::endl;


    auto noMatches = updateEnnovatisBuildings(m_SDlist);

    if constexpr (debug) {
        int i = 0;
        std::cout << "Matches between devices and buildings:" << std::endl;
        for (auto &[device, building]: m_devBuildMap)
            std::cout << ++i << ": Device: " << device->getInfo()->strasse << " -> Building: " << building->getName()
                      << std::endl;

        std::cout << "No matches for the following buildings:" << std::endl;
        for (auto &building: *noMatches)
            std::cout << building->getName() << std::endl;
    }
    initEnnovatisDevices();
    switchTo(m_sequenceList);
    return true;
}

void EnergyPlugin::helper_initTimestepGrp(size_t maxTimesteps, osg::ref_ptr<osg::Group> &timestepGroup)
{
    for (int t = 0; t < maxTimesteps; ++t) {
        timestepGroup = new osg::Group();
        std::string groupName = "timestep" + std::to_string(t);
        timestepGroup->setName(groupName);
        m_sequenceList->addChild(timestepGroup);
        m_sequenceList->setValue(t);
    }
}

void EnergyPlugin::helper_initTimestepsAndMinYear(size_t &maxTimesteps, int &minYear,
                                                  const std::vector<std::string> &header)
{
    for (const auto &h: header) {
        if (h.find("Strom") != std::string::npos) {
            auto minYearStr = std::regex_replace(h, std::regex("[^0-9]*"), std::string("$1"));
            int min_year_tmp = std::stoi(minYearStr);
            if (min_year_tmp < minYear)
                minYear = min_year_tmp;
            ++maxTimesteps;
        }
    }
}

void EnergyPlugin::helper_projTransformation(bool mapdrape, PJ *P, PJ_COORD &coord, DeviceInfo::ptr deviceInfoPtr,
                                             const double &lat, const double &lon)
{
    if (!mapdrape) {
        deviceInfoPtr->lon = lon;
        deviceInfoPtr->lat = lat;
        deviceInfoPtr->height = 0.f;
        return;
    }

    // x = lon, y = lat
    coord.lpzt.lam = lon;
    coord.lpzt.phi = lat;
    float alt = 0.;

    coord = proj_trans(P, PJ_FWD, coord);

    deviceInfoPtr->lon = coord.xy.x + m_offset[0];
    deviceInfoPtr->lat = coord.xy.y + m_offset[1];
    deviceInfoPtr->height = alt + m_offset[2];
}

void EnergyPlugin::helper_handleEnergyInfo(size_t maxTimesteps, int minYear, const CSVStream::CSVRow &row,
                                           DeviceInfo::ptr deviceInfoPtr)
{
    for (size_t year = minYear; year < minYear + maxTimesteps; ++year) {
        auto str_yr = std::to_string(year);
        auto strom = "Strom " + str_yr;
        auto waerme = "Wärme " + str_yr;
        auto kaelte = "Kälte " + str_yr;
        auto deviceInfoTimestep = std::make_shared<energy::DeviceInfo>(*deviceInfoPtr);
        float strom_val = 0.f;
        access_CSVRow(row, strom, strom_val);
        access_CSVRow(row, waerme, deviceInfoTimestep->waerme);
        access_CSVRow(row, kaelte, deviceInfoTimestep->kaelte);
        deviceInfoTimestep->strom = strom_val / 1000.; // kW -> MW
        auto timestep = year - 2000;
        deviceInfoTimestep->timestep = timestep;
        auto device =
            std::make_shared<energy::Device>(deviceInfoTimestep, m_sequenceList->getChild(timestep)->asGroup());
        auto deviceSensor = std::make_shared<energy::DeviceSensor>(device, device->getGroup());
        m_SDlist[deviceInfoPtr->ID].push_back(deviceSensor);
    }
}

bool EnergyPlugin::loadDBFile(const std::string &fileName, const ProjTrans &projTrans)
{
    try {
        auto csvStream = CSVStream(fileName);
        size_t maxTimesteps = 0;
        int minYear = 2000;
        const auto &header = csvStream.getHeader();
        helper_initTimestepsAndMinYear(maxTimesteps, minYear, header);

        CSVStream::CSVRow row;

        std::string sensorType;
        osg::ref_ptr<osg::Group> timestepGroup;
        bool mapdrape = true;

        auto P = proj_create_crs_to_crs(PJ_DEFAULT_CTX, projTrans.projFrom.c_str(), projTrans.projTo.c_str(), NULL);
        PJ_COORD coord;
        coord.lpzt.z = 0.0;
        coord.lpzt.t = HUGE_VAL;

        if (!P) {
            fprintf(stderr, "Energy Plugin: Ignore mapping. No valid projection was found between given proj string in "
                            "config EnergyCampus.toml\n");
            mapdrape = false;
        }

        helper_initTimestepGrp(maxTimesteps, timestepGroup);

        while (csvStream >> row) {
            auto deviceInfoPtr = std::make_shared<DeviceInfo>();
            auto lat = std::stod(row["lat"]);
            auto lon = std::stod(row["lon"]);

            // location
            helper_projTransformation(mapdrape, P, coord, deviceInfoPtr, lat, lon);

            deviceInfoPtr->ID = row["GebäudeID"];
            deviceInfoPtr->strasse = row["Straße"] + " " + row["Nr"];
            deviceInfoPtr->name = row["Details"];
            access_CSVRow(row, "Baujahr", deviceInfoPtr->baujahr);
            access_CSVRow(row, "Grundfläche (GIS)", deviceInfoPtr->flaeche);

            // electricity, heat, cold
            helper_handleEnergyInfo(maxTimesteps, minYear, row, deviceInfoPtr);
        }
        proj_destroy(P);
    } catch (const CSVStream_Exception &ex) {
        std::cout << ex.what() << std::endl;
        return false;
    }
    return true;
}

bool EnergyPlugin::update()
{
    for (auto s = m_SDlist.begin(); s != m_SDlist.end(); s++) {
        if (s->second.empty())
            continue;
        for (auto timeElem: s->second) {
            if (timeElem)
                timeElem->update();
        }
    }

    if constexpr (debug) {
        auto result = m_debug_worker.getResult();
        if (result)
            for (auto &requ: *result)
                std::cout << "Response:\n" << requ << "\n";
    }

    for (auto &sensor: m_ennovatisDevicesSensors)
        sensor->update();

    for (auto &[name, sensor]: m_cityGMLObjs)
        sensor->update();

    return false;
}

void EnergyPlugin::setTimestep(int t)
{
    m_sequenceList->setValue(t);
    for (auto &sensor: m_ennovatisDevicesSensors)
        sensor->setTimestep(t);

    for (auto &[_,sensor]: m_cityGMLObjs)
        sensor->updateTime(t);
}

COVERPLUGIN(EnergyPlugin)
