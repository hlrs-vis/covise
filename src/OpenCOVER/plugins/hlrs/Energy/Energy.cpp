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

#include "Energy.h"

#include "EnnovatisDevice.h"
#include "EnnovatisDeviceSensor.h"
#include "build_options.h"
#include "cover/ui/SelectionList.h"
#include <core/TxtInfoboard.h>
#include <core/PrototypeBuilding.h>
#include <ennovatis/building.h>
#include <ennovatis/date.h>
#include <ennovatis/sax.h>
#include <ennovatis/rest.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <osg/Group>
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
#include <proj_api.h>

using namespace opencover;

namespace {

constexpr bool debug = build_options.debug_ennovatis;
constexpr auto proj_to = "+proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs ";
constexpr auto proj_from = "+proj=latlong";
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

EnergyPlugin *EnergyPlugin::plugin = NULL;

EnergyPlugin::EnergyPlugin(): coVRPlugin(COVER_PLUGIN_NAME), ui::Owner("EnergyPlugin", cover->ui)
{
    fprintf(stderr, "Starting Energy Plugin\n");
    plugin = this;

    m_buildings = std::make_unique<ennovatis::Buildings>();

    EnergyGroup = new osg::Group();
    EnergyGroup->setName("Energy");
    cover->getObjectsRoot()->addChild(EnergyGroup);

    sequenceList = new osg::Sequence();
    sequenceList->setName("DB");
    m_ennovatis = new osg::Group();
    m_ennovatis->setName("Ennovatis");

    m_switch = new osg::Switch();
    m_switch->setName("Switch");
    m_switch->addChild(sequenceList);
    m_switch->addChild(m_ennovatis);
    EnergyGroup->addChild(m_switch);

    GDALAllRegister();

    SDlist.clear();

    EnergyTab = new ui::Menu("Energy Campus", EnergyPlugin::plugin);
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

    offset = configFloatArray("General", "offset", std::vector<double>{0, 0, 0})->value();
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
    m_enabledEnnovatisDevices = std::make_shared<opencover::ui::SelectionList>(EnergyTab, "Enabled Devices: ");
    m_enabledEnnovatisDevices->setCallback([this](int value) { selectEnabledDevice(); });
    m_ennovatisChannelList = std::make_shared<opencover::ui::SelectionList>(EnergyTab, "Channels: ");

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
    for (auto s: SDlist) {
        if (s.second.empty())
            continue;
        for (auto t: s.second)
            t->init(rad, scaleH, comp);
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
    auto maxColor = osg::Vec4(configMaxColorVec[0], configMaxColorVec[1], configMaxColorVec[2],
                              configMaxColorVec[3]);
    auto minColor = osg::Vec4(configMinColorVec[0], configMinColorVec[1], configMinColorVec[2],
                              configMinColorVec[3]);
    return core::CylinderAttributes(configRadiusCycl, configDefaultHeightCycl, maxColor, minColor, defaultColor);
}

void EnergyPlugin::initEnnovatisDevices()
{
    m_ennovatis->removeChildren(0, m_ennovatis->getNumChildren());
    m_ennovatisDevicesSensors.clear();
    auto cylinderAttributes = getCylinderAttributes();
    for (auto &b: *m_buildings) {
        cylinderAttributes.position = osg::Vec3(b.getLat(), b.getLon(), b.getHeight());
        auto drawableBuilding = std::make_unique<core::PrototypeBuilding>(cylinderAttributes);
        auto infoboardPos =
            osg::Vec3(b.getLat() + cylinderAttributes.radius + 5, b.getLon(), b.getHeight() + cylinderAttributes.height);
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

void EnergyPlugin::switchTo(const osg::Node *child)
{
    m_switch->setAllChildrenOff();
    m_switch->setChildValue(child, true);
}

void EnergyPlugin::setComponent(Components c)
{
    switchTo(sequenceList);
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
    selectedComp = c;
    reinitDevices(c);
}

EnergyPlugin::~EnergyPlugin()
{}

bool EnergyPlugin::loadDB(const std::string &path)
{
    if (!loadDBFile(path)) {
        return false;
    }

    if ((int)sequenceList->getNumChildren() > coVRAnimationManager::instance()->getNumTimesteps()) {
        coVRAnimationManager::instance()->setNumTimesteps(sequenceList->getNumChildren(), sequenceList);
    }

    rad = 3.;
    scaleH = 0.1;

    reinitDevices(selectedComp);
    return true;
}

bool EnergyPlugin::loadChannelIDs(const std::string &pathToJSON)
{
    std::ifstream inputFilestream(pathToJSON);
    ennovatis::sax_channelid_parser slp(m_buildings);
    if (!slp.parse_filestream(inputFilestream))
        return false;

    if constexpr (debug)
        for (auto &log: slp.getDebugLogs())
            std::cout << log << std::endl;

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
    Device *devPick;
    auto fillLatLon = [&](ennovatis::Building &b, Device *dev) {
        b.setLat(devPick->devInfo->lat);
        b.setLon(devPick->devInfo->lon);
    };

    auto updateBuildingInfo = [&](ennovatis::Building &b, Device *dev) {
        m_devBuildMap[dev] = &b;
        // name in ennovatis is the street => first set it for street => then set the name
        b.setStreet(b.getName());
        b.setName(dev->devInfo->name);
        b.setHeight(dev->devInfo->height);
        fillLatLon(b, dev);
    };

    for (auto &building: *m_buildings) {
        lastDst = 100;
        devPick = nullptr;
        const auto &ennovatis_strt = building.getName();
        for (const auto &[_, devices]: deviceList) {
            const auto &d = devices.front();
            const auto &device_strt = d->devInfo->strasse;
            auto lvnstnDst = computeLevensteinDistance(ennovatis_strt, device_strt);

            // if the distance is 0, we have a perfect match
            if (!lvnstnDst) {
                lastDst = 0;
                devPick = d;
                break;
            }

            // if the distance is less than the last distance, we have a better match
            if (lvnstnDst < lastDst) {
                lastDst = lvnstnDst;
                devPick = d;
            }
            // if the distance is the same as the last distance, we have a better match if the street number is the same
            else if (lvnstnDst == lastDst && cmpStrtNo(ennovatis_strt, device_strt)) {
                devPick = d;
            }
        }
        if (!lastDst && devPick) {
            updateBuildingInfo(building, devPick);
            continue;
        }
        if (devPick) {
            auto hit = m_devBuildMap.find(devPick);
            if (hit == m_devBuildMap.end()) {
                updateBuildingInfo(building, devPick);
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

    initRESTRequest();

    if constexpr (debug) {
        std::cout << "Load database: " << dbPath << std::endl;
        std::cout << "Load channelIDs: " << channelIdJSONPath << std::endl;
    }

    if (loadDB(dbPath))
        std::cout << "Database loaded in cache" << std::endl;
    else
        std::cout << "Database not loaded" << std::endl;

    if (loadChannelIDs(channelIdJSONPath))
        std::cout << "Ennovatis channelIDs loaded in cache" << std::endl;
    else
        std::cout << "Ennovatis channelIDs not loaded" << std::endl;

    auto noMatches = updateEnnovatisBuildings(SDlist);

    if constexpr (debug) {
        int i = 0;
        std::cout << "Matches between devices and buildings:" << std::endl;
        for (auto &[device, building]: m_devBuildMap)
            std::cout << ++i << ": Device: " << device->devInfo->strasse << " -> Building: " << building->getName()
                      << std::endl;

        std::cout << "No matches for the following buildings:" << std::endl;
        for (auto &building: *noMatches)
            std::cout << building->getName() << std::endl;
    }
    initEnnovatisDevices();
    switchTo(sequenceList);
    return true;
}

bool EnergyPlugin::loadDBFile(const std::string &fileName)
{
    FILE *fp = fopen(fileName.c_str(), "r");
    if (fp == NULL) {
        fprintf(stderr, "Energy Plugin: could not open file\n");
        return false;
    }

    const int lineSize = 1000;
    char buf[lineSize];

    bool mapdrape = true;
    projPJ pj_from = pj_init_plus(proj_from);
    projPJ pj_to = pj_init_plus(proj_to);
    if (!pj_from || !pj_to) {
        fprintf(stderr, "Energy Plugin: Ignoring mapping. No valid projection was found \n");
        mapdrape = false;
    }

    std::string sensorType;
    osg::Group *timestepGroup;

    if (!fgets(buf, lineSize, fp)) {
        fclose(fp);
        return false;
    }

    for (int t = 0; t < maxTimesteps; ++t) {
        timestepGroup = new osg::Group();
        std::string groupName = "timestep" + std::to_string(t);
        timestepGroup->setName(groupName);
        sequenceList->addChild(timestepGroup);
        sequenceList->setValue(t);
    }

    bool firstLine = true;

    while (!feof(fp)) {
        if (!fgets(buf, lineSize, fp))
            break;
        std::string line(buf);
        boost::char_separator<char> sep(",");
        boost::tokenizer<boost::char_separator<char>> tokens(line, sep);
        auto tok = tokens.begin();

        DeviceInfo *di = new DeviceInfo();
        // auto di = std::make_shared<DeviceInfo>();

        di->ID = tok->c_str();
        tok++;

        // location
        if (mapdrape) {
            double xlat = std::strtod(tok->c_str(), NULL);
            ++tok;
            double xlon = std::strtod(tok->c_str(), NULL);
            float alt = 0.;
            xlat *= DEG_TO_RAD;
            xlon *= DEG_TO_RAD;

            int error = pj_transform(pj_from, pj_to, 1, 1, &xlon, &xlat, NULL);

            di->lat = xlon + offset[0];
            di->lon = xlat + offset[1];
            di->height = alt + offset[2];
        } else {
            di->lat = std::strtof(tok->c_str(), NULL);
            ++tok;
            di->lon = std::strtof(tok->c_str(), NULL);
            di->height = 0.f;
        }

        // street
        std::advance(tok, 3);
        std::string street(tok->c_str());
        ++tok;
        street.append(" ");
        street.append(tok->c_str());
        di->strasse = street;

        // details
        ++tok;
        di->name = tok->c_str();
        ++tok;
        di->baujahr = std::strtof(tok->c_str(), NULL);
        ++tok;
        di->flaeche = std::strtof(tok->c_str(), NULL);

        // electricity, heat, cold
        std::advance(tok, 11);
        std::vector<float> stromList, kaelteList, waermeList;

        for (int j = 0; j < maxTimesteps; ++j) {
            ++tok;
            stromList.push_back(std::strtof(tok->c_str(), NULL) / 1000.); // kW -> MW
        }

        std::advance(tok, 7);
        for (int j = 0; j < maxTimesteps; ++j) {
            ++tok;
            kaelteList.push_back(std::strtof(tok->c_str(), NULL));
        }
        std::advance(tok, 7);
        for (int j = 0; j < (maxTimesteps - 2); ++j) // FIXME: Problem with last two timesteps (tokens)
        {
            ++tok;
            waermeList.push_back(std::strtof(tok->c_str(), NULL));
        }

        for (int j = 0; j < maxTimesteps; ++j) {
            DeviceInfo *diT = new DeviceInfo(*di);
            diT->strom = stromList[j];
            diT->kaelte = kaelteList[j];
            diT->waerme = waermeList[j];
            diT->timestep = j;
            Device *sd = new Device(diT, sequenceList->getChild(j)->asGroup());
            SDlist[diT->ID].push_back(sd);
        }
    }
    fclose(fp);
    return true;
}

bool EnergyPlugin::update()
{
    for (auto s = SDlist.begin(); s != SDlist.end(); s++) {
        if (s->second.empty())
            continue;
        for (auto timeElem: s->second) {
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

    return false;
}

void EnergyPlugin::setTimestep(int t)
{
    sequenceList->setValue(t);
    for (auto &sensor: m_ennovatisDevicesSensors)
        sensor->setTimestep(t);
}

bool EnergyPlugin::destroy()
{
    cover->getObjectsRoot()->removeChild(EnergyGroup);
    return false;
}

COVERPLUGIN(EnergyPlugin)
