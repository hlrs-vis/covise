/****************************************************************************\
 **                                                          (C)2024 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading building energy data        **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern, Marko Djuric                                       **
 **                                                                        **
 ** History:                                                               **
 **  2024  v1                                                              **
 **  Marko Djuric 02.2024: add ennovatis client                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/
//TODO: fetch lat lon from googlemaps

#include "Energy.h"

#include "ennovatis/building.h"
#include "ennovatis/sax.h"
#include "ennovatis/REST.h"
#include "build_options.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

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
using json = nlohmann::json;

namespace {

constexpr bool debug = build_options.debug_ennovatis;
    
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
static bool helper_cmpChar(char a, char b)
{
    return (a == b);
}

// Case-insensitive char comparer
static bool helper_cmpCharIgnoreCase(char a, char b)
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
static int computeLevensteinDistance(const std::string &s1, const std::string &s2, bool ignoreCase = false)
{
    const int &len1 = s1.size(), len2 = s2.size();

    // allocate distance matrix
    std::vector<std::vector<int>> d(len1 + 1, std::vector<int>(len2 + 1));

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
}

EnergyPlugin *EnergyPlugin::plugin = NULL;

std::string proj_to = "+proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs ";
std::string proj_from = "+proj=latlong";
float offset[] = {-507080, -5398430, 450};

EnergyPlugin::EnergyPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME), ui::Owner("EnergyPlugin", cover->ui)
{
    fprintf(stderr, "Starting Energy Plugin\n");
    plugin = this;

    m_buildings = std::make_unique<ennovatis::Buildings>();

    EnergyGroup = new osg::Group();
    EnergyGroup->setName("Energy");
    cover->getObjectsRoot()->addChild(EnergyGroup);

    sequenceList = new osg::Sequence();
    sequenceList->setName("Timesteps");
    EnergyGroup->addChild(sequenceList);

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
    componentGroup->setCallback(
        [this](int value)
        { setComponent(Components(value)); });

    // ennovatis
    ennovatis_BtnGroup = new ui::ButtonGroup(EnergyTab, "EnnovatisBtnGroup");
    ennovatis_BtnGroup->setDefaultValue(ennovatis::ChannelGroup::Strom);
    ennovatis_Group = new ui::Group(EnergyTab, "Ennovatis");
    ennovatis_Group->setText("Ennovatis");
    ennovatis_StromBt = new ui::Button(ennovatis_Group, "Ennovatis_Strom", ennovatis_BtnGroup, ennovatis::ChannelGroup::Strom);
    ennovatis_WaermeBt = new ui::Button(ennovatis_Group, "Ennovatis_Waerme", ennovatis_BtnGroup, ennovatis::ChannelGroup::Waerme);
    ennovatis_KaelteBt = new ui::Button(ennovatis_Group, "Ennovatis_Kaelte", ennovatis_BtnGroup, ennovatis::ChannelGroup::Kaelte);
    ennovatis_WasserBt = new ui::Button(ennovatis_Group, "Ennovatis_Wasser", ennovatis_BtnGroup, ennovatis::ChannelGroup::Wasser);
    // TODO: add calender widget
}

void EnergyPlugin::setComponent(Components c)
{
    switch (c)
    {
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
    int scount = 0;
    for (auto s : SDlist)
    {
        if (s.second.empty())
            continue;
        for (auto t : s.second)
        {
            t->init(rad, scaleH, selectedComp);
        }
    }
}

EnergyPlugin::~EnergyPlugin() {}

bool EnergyPlugin::loadDB(const std::string &path)
{
    if (!loadDBFile(path))
    {
        return false;
    }

    if ((int)sequenceList->getNumChildren() >
        coVRAnimationManager::instance()->getNumTimesteps())
    {
        coVRAnimationManager::instance()->setNumTimesteps(
            sequenceList->getNumChildren(), sequenceList);
    }

    rad = 3.;
    scaleH = 0.1;

    for (auto s : SDlist)
    {
        if (s.second.empty())
            continue;
        for (auto t : s.second)
            t->init(rad, scaleH, selectedComp);
    }
    return true;
}

bool EnergyPlugin::loadChannelIDs(const std::string &pathToJSON) {
    std::ifstream inputFilestream(pathToJSON);
    ennovatis::sax_channelid_parser slp(m_buildings);
    if (!json::sax_parse(inputFilestream, &slp))
        return false;

    // TODO: use another container than a vector to access building data faster
    
    if constexpr (build_options.debug_ennovatis)
        for (auto &log : slp.getDebugLogs())
            std::cout << log << std::endl;
    
    return true;
}

void EnergyPlugin::initRESTRequest() {
    m_req.url = configString("Ennovatis", "restUrl", "default")->value();
    m_req.projEid = configString("Ennovatis", "projEid", "default")->value();
    m_req.channelId = "";
    m_req.dtf = std::chrono::system_clock::now() - std::chrono::hours(24);
    m_req.dtt = std::chrono::system_clock::now();
}

std::unique_ptr<EnergyPlugin::const_buildings> EnergyPlugin::createQuartersMap(buildings_const_Ptr buildings,
                                                                               const DeviceList &deviceList)
{
    auto lastDst = 0;
    auto noDeviceMatches = const_buildings();
    Device *devPick;
    for (const auto &building: *buildings) {
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
                continue;
            }

            // if the distance is the same as the last distance, we have a better match if the street number is the same
            if (lvnstnDst == lastDst)
                if (cmpStrtNo(ennovatis_strt, device_strt))
                    devPick = d;
        }
        if (!lastDst) {
            m_quarters[devPick] = &building;
            continue;
        }
        if (devPick) {
            auto hit = m_quarters.find(devPick);
            if (hit == m_quarters.end())
                m_quarters[devPick] = &building;
            else
                noDeviceMatches.push_back(&building);
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

    auto noMatches = createQuartersMap(m_buildings.get(), SDlist);

    if constexpr (debug) {
        int i = 0;
        std::cout << "Matches between devices and buildings:" << std::endl;
        for (auto &[device, building]: m_quarters)
            std::cout << ++i << ": Device: " << device->devInfo->strasse << " -> Building: " << building->getName()
                      << std::endl;

        std::cout << "No matches for the following buildings:" << std::endl;
        for (auto &building: *noMatches)
            std::cout << building->getName() << std::endl;
    }

    // TODO: put this in callback for rest calls (e.g. when the user clicks on a building in the UI)
    // test rest to ennovatis
    // std::string url = m_req();
    // std::string response;
    // ennovatis::performCurlRequest(url, response);
    // std::cout << response << std::endl;

    return true;
}

bool EnergyPlugin::loadDBFile(const std::string &fileName)
{
    FILE *fp = fopen(fileName.c_str(), "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Energy Plugin: could not open file\n");
        return false;
    }

    const int lineSize = 1000;
    char buf[lineSize];

    bool mapdrape = true;
    projPJ pj_from = pj_init_plus(proj_from.c_str());
    projPJ pj_to = pj_init_plus(proj_to.c_str());
    if (!pj_from || !pj_to)
    {
        fprintf(stderr, "Energy Plugin: Ignoring mapping. No valid projection was found \n");
        mapdrape = false;
    }

    std::string sensorType;
    osg::Group *timestepGroup;

    if (!fgets(buf, lineSize, fp))
    {
        fclose(fp);
        return false;
    }

    for (int t = 0; t < maxTimesteps; ++t)
    {
        timestepGroup = new osg::Group();
        std::string groupName = "timestep" + std::to_string(t);
        timestepGroup->setName(groupName);
        sequenceList->addChild(timestepGroup);
        sequenceList->setValue(t);
    }

    bool firstLine = true;

    while (!feof(fp))
    {
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
        if (mapdrape)
        {
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
        }
        else
        {
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

        for (int j = 0; j < maxTimesteps; ++j)
        {
            ++tok;
            stromList.push_back(std::strtof(tok->c_str(), NULL) / 1000.); // kW -> MW
        }

        std::advance(tok, 7);
        for (int j = 0; j < maxTimesteps; ++j)
        {
            ++tok;
            kaelteList.push_back(std::strtof(tok->c_str(), NULL));
        }
        std::advance(tok, 7);
        for (int j = 0; j < (maxTimesteps - 2); ++j) // FIXME: Problem with last two timesteps (tokens)
        {
            ++tok;
            waermeList.push_back(std::strtof(tok->c_str(), NULL));
        }

        for (int j = 0; j < maxTimesteps; ++j)
        {
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
    for (auto s = SDlist.begin(); s != SDlist.end(); s++)
    {
        if (s->second.empty())
            continue;
        for (auto timeElem : s->second)
        {
            timeElem->update();
        }
    }
    return false;
}

void EnergyPlugin::setTimestep(int t)
{
    sequenceList->setValue(t);
}

bool EnergyPlugin::destroy()
{
    cover->getObjectsRoot()->removeChild(EnergyGroup);
    return false;
}

COVERPLUGIN(EnergyPlugin)
