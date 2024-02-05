/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

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
 **  Marko Djuric 01.2024:                                                 **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "Energy.h"
#include "ennovatis/sax.h"
#include "ennovatis/REST.h"
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

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

EnergyPlugin *EnergyPlugin::plugin = NULL;

std::string proj_to = "+proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs ";
std::string proj_from = "+proj=latlong";
float offset[] = {-507080, -5398430, 450};

EnergyPlugin::EnergyPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME), ui::Owner("EnergyPlugin", cover->ui)
{
    fprintf(stderr, "Starting Energy Plugin\n");
    plugin = this;

    m_buildings = std::make_shared<ennovatis::Buildings>();

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

    componentGroup = new ui::ButtonGroup(EnergyTab, "ComponentGroup");
    componentGroup->setDefaultValue(Strom);
    componentList = new ui::Group(EnergyTab, "Component");
    componentList->setText("Messwerte (jährlich)");
    StromBt = new ui::Button(componentList, "Strom", componentGroup, Strom);
    WaermeBt = new ui::Button(componentList, "Wärme", componentGroup, Waerme);
    KaelteBt = new ui::Button(componentList, "Kälte", componentGroup, Kaelte);
    componentGroup->setCallback(
        [this](int value)
        { setComponent(Components(value)); });
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
    json::sax_parse(inputFilestream, &slp);
    
    // if constexpr (build_options.debug_ennovatis)
    for (auto &log : slp.getDebugLogs())
        std::cout << log << std::endl;
    
    // test rest to ennovatis 
    // std::string url = "http://129.69.40.93:4029/ControllingService/DataManagement/ReadData?projEid=00-2914207990135212101590650999999998_1&dtf=01.01.2022&dtt=01.02.2022&ts=86400&tsp=0&tst=1&etst=1024&u=&cEid=001FC6E826360571614745481820999977314_5";
    // std::string response;
    // ennovatis::performCurlRequest(url, response);
    // std::cout << response << std::endl;
    
    return true;
}

bool EnergyPlugin::init()
{
    auto dbPath= configString("CSV", "filename", "default")->value();
    auto channelIdJSONPath = configString("Ennovatis", "jsonPath", "default")->value();

    std::cerr << "load database: " << dbPath << std::endl;
    std::cerr << "load channelIDs: " << channelIdJSONPath << std::endl;
    
    if(loadDB(dbPath))
    {
        std::cerr << "database loaded in cache" << std::endl;
    }
    else
    {
        std::cerr << "database not loaded" << std::endl;
    }
    
    if(loadChannelIDs(channelIdJSONPath))
    {
        std::cerr << "Ennovatis channelIDs loaded in cache" << std::endl;
    }
    else
    {
        std::cerr << "Ennovatis channelIDs not loaded" << std::endl;
    }
    
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
