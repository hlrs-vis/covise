/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Energy_PLUGIN_H
#define _Energy_PLUGIN_H
/****************************************************************************\
 **                                                          (C)2024 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading building energy data       **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern, Marko Djuric                                       **
 **                                                                        **
 ** History:                                                               **
 **  2024  v1                                                              **
 **  Marko Djuric 01.2024:                                                 **
 **                                                                        **
\****************************************************************************/

// #include <memory>
#include <memory>
#include <util/common.h>

#include <PluginUtil/coSensor.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>

#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>

#include <cover/coVRTui.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <util/coTypes.h>

#include <gdal_priv.h>

#include "Device.h"
#include "ennovatis/REST.h"
#include "ennovatis/building.h"
#include "ennovatis/sax.h"

enum Components
{
    Strom,
    Waerme,
    Kaelte
};

class EnergyPlugin : public opencover::coVRPlugin,
                     public opencover::ui::Owner,
                     public opencover::coTUIListener
{
public:
    EnergyPlugin();
    ~EnergyPlugin();
    bool init();
    bool destroy();
    bool update();

    void setTimestep(int t);
    static EnergyPlugin *instance() { return plugin; };

    opencover::coTUITab *coEnergyTab = nullptr;
    opencover::ui::Menu *EnergyTab = nullptr;
    opencover::ui::Button *ShowGraph = nullptr;
    opencover::ui::ButtonGroup *componentGroup = nullptr;
    opencover::ui::Group *componentList = nullptr;
    opencover::ui::Button *StromBt = nullptr;
    opencover::ui::Button *WaermeBt = nullptr;
    opencover::ui::Button *KaelteBt = nullptr;
    
    // ennovatis UI
    opencover::ui::ButtonGroup *ennovatis_BtnGroup = nullptr;
    opencover::ui::Group *ennovatis_Group = nullptr;
    opencover::ui::Button *ennovatis_StromBt = nullptr;
    opencover::ui::Button *ennovatis_WaermeBt = nullptr;
    opencover::ui::Button *ennovatis_KaelteBt = nullptr;
    opencover::ui::Button *ennovatis_WasserBt = nullptr;

    void setComponent(Components c);
    int selectedComp = 0;

    osg::ref_ptr<osg::Group> EnergyGroup;
    osg::Group *parent = nullptr;

    typedef std::map<std::string, std::vector<Device *>> DeviceList;
    DeviceList SDlist;
    osg::Sequence *sequenceList;

private:
    typedef const ennovatis::Building* building_const_ptr;
    typedef const ennovatis::Buildings* buildings_const_Ptr;
    typedef std::vector<building_const_ptr> const_buildings;
    typedef std::map<const Device *, building_const_ptr> Quarters;

    bool loadDBFile(const std::string &fileName);
    bool loadDB(const std::string &path);
    void initRESTRequest();
    
    /**
     * Loads Ennovatis channelids from the specified JSON file into cache.
     *
     * @param pathToJSON The path to the JSON file which contains the channelids for REST-calls.
     * @return True if the data was successfully loaded, false otherwise.
     */
    bool loadChannelIDs(const std::string &pathToJSON);
    // void createQuartersMap(BuildingsPtr buildings, const DeviceList &deviceList);
    std::unique_ptr<const_buildings> createQuartersMap(buildings_const_Ptr buildings, const DeviceList &deviceList);

    int maxTimesteps = 10;
    static EnergyPlugin *plugin;
    float rad, scaleH;
    ennovatis::BuildingsPtr m_buildings;
    ennovatis::RESTRequest m_req;
    Quarters m_quarters;
};

#endif
