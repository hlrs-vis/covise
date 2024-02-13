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
\****************************************************************************/

#ifndef _Energy_PLUGIN_H
#define _Energy_PLUGIN_H

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
#include <cover/ui/EditField.h>
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
    opencover::ui::ButtonGroup *ennovatisBtnGroup = nullptr;
    opencover::ui::Group *ennovatisGroup = nullptr;
    opencover::ui::EditField *ennovatisFrom = nullptr;
    opencover::ui::EditField *ennovatisTo = nullptr;
    std::array<opencover::ui::Button *, ennovatis::ChannelGroup::None> ennovatisBtns;

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
    void setEnnovatisChannelGrp(ennovatis::ChannelGroup group);
    void setRESTDate(const std::string &toSet, bool isFrom);
    
    /**
     * Loads Ennovatis channelids from the specified JSON file into cache.
     *
     * @param pathToJSON The path to the JSON file which contains the channelids for REST-calls.
     * @return True if the data was successfully loaded, false otherwise.
     */
    bool loadChannelIDs(const std::string &pathToJSON);

    /**
     * @brief Creates a quarters map for the EnergyPlugin.
     * 
     * This function creates a link map between buildings and devices.
     * 
     * TODO: apply this while parsing the JSON file
     * @param buildings A pointer to the buildings object. Make sure vector is sorted.
     * @param deviceList The list of devices. Make sure map is sorted.
     * @return A unique pointer to buildings which have ne matching device.
     */
    std::unique_ptr<const_buildings> createQuartersMap(buildings_const_Ptr buildings, const DeviceList &deviceList);

    int maxTimesteps = 10;
    static EnergyPlugin *plugin;
    float rad, scaleH;
    ennovatis::BuildingsPtr m_buildings;
    ennovatis::RESTRequest m_req;
    Quarters m_quarters;
};

#endif
