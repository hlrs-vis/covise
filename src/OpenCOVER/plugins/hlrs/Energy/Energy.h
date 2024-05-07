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

#include <memory>
#include <osg/Group>
#include <osg/Node>
#include <osg/Sequence>
#include <osg/ref_ptr>
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
#include "EnnovatisDeviceSensor.h"
#include "ennovatis/rest.h"
#include "ennovatis/building.h"
#include "ennovatis/sax.h"
#include "core/PrototypeBuilding.h"

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
    opencover::ui::SelectionList *m_ennovatisSelectionsList = nullptr;
    opencover::ui::Group *m_ennovatisGroup = nullptr;
    opencover::ui::EditField *m_ennovatisFrom = nullptr;
    opencover::ui::EditField *m_ennovatisTo = nullptr;
    opencover::ui::Button *m_ennovatisUpdate = nullptr;
    std::shared_ptr<opencover::ui::SelectionList> m_ennovatisChannelList = nullptr;
    std::shared_ptr<opencover::ui::SelectionList> m_enabledEnnovatisDevices = nullptr;

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
    typedef std::map<Device *, building_const_ptr> DeviceBuildingMap;

    bool loadDBFile(const std::string &fileName);
    bool loadDB(const std::string &path);
    void initRESTRequest();
    void initEnnovatisUI();
    void selectEnabledDevice();
    void setEnnovatisChannelGrp(ennovatis::ChannelGroup group);
    void setRESTDate(const std::string &toSet, bool isFrom);
    void updateEnnovatis();
    void reinitDevices(int comp);
    void updateEnnovatisChannelGrp();
    core::CylinderAttributes getCylinderAttributes();
    void initEnnovatisDevices();
    void switchTo(const osg::Node *child);
    
    /**
     * Loads Ennovatis channelids from the specified JSON file into cache.
     *
     * @param pathToJSON The path to the JSON file which contains the channelids for REST-calls.
     * @return True if the data was successfully loaded, false otherwise.
     */
    bool loadChannelIDs(const std::string &pathToJSON);

    /**
     * Initializes the Ennovatis buildings.
     *
     * This function takes a `DeviceList` object as a parameter and returns a `std::unique_ptr` to a `const_buildings` object.
     * The `const_buildings` object represents the initialized Ennovatis buildings.
     *
     * TODO: apply this while parsing the JSON file
     * @param deviceList The list of devices. Make sure map is sorted.
     * @return A unique pointer to buildings which have ne matching device.
     */
    std::unique_ptr<const_buildings> updateEnnovatisBuildings(const DeviceList &deviceList);

    int maxTimesteps = 10;
    static EnergyPlugin *plugin;
    float rad, scaleH;
    std::vector<double> offset;

    ennovatis::BuildingsPtr m_buildings;
    std::shared_ptr<ennovatis::rest_request> m_req;
    //current selected channel group
    std::shared_ptr<ennovatis::ChannelGroup> m_channelGrp;
    // not necessary but better for debugging
    DeviceBuildingMap m_devBuildMap;
    std::vector<std::unique_ptr<EnnovatisDeviceSensor>> m_ennovatisDevicesSensors;
    osg::ref_ptr<osg::Group> m_ennovatis;
    // switch used to toggle between ennovatis and db data
    osg::ref_ptr<osg::Switch> m_switch;
    osg::Vec4 m_defaultColor;
};

#endif
