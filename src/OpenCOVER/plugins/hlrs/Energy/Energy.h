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
#include <map>
#include <osg/MatrixTransform>
#include <string>
#include <osg/Group>
#include <osg/Node>
#include <osg/Sequence>
#include <osg/ref_ptr>
#include <proj.h>
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

#include <Device.h>
#include <DeviceSensor.h>
#include <EnnovatisDeviceSensor.h>
#include <ennovatis/rest.h>
#include <ennovatis/building.h>
#include <core/PrototypeBuilding.h>
#include <utils/read/csv/csv.h>
#include <CityGMLDeviceSensor.h>

class EnergyPlugin: public opencover::coVRPlugin, public opencover::ui::Owner, public opencover::coTUIListener {
    enum Components { Strom, Waerme, Kaelte };
    struct ProjTrans {
        std::string projFrom;
        std::string projTo;
    };

public:
    EnergyPlugin();
    ~EnergyPlugin();
    EnergyPlugin(const EnergyPlugin &) = delete;
    void operator=(const EnergyPlugin &) = delete;

    bool init();
    // bool destroy();
    bool update();
    void setTimestep(int t);
    void setComponent(Components c);
    static EnergyPlugin *instance()
    {
        if (!m_plugin)
            m_plugin = new EnergyPlugin;
        return m_plugin;
    };


private:
    // typedef const ennovatis::Building *building_const_ptr;
    // typedef const ennovatis::Buildings *buildings_const_Ptr;
    typedef const ennovatis::Building *building_const_ptr;
    typedef const ennovatis::Buildings *buildings_const_Ptr;
    typedef std::vector<building_const_ptr> const_buildings;
    typedef std::map<energy::Device::ptr, building_const_ptr> DeviceBuildingMap;
    // typedef std::vector<ennovatis::Building::ptr> const_buildings;
    // typedef std::map<energy::Device::ptr, ennovatis::Building::ptr> DeviceBuildingMap;
    typedef std::map<std::string, std::vector<energy::DeviceSensor::ptr>> DeviceList;

    void helper_initTimestepGrp(size_t maxTimesteps, osg::ref_ptr<osg::Group> &timestepGroup);
    void helper_initTimestepsAndMinYear(size_t &maxTimesteps, int &minYear, const std::vector<std::string> &header);
    void helper_projTransformation(bool mapdrape, PJ *P, PJ_COORD &coord, energy::DeviceInfo::ptr deviceInfoPtr,
                                   const double &lat, const double &lon);
    void helper_handleEnergyInfo(size_t maxTimesteps, int minYear, const opencover::utils::read::CSVStream::CSVRow &row,
                                 energy::DeviceInfo::ptr deviceInfoPtr);
    bool loadDBFile(const std::string &fileName, const ProjTrans &projTrans);
    bool loadDB(const std::string &path, const ProjTrans &projTrans);
    void initRESTRequest();
    void initEnnovatisUI();
    void initCityGMLUI();
    void enableCityGML(bool on);
    void addCityGMLObjects(osg::MatrixTransform *node);
    void selectEnabledDevice();
    void setEnnovatisChannelGrp(ennovatis::ChannelGroup group);
    void setRESTDate(const std::string &toSet, bool isFrom);
    void updateEnnovatis();
    void reinitDevices(int comp);
    void updateEnnovatisChannelGrp();
    void initEnnovatisDevices();
    void switchTo(const osg::ref_ptr<osg::Node> child);

    /**
     * Loads Ennovatis channelids from the specified JSON file into cache.
     *
     * @param pathToJSON The path to the JSON file which contains the channelids for REST-calls.
     * @return True if the data was successfully loaded, false otherwise.
     */
    bool loadChannelIDs(const std::string &pathToJSON, const std::string &pathToCSV);
    bool updateChannelIDsFromCSV(const std::string &pathToCSV);
    core::CylinderAttributes getCylinderAttributes();

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
    // std::unique_ptr<ennovatis::Buildings> updateEnnovatisBuildings(const DeviceList &deviceList);

    static EnergyPlugin *m_plugin;

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
    opencover::ui::SelectionList *m_ennovatisChannelList = nullptr;
    opencover::ui::SelectionList *m_enabledEnnovatisDevices = nullptr;

    // citygml UI
    opencover::ui::Group *m_cityGMLGroup = nullptr;
    opencover::ui::Button *m_cityGMLEnable = nullptr;

    float rad, scaleH;
    int m_selectedComp = 0;
    std::vector<double> m_offset;

    ennovatis::BuildingsPtr m_buildings;
    DeviceList m_SDlist;
    std::shared_ptr<ennovatis::rest_request> m_req;

    // current selected channel group
    std::shared_ptr<ennovatis::ChannelGroup> m_channelGrp;

    // not necessary but better for debugging
    DeviceBuildingMap m_devBuildMap;
    std::vector<std::unique_ptr<EnnovatisDeviceSensor>> m_ennovatisDevicesSensors;
    osg::ref_ptr<osg::Group> m_ennovatis;

    // switch used to toggle between ennovatis, db and citygml data
    osg::ref_ptr<osg::Switch> m_switch;
    osg::ref_ptr<osg::Sequence> m_sequenceList;
    osg::ref_ptr<osg::Group> m_EnergyGroup;
    osg::ref_ptr<osg::Group> m_cityGML;
    std::map<std::string, std::unique_ptr<CityGMLDeviceSensor>> m_cityGMLObjs;
    osg::Vec4 m_defaultColor;
};

#endif
