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
 ** Author: Leyla Kern                                                     **
 **                                                                        **
 ** History:                                                               **
 **  2024  v1                                                              **
 **                                                                        **
\****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <util/common.h>

#include <PluginUtil/coSensor.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

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

enum Components
{
    Strom,
    Waerme,
    Kaelte
};

class EnergyPlugin : public opencover::coVRPlugin,
                     public ui::Owner,
                     public opencover::coTUIListener
{
public:
    EnergyPlugin();
    ~EnergyPlugin();
    bool init();
    bool destroy();
    bool update();
    bool loadFile(std::string fileName);

    void setTimestep(int t);
    static EnergyPlugin *instance() { return plugin; };

    coTUITab *coEnergyTab = nullptr;

    ui::Menu *EnergyTab = nullptr;
    ui::Button *ShowGraph = nullptr;
    ui::ButtonGroup *componentGroup = nullptr;
    ui::Group *componentList = nullptr;
    ui::Button *StromBt = nullptr;
    ui::Button *WaermeBt = nullptr;
    ui::Button *KaelteBt = nullptr;
    void setComponent(Components c);
    int selectedComp = 0;

    osg::ref_ptr<osg::Group> EnergyGroup;
    osg::Group *parent = nullptr;

    std::map<std::string, std::vector<Device *>> SDlist;
    osg::Sequence *sequenceList;

private:
    int maxTimesteps = 10;
    static EnergyPlugin *plugin;
    float rad, scaleH;
};

#endif
