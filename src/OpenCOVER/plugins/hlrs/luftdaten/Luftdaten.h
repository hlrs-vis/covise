/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Luft_PLUGIN_H
#define _Luft_PLUGIN_H
/****************************************************************************\
 **                                                          (C)2020 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading Luftdaten sensor data       **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern                                                     **
 **                                                                        **
 ** History:                                                               **
 ** April 2020  v1                                                         **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <util/common.h>
#include <string.h>
#include <stdio.h>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/coSensor.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>

#include <util/coTypes.h>
#include <osg/Material>
#include <osg/Vec3>
#include <osg/ShapeDrawable>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Action.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <cover/coVRTui.h>

#include <gdal_priv.h>

#include "Device.h"

enum Components
{
    PM10,
    PM2,
    Temp,
    Humi
};

class LuftdatenPlugin: public opencover::coVRPlugin, public ui::Owner, public opencover::coTUIListener
{
public:
    LuftdatenPlugin();
    ~LuftdatenPlugin();
    bool init();
    bool destroy();
    bool update();
    bool loadFile(std::string fileName);
    
    int mergeTimesteps();
    void setTimestep(int t);
    static LuftdatenPlugin *instance() {return plugin; };
    
    float getAlt(double x, double y);
    void openImage(std::string &name);
    void closeImage();
    
    coTUITab *coLuftTab = nullptr;
    coTUIWebview *WebView = nullptr;
    
    ui::Menu *LuftTab = nullptr;
    ui::Button *ShowGraph = nullptr;
    ui::ButtonGroup *componentGroup = nullptr;
    ui::Group *componentList = nullptr;
    ui::Button *pm10Bt = nullptr;
    ui::Button *pm2Bt = nullptr;
    ui::Button *tempBt = nullptr;
    ui::Button *humiBt = nullptr;
    void setComponent(Components c);
    int selectedComp = 0;
    
    osg::ref_ptr<osg::Group> LuftdatenGroup;
    osg::Group *parent = nullptr;
    
    std::map<std::string, std::vector<Device *>> SDlist;
    osg::Sequence *sequenceList;
    
    bool loaded = false;
    bool graphVisible =false;
private:
    static LuftdatenPlugin *plugin;
    float rad, scaleH;
    
    bool mapAlt = true;
    int TIME_INTERVAL = 5*60; //set time interval to 5min
    time_t initialTime;

    float *rasterData=NULL;
    double xOrigin;
    double yOrigin;
    double pixelWidth;
    double pixelHeight;
    int cols;
    int rows;
    GDALDataset  *heightDataset;
    GDALRasterBand  *heightBand;
};

#endif

