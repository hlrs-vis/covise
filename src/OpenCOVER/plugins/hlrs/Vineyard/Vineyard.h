/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Vineyard_PLUGIN_H
#define _Vineyard_PLUGIN_H
/****************************************************************************\
 **                                                          (C)2020 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading Vineyard sensor data       **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **
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
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <gdal.h>
#include <ogrsf_frmts.h>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/ProxyNode>
#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>


enum PVori { O_LANDSCAPE = 0, O_PORTRAIT };

class VineyardPlugin: public opencover::coVRPlugin, public ui::Owner, public opencover::coTUIListener
{
public:
    VineyardPlugin();
    ~VineyardPlugin();
    bool init();
    bool destroy();
    bool update();
    bool loadPVShp(const std::string& filename);
    static VineyardPlugin *instance() {return plugin; };
    
    coTUITab *coLuftTab = nullptr;
    
    ui::Menu *VineTab = nullptr;
    ui::Button *ShowPV = nullptr;

    osg::ref_ptr<osg::MatrixTransform> VineyardRoot;
    osg::ref_ptr<osg::MatrixTransform> PVGroup;
    osg::ref_ptr<osg::Node> PVP;
    osg::ref_ptr<osg::Node> PVL;
private:
    static VineyardPlugin *plugin;
};

#endif

