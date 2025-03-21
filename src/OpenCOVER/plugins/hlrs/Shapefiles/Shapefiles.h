/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Shapefiles_PLUGIN_H
#define _Shapefiles_PLUGIN_H
/****************************************************************************\
 **                                                          (C)2020 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading Shapefiles sensor data       **
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
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/Material>
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

class ShapefilesPlugin: public opencover::coVRPlugin, public ui::Owner, public opencover::coTUIListener
{
public:
    ShapefilesPlugin();
    ~ShapefilesPlugin();
    bool init();
    bool destroy();
    bool update();
    bool loadPVShp(const std::string& filename);
    static ShapefilesPlugin *instance() {return plugin; };

    static int SloadSHP(const char *filename, osg::Group *parent, const char *);
    static int SunloadSHP(const char *filename, const char *);
    
    coTUITab *coLuftTab = nullptr;
    
    ui::Menu *ShapefileTab = nullptr;
    ui::Button *ShowShape = nullptr;

    osg::ref_ptr<osg::MatrixTransform> ShapefilesRoot;
    osg::ref_ptr<osg::MatrixTransform> SHPGroup;
    osg::ref_ptr<osg::Node> PVP;
    osg::ref_ptr<osg::Node> PVL;
    osg::ref_ptr<osg::Geode> geode;
    static osg::ref_ptr<osg::Material> globalDefaultMaterial;

private:
    static ShapefilesPlugin *plugin;
    int loadSHP(const char *filename, osg::Group *parent);
    int unloadSHP(const char *filename);
    void drawTrajectory(OGRLineString* lineString);

};

#endif

