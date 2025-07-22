/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RecordPath_PLUGIN_H
#define _RecordPath_PLUGIN_H
 /****************************************************************************\
 **                                                            (C)2023 HLRS  **
 **                                                                          **
 ** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
 **    Visualises path and workpiece of CNC machining                        **
 **                                                                          **
 ** Author: U.Woessner, A.Kaiser		                                     **
 **                                                                          **
 ** History:  								                                 **
 ** April-05  v1	    				       		                         **
 ** April-23  v2                                                             **
 **                                                                          **
 \****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>

#include <cover/coTabletUI.h>
#include <PluginUtil/colors/coColorMap.h>

#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/LineWidth>
#include <osg/PolygonMode>
#include <osg/Switch>
#include <PluginUtil/coSphere.h>
#include <array>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Label.h>
#include <cover/ui/FileBrowser.h>

#include <cover/ui/CovconfigLink.h>

#include "CNCTree.h"

constexpr size_t MAXSAMPLES = 1200;

using namespace covise;
using namespace opencover;

class CNCPlugin : public coVRPlugin, public ui::Owner
{
public:
    CNCPlugin();
    virtual ~CNCPlugin();
    static CNCPlugin *instance();
    bool init() override;

    int loadGCode(const char *filename, osg::Group *loadParent);
    static int sloadGCode(const char *filename, osg::Group *loadParent, const char *covise_key);
    static int unloadGCode(const char *filename, const char *covise_key);

    void straightFeed(double x, double y, double z, double a, double b, double c, double feedRate, int tool, int gmode);
    void arcFeed(double x, double y, double z, double centerX, double centerY, int rotation, double feedRate, int tool); //rotation positive: counterclockwise
private:

    // this will be called in PreFrame
//    ui::Menu *PathTab = nullptr;
/*    ui::Button*record = nullptr, *playPause = nullptr;
    ui::Action *reset = nullptr, *saveButton = nullptr;
    ui::Button *viewPath = nullptr, *viewlookAt = nullptr, *viewDirections = nullptr;
    ui::Label *numSamples = nullptr;
    ui::EditField *recordRateTUI = nullptr;
    ui::EditField *lengthEdit = nullptr;
    ui::EditField*radiusEdit = nullptr;
    ui::FileBrowser *fileNameBrowser = nullptr;
    ui::SelectionList *renderMethod = nullptr;
    covise::ColorMapSelector colorMap;

    ui::Button *EnableWpButton = nullptr;
    std::unique_ptr<ConfigBool> enableWp;
*/

    std::shared_ptr<config::File>cnc_config;
    ui::Menu *CNCPluginMenu;
    std::unique_ptr<ui::ButtonConfigValue> showPathBtn;
    std::unique_ptr<ui::ButtonConfigValue> showWorkpieceBtn;

    osg::ref_ptr<osg::StateSet> geoState;
    osg::ref_ptr<osg::Material> linemtl;
    osg::ref_ptr<osg::LineWidth> lineWidth;
    void setTimestep(int t) override;

    static CNCPlugin* thePlugin;

//    osg::Vec4 getColor(float pos);
    int frameNumber = 0;
    osg::Group *parentNode = nullptr;
    osg::Vec3Array *vert = nullptr;
    osg::Vec4Array *color = nullptr;
    osg::DrawArrayLengths *primitives = nullptr;
    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;

    void save();

    // scaling factor for inputfile (overall)
    double scaleFactor = 1.0;

    int lastTimestep = 0;

    // path new
    osg::Vec3Array *pathVert = nullptr;
    osg::Vec4Array *pathColor = nullptr;
    osg::DrawArrayLengths *pathPrimitives = nullptr;
    osg::ref_ptr<osg::Geometry> pathGeom;
    osg::ref_ptr<osg::Geode> pathGeode;
    void createPath(osg::Group* loadParent);
    vector<double> arcApproximation(int t);
    double approxLength = 3.2;      //approxLength *2PI = #CornersInPolygon
    osg::Vec4 colorG0, colorG1, colorG2, colorG3;
    bool colorModeGCode = true;

    //workpiece wp
    osg::ref_ptr<osg::Switch> wpSwitch;
    osg::ref_ptr<osg::Group> wpGroup;
    osg::ref_ptr<osg::Geode> wpDynamicGeode;
    osg::ref_ptr<osg::Geode> wpStaticGeode;
    osg::ref_ptr<osg::Geode> wpDynamicGeodeX;
    osg::ref_ptr<osg::Geode> wpStaticGeodeX;
    osg::ref_ptr<osg::Geode> wpDynamicGeodeY;
    osg::ref_ptr<osg::Geode> wpStaticGeodeY;
    osg::ref_ptr<osg::Geometry> wpDynamicGeom;
    osg::ref_ptr<osg::Geometry> wpStaticGeom;
    osg::ref_ptr<osg::Geometry> wpDynamicGeomX;
    osg::ref_ptr<osg::Geometry> wpStaticGeomX;
    osg::ref_ptr<osg::Geometry> wpDynamicGeomY;
    osg::ref_ptr<osg::Geometry> wpStaticGeomY;
    osg::ref_ptr<osg::Vec4Array> wpDynamicColors;
    osg::ref_ptr<osg::Vec4Array> wpStaticColors;
    osg::ref_ptr<osg::Vec3Array> wpDynamicNormals;
    osg::ref_ptr<osg::Vec3Array> wpStaticNormals;
    osg::ref_ptr<osg::Vec3Array> wpDynamicNormalsX;
    osg::ref_ptr<osg::Vec3Array> wpStaticNormalsX;
    osg::ref_ptr<osg::Vec3Array> wpDynamicNormalsY;
    osg::ref_ptr<osg::Vec3Array> wpStaticNormalsY;
    osg::DrawArrayLengths *wpDynamicPrimitives = nullptr;
    osg::DrawArrayLengths *wpStaticPrimitives = nullptr;
    osg::DrawElementsUInt *wpDynamicVerticalPrimX = nullptr; //parallel X
    osg::DrawElementsUInt *wpDynamicVerticalPrimY = nullptr; //parallel Y
    osg::DrawElementsUInt *wpStaticVerticalPrimX = nullptr; //parallel X
    osg::DrawElementsUInt *wpStaticVerticalPrimY = nullptr; //parallel Y
    
    osg::ref_ptr<osg::StateSet> wpStateSet;
    osg::ref_ptr<osg::Material> wpMaterial;
    osg::ref_ptr<osg::LineWidth> wpLineWidth;

    void createWorkpiece(osg::Group*);
    void wpAddQuadsToTree(TreeNode*);
    void wpAddQuadsG0G1(double z, int t, TreeNode*);
    void wpAddQuadsG2G3(double z, int t, TreeNode*);
    
    //double distancePointLine(double px, double py, double x1, double y1, double x2, double y2);
    double distancePointLineSegment(double px, double py, double x1, double y1, double x2, double y2);
    double distancePointPoint(double px, double py, double x1, double y1);
    double anglePointPoint(double px, double py, double x1, double y1);
    bool checkInsideArcG2(double pAngle, double angle1, double angle2);

    void wpTreeToGeometry(osg::Geometry& dynamicGeo, osg::Geometry& staticGeo, osg::Geometry& dynamicGeoX, osg::Geometry& staticGeoX, osg::Geometry& dynamicGeoY, osg::Geometry& staticGeoY);
    void wpTreeToGeoTop(osg::Vec3Array& pointsDynamic, osg::Vec3Array& pointsStatic);
    void wpTreeToGeoSideWalls(osg::Vec3Array& pointsDynamic, osg::Vec3Array& pointsStatic, osg::DrawElementsUInt& wpDynamicVerticalPrimX, osg::DrawElementsUInt& wpDynamicVerticalPrimY, osg::DrawElementsUInt& wpStaticVerticalPrimX, osg::DrawElementsUInt& wpStaticVerticalPrimY);
    void wpAddVertexsForGeo(osg::Vec3Array* points, int minIX, int maxIX, int minIY, int maxIY, double z, int &primPosCounter);
    void wpAddSideForGeo(osg::DrawElementsUInt* wpVerticalPrimitivesX, osg::DrawElementsUInt* wpVerticalPrimitivesY, int primPosTop, int primPosBot, int side);

    void wpCreateTimestepVector(TreeNode*);
    void setWpSize();
    void setWpResolution();
    void setWpMaterial();
    void extractToolInfos(const std::string& filename);
    void setActiveTool(int slot);

    void wpMillCutVec(int t);
    void wpResetCutsVec();

    TreeNode* treeRoot;
    std::vector<std::vector<TreeNode*>> timestepVec;
    int primitivePosCounterDynamic = 0;
    int primitivePosCounterStatic = 0;
    int primitiveResetCounterDynamic = 0;

    std::vector<double> pathX, pathY, pathZ, pathCenterX, pathCenterY, pathFeedRate;
    std::vector<int> pathG, pathTool, pathLineStrip;
    bool wpSizeExtracted = false;
    double wpMinX, wpMaxX, wpMinY, wpMaxY, wpMinZ, wpMaxZ;
    double wpLengthX, wpLengthY, wpLengthZ;

    double wpAllowance = 10.0;
    double wpResolution = 0.05;
    double wpResX, wpResY;              //is
    int wpTotalQuadsX, wpTotalQuadsY;

    struct ToolInfo {
        int toolNumber;
        double diameter;
        double cornerRadius;
        double coneAngle;
        double zMin;
        std::string toolType;
    };
    std::vector<ToolInfo> toolInfoList;
    double cuttingRad = 0.5; // 0.5 / 1000;
    int activeTool;
    double pointAngle = 180;
};

#endif
