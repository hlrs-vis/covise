/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RecordPath_PLUGIN_H
#define _RecordPath_PLUGIN_H
/****************************************************************************\
**                                                            (C)2005 HLRS  **
**                                                                          **
** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** April-05  v1	    				       		                         **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>

#include <cover/coTabletUI.h>
#include <PluginUtil/coColorMap.h>

#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/LineWidth>
#include <osg/PolygonMode>
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

    void straightFeed(double x, double y, double z, double a, double b, double c, double feedRate);
private:

    // this will be called in PreFrame
    ui::Menu *PathTab = nullptr;
    ui::Button*record = nullptr, *playPause = nullptr;
    ui::Action *reset = nullptr, *saveButton = nullptr;
    ui::Button *viewPath = nullptr, *viewlookAt = nullptr, *viewDirections = nullptr;
    ui::Label *numSamples = nullptr;
    ui::EditField *recordRateTUI = nullptr;
    ui::EditField *lengthEdit = nullptr;
    ui::EditField*radiusEdit = nullptr;
    ui::FileBrowser *fileNameBrowser = nullptr;
    ui::SelectionList *renderMethod = nullptr;
    covise::ColorMapSelector colorMap;

    osg::ref_ptr<osg::StateSet> geoState;
    osg::ref_ptr<osg::Material> linemtl;
    osg::ref_ptr<osg::LineWidth> lineWidth;
    void setTimestep(int t) override;

    static CNCPlugin* thePlugin;

    osg::Vec4 getColor(float pos);
    int frameNumber = 0;
    osg::Group *parentNode = nullptr;
    osg::Vec3Array *vert = nullptr;
    osg::Vec4Array *color = nullptr;
    osg::DrawArrayLengths *primitives = nullptr;
    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;

    void save();

    //workpiece wp
    osg::Group *wpGroup = nullptr;
    osg::Geode *wpTopGeode = nullptr;
    osg::ref_ptr<osg::Geometry> wpTopGeom;
    osg::Vec4Array *wpColors = nullptr;
    osg::DrawArrayLengths *wpTopPrimitives = nullptr;
    osg::DrawElementsUInt *wpVerticalPrimitives = nullptr;
    osg::ref_ptr<osg::StateSet> wpStateSet;
    osg::ref_ptr<osg::Material> wpMaterial;
    osg::ref_ptr<osg::LineWidth> wpLineWidth;
    void createWpGeodes(osg::Group *);
    void setWpSize();
    void setWpResolution();
    void setWpMaterial();
    //osg::Geometry *createWpSurface(osg::Vec3 *, osg::Vec3 *, double length_a);
    //osg::Geometry *createWpTop(std::array<double, 5> *minMaxCoords, double length_a);
    osg::Geometry *createWpTop(double minX, double maxX, double minY, double maxY, double z);
    void wpMillCut(osg::Geometry *geo, osg::Vec3Array *piece, int t);
    double distancePointLine(double px, double py, double x1, double y1, double x2, double y2);
    void wpResetCuts(osg::Vec3Array *piece, int t);
    void wpCutFaces(osg::Geometry *geo, osg::Vec3Array *piece);

    std::vector<double> pathX, pathY, pathZ;
    double wpMinX, wpMaxX, wpMinY, wpMaxY, wpMinZ, wpMaxZ;
    double wpLengthX, wpLengthY, wpLengthZ;
    double wpAllowance = 0.005;  // 5 / 1000;    //größenzugabe
    double wpResolution = 0.0004; //0.1 / 1000;   //aimed
    double wpResX, wpResY;              //is
    int wpTotalQuadsX, wpTotalQuadsY;
    //int ix_total;           //deprecated?
    double cuttingRad = 0.0005; // 0.5 / 1000;

    std::vector<int> cuttedQuadsIX, cuttedQuadsIY;
    std::vector<int> cuttedFaces;

    int test1, test2;

    /*
    // Volume Sticks
    osg::ref_ptr<osg::Geometry> stickGeom;
    osg::ref_ptr<osg::Geode> stickGeode;

    osg::Group *stickParentNode = nullptr;
    osg::Vec3Array *stickVert = nullptr;
    osg::Vec4Array *stickColor = nullptr;
    osg::DrawArrayLengths *stickPrimitives = nullptr;

    // Triangles
    osg::ref_ptr<osg::Geometry> triGeomTop;
    osg::ref_ptr<osg::Geometry> triGeomBot;
    osg::ref_ptr<osg::Geode> triGeode;

    osg::Group *triParentNode = nullptr;
    osg::Vec3Array *triVertTop = nullptr;
    osg::Vec3Array *triVertBot = nullptr;
    osg::Vec4Array *triColor = nullptr;
    osg::DrawArrayLengths *triPrimitivesTop = nullptr;
    osg::DrawElementsUInt *triPrimitivesBot = nullptr;
    */
};
#endif
