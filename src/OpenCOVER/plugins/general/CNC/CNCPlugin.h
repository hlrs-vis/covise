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
    osg::ref_ptr<osg::Group> wpGroup; //osg::Group *wpGroup = nullptr;
    osg::ref_ptr<osg::Geode> wpTopGeode; //osg::Geode *wpTopGeode = nullptr;
    osg::ref_ptr<osg::Geometry> wpTopGeom;
    osg::ref_ptr<osg::Vec4Array> wpTopColors; //osg::Vec4Array *wpColors = nullptr;
    osg::ref_ptr<osg::Vec3Array> wpTopNormals; //osg::Vec3Array *wpNormals = nullptr;
    osg::DrawArrayLengths *wpTopPrimitives = nullptr;
    osg::DrawElementsUInt *wpVerticalPrimitivesX = nullptr; //parallel X
    osg::DrawElementsUInt *wpVerticalPrimitivesY = nullptr; //parallel Y
    osg::ref_ptr<osg::Geode> wpBotGeode;
    osg::ref_ptr<osg::Geometry> wpBotGeom;
    osg::ref_ptr<osg::Vec4Array> wpBotColors;
    osg::ref_ptr<osg::Vec3Array> wpBotNormals;
    osg::DrawArrayLengths *wpBotPrimitives = nullptr;
    osg::ref_ptr<osg::StateSet> wpStateSet;
    osg::ref_ptr<osg::Material> wpMaterial;
    osg::ref_ptr<osg::LineWidth> wpLineWidth;
    void createWpGeodes(osg::Group *);
    void setWpSize();
    void setWpResolution();
    void setWpMaterial();
    //osg::Geometry *createWpSurface(osg::Vec3 *, osg::Vec3 *, double length_a);
    osg::ref_ptr<osg::Geometry> createWpBottom(double minX, double maxX, double minY, double maxY, double minZ, double maxZ);
    osg::ref_ptr<osg::Geometry> createWpTop(double minX, double maxX, double minY, double maxY, double z);
    osg::ref_ptr<osg::Geometry> createWpTopTree(double minX, double maxX, double minY, double maxY, double z);
    void wpMillCut(osg::Geometry *geo, osg::Vec3Array *piece, int t);
    void wpPrepareMillCut(osg::Geometry* geo, osg::Vec3Array* piece, int t);
    void wpPrepareMillCutTree(double minX, double maxX, double minY, double maxY, double z, int t);
    double distancePointLine(double px, double py, double x1, double y1, double x2, double y2);
    double distancePointLineSegment(double px, double py, double x1, double y1, double x2, double y2);
    void wpResetCuts(osg::Vec3Array *piece, int t);
    void wpCutFaces(osg::Geometry *geo, osg::Vec3Array *piece);
    void wpCutFacesTree(double minX, double maxX, double minY, double maxY, double z);
    void wpAddVertexsForGeo(osg::Vec3Array* points, int minIX, int maxIX, int minIY, int maxIY, double z);

    std::vector<double> pathX, pathY, pathZ;
    double wpMinX, wpMaxX, wpMinY, wpMaxY, wpMinZ, wpMaxZ;
    double wpLengthX, wpLengthY, wpLengthZ;
    double wpAllowance = 0.001;  // 5 / 1000;    //größenzugabe
    //double wpResolution = 0.00002; //0.1 / 1000;   //aimed
    double wpResolution = 0.00010;
    double wpResX, wpResY;              //is
    int wpTotalQuadsX, wpTotalQuadsY;
    //int ix_total;           //deprecated?
    double cuttingRad = 0.0005; // 0.5 / 1000;

    std::vector<int> cuttedQuadsIX, cuttedQuadsIY;
    std::vector<int> cuttedFaces;

    //TreeNode::TreeNode* createTree(int minIX, int maxIX, int minIY, int maxIY, double z);
    TreeNode* createTree(int minIX, int maxIX, int minIY, int maxIY, double z);
    TreeNode* treeRoot;


    int test1, test2;

};

#endif
