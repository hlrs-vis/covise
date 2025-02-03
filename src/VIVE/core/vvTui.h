/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include "vvTabletUI.h"
#include <util/DLinkList.h>
#include "vvNavigationManager.h"

#include <vsg/maths/mat4.h>
#include <vsg/maths/vec3.h>

namespace vive
{
class coInputTUI: public vvTUIListener
{
public:
    coInputTUI(vvTabletUI *tui);
    virtual ~coInputTUI();
    void updateTUI(); // this is called only if anything has changed
	void update(); // this is called once per frame

private:
    virtual void tabletEvent(vvTUIElement *tUIItem);
    virtual void tabletPressEvent(vvTUIElement *tUIItem);
    virtual void tabletReleaseEvent(vvTUIElement *tUIItem);

    vvTabletUI *tui = nullptr;

    vvTUITab *inputTab;
    vvTUIFrame *personContainer;
    vvTUILabel * personsLabel;
    vvTUIComboBox * personsChoice;
    vvTUILabel *eyeDistanceLabel;
    vvTUIEditFloatField *eyeDistanceEdit;

    vvTUIFrame *bodiesContainer;
    vvTUILabel *bodiesLabel;
    vvTUIComboBox *bodiesChoice;
    
    vvTUIEditFloatField *bodyTrans[3];
    vvTUILabel *bodyTransLabel[3];
    vvTUIEditFloatField *bodyRot[3];
    vvTUILabel *bodyRotLabel[3];

    vvTUILabel *devicesLabel;
    vvTUIComboBox *devicesChoice;
    
    vvTUIEditFloatField *deviceTrans[3];
    vvTUILabel *deviceTransLabel[3];
    vvTUIEditFloatField *deviceRot[3];
    vvTUILabel *deviceRotLabel[3];

    vvTUIFrame *debugContainer;
    vvTUILabel *debugLabel;
    vvTUIToggleButton *debugMouseButton;
    vvTUIToggleButton *debugDriverButton;
    vvTUIToggleButton *debugRawButton;
    vvTUIToggleButton *debugTransformedButton;
    vvTUIToggleButton *debugMatrices, *debugOther;
	vvTUIToggleButton *calibrateTrackingsystem;
	vvTUIToggleButton *calibrateToHand;
	vvTUILabel *calibrationLabel;

	int calibrationStep;
	vsg::vec3 calibrationPositions[3];
};
/*
class DontDrawBin : public osgUtil::RenderBin::DrawCallback
{
    virtual void drawImplementation(osgUtil::RenderBin *bin, osg::RenderInfo &renderInfo, osgUtil::RenderLeaf *&previous){};
};
class BinListEntry : public vvTUIListener
{
public:
    BinListEntry(vvTabletUI *tui, osgUtil::RenderBin *rb, int num);
    virtual ~BinListEntry();
    virtual void tabletEvent(vvTUIElement *tUIItem);
    void updateBin();

private:
    vvTUIToggleButton *tb;
    int binNumber;
    osgUtil::RenderBin *renderBin();
};

class BinList : public std::list<BinListEntry *>
{

public:
    BinList(vvTabletUI *tui);
    virtual ~BinList();
    void refresh();
    void removeAll();
    void updateBins();

private:
    vvTabletUI *tui = nullptr;
};
class BinRenderStage : public osgUtil::RenderStage
{
public:
    BinRenderStage(osgUtil::RenderStage &rs)
        : osgUtil::RenderStage(rs){};
    osgUtil::RenderBin *getPreRenderStage()
    {
        if (_preRenderList.size() > 0)
            return (*_preRenderList.begin()).second.get();
        else
            return NULL;
    };
    osgUtil::RenderBin *getPostRenderStage()
    {
        if (_postRenderList.size() > 0)
            return (*_postRenderList.begin()).second.get();
        else
            return NULL;
    };
};*/

class VVCORE_EXPORT vvTui : public vvTUIListener
{
public:
    vvTui(vvTabletUI *tui);
    virtual ~vvTui();
    void update();
    void config();
    void updateFPS(double fps);
    virtual void tabletEvent(vvTUIElement *tUIItem);
    virtual void tabletPressEvent(vvTUIElement *tUIItem);
    virtual void tabletReleaseEvent(vvTUIElement *tUIItem);
    vvTUITabFolder *mainFolder;

    vvTUITab *getviveTab()
    {
        return viveTab;
    };
    vvTUIFrame *getTopContainer()
    {
        return topContainer;
    };
    void updateState();
    static vvTui *instance();

    void getTmpFileName(std::string url);
    vvTUIFileBrowserButton *getExtFB();

    //BinList *binList;

private:
    static vvTui *vrtui;
    vvTabletUI *tui = nullptr;

    vvTUITab *viveTab;
    vvTUIFrame *topContainer;
    vvTUIFrame *bottomContainer;
    vvTUIFrame *rightContainer;
    vvTUITab *presentationTab = nullptr;
    vvTUILabel *PresentationLabel = nullptr;
    vvTUIButton *PresentationForward = nullptr;
    vvTUIButton *PresentationBack = nullptr;
    vvTUIEditIntField *PresentationStep = nullptr;
    vvTUIToggleButton *Walk;
    vvTUIToggleButton *DebugBins;
    vvTUIToggleButton *FlipStereo;
    vvTUIToggleButton *Drive;
    vvTUIToggleButton *Fly;
    vvTUIToggleButton *XForm;
    vvTUIToggleButton *Scale;
    vvTUIToggleButton *Collision;
    vvTUIToggleButton *DisableIntersection;
    vvTUIToggleButton *testImage;
    vvTUIToggleButton *Freeze;
    vvTUIToggleButton *Wireframe;
    vvTUIToggleButton *Menu;
    vvTUIEditFloatField *posX;
    vvTUIEditFloatField *posY;
    vvTUIEditFloatField *posZ;
    vvTUIEditFloatField *fovH;
    vvTUILabel *fovLabel;
    vvTUIEditFloatField *stereoSep;
    vvTUILabel *stereoSepLabel;

    vvTUIEditFloatField *nearEdit;
    vvTUIEditFloatField *farEdit;
    vvTUILabel *nearLabel;
    vvTUILabel *farLabel;

    vvTUIButton *Quit;
    vvTUIButton *ViewAll;
    vvTUILabel *speedLabel;
    vvTUILabel *scaleLabel;
    vvTUILabel *viewerLabel;
    vvTUILabel *FPSLabel;
    vvTUILabel *CFPSLabel;
    vvTUIEditFloatField *CFPS;
    vvTUIFloatSlider *NavSpeed;
    vvTUIFloatSlider *ScaleSlider;
    vvTUIComboBox *SceneUnit;
    vvTUIColorButton *backgroundColor;
    vvTUILabel *backgroundLabel;
    vvTUILabel *LODScaleLabel;
    vvTUIEditFloatField *LODScaleEdit;

    vvTUILabel *debugLabel;
    vvTUIEditIntField *debugLevel;
#ifndef NOFB
    vvTUIFileBrowserButton *FileBrowser;
    vvTUIFileBrowserButton *SaveFileFB;
#endif

    vvTUILabel *driveLabel;
    vvTUINav *driveNav;
    vvTUILabel *panLabel;
    vvTUINav *panNav;
    vsg::dvec3 viewPos;
    void doTabWalk();
    void doTabFly();
    void doTabScale();
    void doTabXform();
    void startTabNav();
    void stopTabNav();

    float getPhiZVerti(float y2, float y1, float x2, float widthX, float widthY);

    float getPhiZHori(float x2, float x1, float y2, float widthY, float widthX);

    float getPhi(float relCoord1, float width1);

    void makeRot(float heading, float pitch, float roll, int headingBool, int pitchBool, int rollBool);

    bool collision;
    vvNavigationManager::NavMode navigationMode;
    float actScaleFactor;
    float mx, my, x0, y0, relx0, rely0;
    vsg::dmat4 mat0;
    vsg::dmat4 mat0_Scale;
    float currentVelocity;
    float driveSpeed;
    float ScaleValue;
    float widthX, widthY, originX, originY;
    double lastUpdateTime;
    coInputTUI *inputTUI;
};
}
