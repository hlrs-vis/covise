/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VR_TUI_H
#define CO_VR_TUI_H

/*! \file
 \brief  make OpenCOVER core functionality available via tablet user interface

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   28.04.2004
 */

#include "coTabletUI.h"
#include <util/DLinkList.h>
#include "coVRNavigationManager.h"

#include <osg/Matrix>
#include <osg/Vec3>
#include <osgUtil/RenderBin>
#include <osgUtil/RenderStage>

namespace opencover
{
class coPluginEntry : public coTUIListener
{
public:
    coPluginEntry(const char *libName, coTUITab *, int n);
    virtual ~coPluginEntry();
    char *name;
    coTUIToggleButton *tuiEntry;

private:
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
};

class coInputTUI: public coTUIListener
{
public:
    coInputTUI();
    virtual ~coInputTUI();
    void updateTUI(); // this is called only if anything has changed
	void update(); // this is called once per frame

private:
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
    
    coTUITab *inputTab;
    coTUIFrame *personContainer;
    coTUILabel * personsLabel;
    coTUIComboBox * personsChoice;
    coTUILabel *eyeDistanceLabel;
    coTUIEditFloatField *eyeDistanceEdit;

    coTUIFrame *bodiesContainer;
    coTUILabel *bodiesLabel;
    coTUIComboBox *bodiesChoice;
    
    coTUIEditFloatField *bodyTrans[3];
    coTUILabel *bodyTransLabel[3];
    coTUIEditFloatField *bodyRot[3];
    coTUILabel *bodyRotLabel[3];

    coTUILabel *devicesLabel;
    coTUIComboBox *devicesChoice;
    
    coTUIEditFloatField *deviceTrans[3];
    coTUILabel *deviceTransLabel[3];
    coTUIEditFloatField *deviceRot[3];
    coTUILabel *deviceRotLabel[3];

    coTUIFrame *debugContainer;
    coTUILabel *debugLabel;
    coTUIToggleButton *debugMouseButton;
    coTUIToggleButton *debugDriverButton;
    coTUIToggleButton *debugRawButton;
    coTUIToggleButton *debugTransformedButton;
    coTUIToggleButton *debugMatrices, *debugOther;
	coTUIToggleButton *calibrateTrackingsystem;
	coTUIToggleButton *calibrateToHand;
	coTUILabel *calibrationLabel;

	int calibrationStep;
	osg::Vec3 calibrationPositions[3];
};

class coPluginEntryList : public covise::DLinkList<coPluginEntry *>
{
public:
    coPluginEntryList(coTUITab *);
    virtual ~coPluginEntryList();
    void updateState();

private:
    coTUITab *myTab;
};

class DontDrawBin : public osgUtil::RenderBin::DrawCallback
{
    virtual void drawImplementation(osgUtil::RenderBin *bin, osg::RenderInfo &renderInfo, osgUtil::RenderLeaf *&previous){};
};
class BinListEntry : public coTUIListener
{
public:
    BinListEntry(osgUtil::RenderBin *rb, int num);
    virtual ~BinListEntry();
    virtual void tabletEvent(coTUIElement *tUIItem);
    void updateBin();

private:
    coTUIToggleButton *tb;
    int binNumber;
    osgUtil::RenderBin *renderBin();
};

class BinList : public std::list<BinListEntry *>
{

public:
    BinList();
    virtual ~BinList();
    void refresh();
    void removeAll();
    void updateBins();
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
};

class COVEREXPORT coVRTui : public coTUIListener
{
public:
    coVRTui();
    virtual ~coVRTui();
    void update();
    void config();
    void updateFPS(double fps);
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
    coTUITabFolder *mainFolder;

    coTUITab *getCOVERTab()
    {
        return coverTab;
    };
    coTUIFrame *getTopContainer()
    {
        return topContainer;
    };
    void updateState();
    static coVRTui *instance();

    void getTmpFileName(std::string url);
    coTUIFileBrowserButton *getExtFB();

    BinList *binList;

private:
    static coVRTui *tui;

    coTUITab *coverTab;
    coTUIFrame *topContainer;
    coTUIFrame *bottomContainer;
    coTUIFrame *rightContainer;
    coTUITab *animTab;
    coTUIToggleButton *Animate;
    coTUIFloatSlider *AnimSpeed;
    coTUILabel *AnimSpeedLabel;
    coTUIButton *AnimForward;
    coTUIButton *AnimBack;
    coTUIToggleButton *AnimRotate;
    coTUISlider *AnimTimestep;
    coTUILabel *AnimTimestepLabel;
    coTUISlider *AnimStartStep;
    coTUILabel *AnimStartStepLabel;
    coTUISlider *AnimStopStep;
    coTUILabel *AnimStopStepLabel;
    coTUILabel *PresentationLabel;
    coTUIButton *PresentationForward;
    coTUIButton *PresentationBack;
    coTUIEditIntField *PresentationStep;
    coTUIToggleButton *AnimOscillate;
    coTUIToggleButton *Walk;
    coTUIToggleButton *DebugBins;
    coTUIToggleButton *FlipStereo;
    coTUIComboBox *ShadowChoice;
    coTUIToggleButton *Drive;
    coTUIToggleButton *Fly;
    coTUIToggleButton *XForm;
    coTUIToggleButton *Scale;
    coTUIToggleButton *Collision;
    coTUIToggleButton *DisableIntersection;
    coTUIToggleButton *testImage;
    coTUIToggleButton *Freeze;
    coTUIToggleButton *Wireframe;
    coTUIToggleButton *Menu;
    coTUIEditFloatField *posX;
    coTUIEditFloatField *posY;
    coTUIEditFloatField *posZ;
    coTUIEditFloatField *fovH;
    coTUILabel *fovLabel;
    coTUIEditFloatField *stereoSep;
    coTUILabel *stereoSepLabel;

    coTUIEditFloatField *nearEdit;
    coTUIEditFloatField *farEdit;
    coTUILabel *nearLabel;
    coTUILabel *farLabel;

    coTUIButton *Quit;
    coTUIButton *ViewAll;
    coTUILabel *speedLabel;
    coTUILabel *scaleLabel;
    coTUILabel *viewerLabel;
    coTUILabel *FPSLabel;
    coTUILabel *CFPSLabel;
    coTUIEditFloatField *CFPS;
    coTUIFloatSlider *NavSpeed;
    coTUIFloatSlider *ScaleSlider;
    coTUIColorButton *backgroundColor;
    coTUILabel *backgroundLabel;
    coTUILabel *LODScaleLabel;
    coTUIEditFloatField *LODScaleEdit;

    coTUILabel *debugLabel;
    coTUIEditIntField *debugLevel;
#ifndef NOFB
    coTUIFileBrowserButton *FileBrowser;
    coTUIFileBrowserButton *SaveFileFB;
#endif

    coTUILabel *driveLabel;
    coTUINav *driveNav;
    coTUILabel *panLabel;
    coTUINav *panNav;
    osg::Vec3 viewPos;
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
    coVRNavigationManager::NavMode navigationMode;
    float actScaleFactor;
    float mx, my, x0, y0, relx0, rely0;
    osg::Matrix mat0;
    osg::Matrix mat0_Scale;
    float currentVelocity;
    float driveSpeed;
    float ScaleValue;
    float widthX, widthY, originX, originY;
    int numTimesteps;
    int animationCurrentFrame;
    float animationSpeed;
    float animationSliderMin, animationSliderMax;
    bool animationEnabled;
    bool animationRotate;
    bool animationOscillate;
    double lastUpdateTime;
    coInputTUI *inputTUI;
};
}
#endif
