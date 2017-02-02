/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MOVE_PLUGIN_H
#define _MOVE_PLUGIN_H
/*
 **                                                            (C)1999 RUS   **
 **                                                                          **
 ** Description: Move arbitrary Objects                                  **
 **                                                                          **
 **                                                                          **
 ** Author: Uwe Woessner		                                    **
 **                                                                          **
 ** History:  			   	                                    **
 ** derived from VRUITest, 2001-01-16	                                    **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#define MAX_LEVELS 100
#define MAX_UNDOS 100
#include <OpenVRUI/sginterface/vruiActionUserData.h>

#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>
#include <cover/coTabletUI.h>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <cover/coVRPlugin.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coPotiMenuItem;
class coTrackerButtonInteraction;
}

namespace opencover
{
class coVRLabel;
}

using namespace vrui;
using namespace opencover;

class MoveInfo : public vruiUserData
{
public:
    MoveInfo();
    ~MoveInfo();
    osg::Matrix initialMat;
    float lastScaleX;
    float lastScaleY;
    float lastScaleZ;
    bool originalDCS;
};

class Move : public coVRPlugin, public coMenuListener, public coTUIListener, public coSelectionListener
{
public:
    Move();
    virtual ~Move();
    bool init();
    void preFrame();
    void message(int type, int len, const void *buf);
    void newInteractor(const RenderObject *container, coInteractor *it);
    void addNode(osg::Node *, RenderObject *);
    void removeNode(osg::Node *, bool isGroup, osg::Node *realNode);

    virtual bool selectionChanged();
    virtual bool pickedObjChanged();

private:
    // toolbar and kids
    coSubMenuItem *pinboardEntry;
    coRowMenu *moveMenu;
    coCheckboxMenuItem *moveToggle;
    coCheckboxMenuItem *showNames;
    coCheckboxMenuItem *local;
    coCheckboxMenuItem *movex;
    coCheckboxMenuItem *movey;
    coCheckboxMenuItem *movez;
    coCheckboxMenuItem *moveh;
    coCheckboxMenuItem *movep;
    coCheckboxMenuItem *mover;
    coVRLabel *label;
    coButtonMenuItem *parentItem;
    coButtonMenuItem *childItem;
    coButtonMenuItem *undoItem;
    coButtonMenuItem *redoItem;
    coButtonMenuItem *resetItem;
    coCheckboxMenuItem *explicitItem;
    coCheckboxMenuItem *moveTransformItem;
    coPotiMenuItem *scaleItem;
    coTUITab *moveTab;
    coTUILabel *moveObjectLabel;
    coTUILabel *moveObjectName;
    coTUILabel *hoeheLabel;
    coTUILabel *breiteLabel;
    coTUILabel *tiefeLabel;
    coTUIEditFloatField *hoeheEdit;
    coTUIEditFloatField *breiteEdit;
    coTUIEditFloatField *tiefeEdit;
    coTUIFloatSlider *ScaleSlider;
    //coTUIEditFloatField *ScaleField;
    coTUIButton *Parent;
    coTUIButton *Child;
    coTUIButton *Undo;
    coTUIButton *Redo;
    coTUIButton *Reset;
    coTUIToggleButton *explicitTUIItem;
    coTUIToggleButton *moveTransformTUIItem;
    coTUIToggleButton *moveEnabled;

    coTUIToggleButton *allowX;
    coTUIToggleButton *allowY;
    coTUIToggleButton *allowZ;
    coTUIToggleButton *allowH;
    coTUIToggleButton *allowP;
    coTUIToggleButton *allowR;
    coTUIToggleButton *aspectRatio;

    void getMoveDCS();
    osg::Node *selectedNode;
    osg::Group *selectedNodesParent;
    osg::ref_ptr<osg::MatrixTransform> boundingBoxNode;
    osg::ref_ptr<osg::Node> transformNode;
    //osg::MatrixTransform *moveDCS;
    osg::ref_ptr<osg::MatrixTransform> moveDCS;
    MoveInfo *info;
    osg::Matrix invStartHandMat;
    osg::Matrix startCompleteMat;
    osg::Matrix invStartCompleteMat;
    osg::Matrix startMoveDCSMat;
    osg::Matrix startBaseMat;
    osg::Vec3 startPickPos;
    osg::Node *nodes[MAX_LEVELS];
    bool explicitMode;
    bool moveTransformMode;
    bool printMode;
    bool moveSelection;
    int level;
    int oldLevel;
    // menu event for buttons and stuff
    void tabletPressEvent(coTUIElement *);
    void menuEvent(coMenuItem *);
    void tabletEvent(coTUIElement *);
    void selectLevel();
    int numLevels;
    bool allowMove;
    bool didMove;
    bool unregister;
    int readPos;
    int writePos;
    int maxWritePos;
    osg::Matrix undoMat[MAX_UNDOS];
    osg::MatrixTransform *undoDCS[MAX_UNDOS];
    osg::Matrix lastMat;
    osg::MatrixTransform *lastDCS;
    osg::Node *oldNode;
    osg::Node *candidateNode;
    double startTime;
    void addUndo(osg::Matrix &mat, osg::MatrixTransform *dcs);
    void undo();
    void redo();
    void incread();
    void decread();
    void incwrite();
    void decwrite();
    void updateScale();

    void restrict(osg::Matrix &mat, bool noRot, bool noTrans);
    osg::Node *createBBox();
    coTrackerButtonInteraction *interactionA; ///< interaction for first button
    coTrackerButtonInteraction *interactionB; ///< interaction for second button

    typedef std::map<osg::Node *, const RenderObject *> NodeRoMap;
    typedef std::map<const RenderObject *, coInteractor *> RoInteractorMap;
    NodeRoMap nodeRoMap;
    RoInteractorMap roInteractorMap;
};
#endif
