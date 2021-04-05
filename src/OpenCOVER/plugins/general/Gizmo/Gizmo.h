/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MOVE_PLUGIN_H
#define _MOVE_PLUGIN_H
/*
 **                                                            (C)1999 RUS   **
 **                                                                          **
 ** Description: Move arbitrary Objects (2021)                               **
 **                                                                          **
 **                                                                          **
 ** Author: Matthias Epple		                                             **
 **                                                                          **
 ** History:  			   	                                                 **
 ** this replaces the old Move plugin	                                     **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#define MAX_LEVELS 100
#define MAX_UNDOS 100
#include <OpenVRUI/sginterface/vruiActionUserData.h>

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <cover/coVRPlugin.h>
#include <PluginUtil/coVR3DGizmo.h>
#include <PluginUtil/coVR3DTransGizmo.h>
#include <PluginUtil/coVR3DRotGizmo.h>

#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Group.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>
#include <cover/ui/Action.h>

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
    osg::Matrix _initialMat;
    float _lastScaleX;
    float _lastScaleY;
    float _lastScaleZ;
    bool _originalDCS;
};

class Move : public coVRPlugin, public coSelectionListener, public ui::Owner
{
public:
    Move();
    virtual ~Move();
    bool init();
    void preFrame();
    void message(int toWhom, int type, int len, const void *buf);
    void newInteractor(const RenderObject *container, coInteractor *it);
    void addNode(osg::Node *, const RenderObject *);
    void removeNode(osg::Node *, bool isGroup, osg::Node *realNode);

    virtual bool selectionChanged();
    virtual bool pickedObjChanged();

private:


    // UI Menu
    std::unique_ptr<ui::Menu> _UIgizmoMenu;
    std::unique_ptr<ui::Action> _UIparent, _UIchild, _UIundo, _UIredo, _UIreset;
    std::unique_ptr<ui::Button> _UImove, _UImoveAll, _UItranslate,_UIrotate,_UIscale, _UIdisplayNames, _UIlocalCoords;
    std::unique_ptr<ui::Slider> _UIscaleFactor;
    coVRLabel *_label;
    
    // Helper nodes
    osg::Node *_selectedNode;
    osg::Group *_selectedNodesParent;
    osg::Node *_candidateNode;
    osg::Node *_oldNode;


    osg::ref_ptr<osg::MatrixTransform> boundingBoxNode;
    osg::ref_ptr<osg::MatrixTransform> moveDCS;
    MoveInfo *_info;
    osg::Node *_nodes[MAX_LEVELS];
    bool _explicitMode;          // moveAll


    // variables for selection hirarchy & undo / redo operations
    bool _moveSelection;    // do we need this? 
    int level;
    int numLevels;
    int readPos;
    int writePos;
    int maxWritePos;
    osg::Matrix undoMat[MAX_UNDOS];
    osg::MatrixTransform *undoDCS[MAX_UNDOS];
    osg::MatrixTransform *lastDCS;
    
    coTrackerButtonInteraction *interactionA; // interaction to select objects

    typedef std::map<osg::Node *, const RenderObject *> NodeRoMap;
    typedef std::map<const RenderObject *, coInteractor *> RoInteractorMap;
    NodeRoMap nodeRoMap;
    RoInteractorMap roInteractorMap;

    //Gizmo variables
    std::unique_ptr<coVR3DGizmo> _gizmo;
    osg::Matrix _startMoveDCSMat;  //Matrix of helper/transformation node where the gizmo is added to
    bool _gizmoActive{false};

    void getMoveDCS();
    void selectNode(osg::Node* node, const osg::NodePath& intersectedNodePath, bool& isObject, bool& isNewObject, bool& isAlreadySelected);
    void newObject(osg::Node* node,const osg::NodePath& intersectedNodePath);
    bool isSceneNode(osg::Node* node,const osg::NodePath& intersectedNodePath)const;
    void showOrhideName(osg::Node *node);
    void updateScale();
    osg::Node *createBBox();

    // Gizmo functions
    osg::Matrix calcStartMatrix();  // calc global start Matrix for gizmo by iterating over scene graph
    void doMove();                  // apply the movement
    void activateGizmo(const osg::Matrix& m);
    void deactivateGizmo();

    // functions to undo/redo operations
    void addUndo(osg::Matrix &mat, osg::MatrixTransform *dcs);
    void undo();
    void redo();
    void incread();
    void decread();
    void incwrite();
    void decwrite();

};
#endif
