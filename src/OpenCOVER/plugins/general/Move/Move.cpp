/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coIntersection.h>

#include <config/CoviseConfig.h>

#include "Move.h"

//#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <cover/VRPinboard.h>
#include <osg/BoundingBox>
#include <osg/Quat>
#include <osg/Geode>
#include <osg/NodeVisitor>
#include <osg/Material>
#include <osg/Vec4>
#include <osg/fast_back_stack>
#include <osg/Shape>
#include <osg/PolygonMode>
#include <osg/ShapeDrawable>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <cover/coVRLabel.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <cover/coVRTui.h>
#include <OpenVRUI/osg/OSGVruiUserDataCollection.h>

#include <PluginUtil/PluginMessageTypes.h>
#include <net/tokenbuffer.h>

using covise::coCoviseConfig;
using covise::TokenBuffer;

typedef osg::fast_back_stack<osg::ref_ptr<osg::RefMatrix> > MatrixStack;

class BBoxVisitor : public osg::NodeVisitor
{
public:
    osg::BoundingBox bbox;
    BBoxVisitor()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ACTIVE_CHILDREN)
    {
        matStack.clear();
    }

    void apply(osg::Node &node)
    {
        osg::Transform *mt;
        osg::Geode *geo;

        osg::ref_ptr<osg::RefMatrix> M = matStack.back();
        if ((geo = dynamic_cast<osg::Geode *>(&node)))
        {
            unsigned int i;
            osg::BoundingBox bb;
            for (i = 0; i < geo->getNumDrawables(); i++)
            {
                bb.expandBy(geo->getDrawable(i)->getBound());
            }
            if (M.get())
            {
                bbox.expandBy(osg::Vec3(bb.xMin(), bb.yMin(), bb.zMin()) * *M);
                bbox.expandBy(osg::Vec3(bb.xMax(), bb.yMax(), bb.zMax()) * *M);
            }
            else
            {
                bbox.expandBy(osg::Vec3(bb.xMin(), bb.yMin(), bb.zMin()));
                bbox.expandBy(osg::Vec3(bb.xMax(), bb.yMax(), bb.zMax()));
            }
        }
        if ((mt = dynamic_cast<osg::Transform *>(&node)))
        {
            osg::ref_ptr<osg::RefMatrix> matrix = new osg::RefMatrix;
            mt->computeLocalToWorldMatrix(*matrix, this);
            matStack.push_back(matrix);
        }
        traverse(node);
        if ((mt = dynamic_cast<osg::Transform *>(&node)))
        {
            matStack.pop_back();
        }
    }

private:
    MatrixStack matStack;
};

MoveInfo::MoveInfo()
{
    initialMat.makeIdentity();
    originalDCS = false;
    lastScaleX = 1;
    lastScaleY = 1;
    lastScaleZ = 1;
}

MoveInfo::~MoveInfo()
{
}

Move::Move()
{
}

bool Move::init()
{
    //fprintf(stderr,"Move::Move\n");
    oldLevel = level = 0;
    numLevels = 0;
    info = NULL;
    didMove = false;
    moveSelection = false;
    selectedNode = NULL;
    explicitMode = coCoviseConfig::isOn("COVER.Plugin.Move.Explicit", true);
    printMode = coCoviseConfig::isOn("COVER.Plugin.Move.Print", false);
    readPos = writePos = maxWritePos = 0;
    // create the text
    osg::Vec4 fgcolor(0, 1, 0, 1);
    osg::Vec4 bgcolor(0.5, 0.5, 0.5, 0.5);
    float lineLen = 0.04 * cover->getSceneSize();
    float fontSize = 0.02 * cover->getSceneSize();
    label = new coVRLabel("test", fontSize, lineLen, fgcolor, bgcolor);
    label->hide();

    pinboardEntry = new coSubMenuItem("Move Objects...");
    cover->getMenu()->add(pinboardEntry);
    moveMenu = new coRowMenu("Move", cover->getMenu());

    moveTab = new coTUITab("Move", coVRTui::instance()->mainFolder->getID());
    moveTab->setPos(0, 0);

    allowX = new coTUIToggleButton("X", moveTab->getID());
    allowY = new coTUIToggleButton("Y", moveTab->getID());
    allowZ = new coTUIToggleButton("Z", moveTab->getID());
    allowH = new coTUIToggleButton("H", moveTab->getID());
    allowP = new coTUIToggleButton("P", moveTab->getID());
    allowR = new coTUIToggleButton("R", moveTab->getID());
    Child = new coTUIButton("Child", moveTab->getID());
    Parent = new coTUIButton("Parent", moveTab->getID());
    Undo = new coTUIButton("Undo", moveTab->getID());
    Redo = new coTUIButton("Redo", moveTab->getID());
    Reset = new coTUIButton("Reset", moveTab->getID());
    explicitTUIItem = new coTUIToggleButton("MoveAll", moveTab->getID());
    moveEnabled = new coTUIToggleButton("Move", moveTab->getID());
    //ScaleField = new coTUIEditFloatField("Scale",moveTab->getID());
    ScaleSlider = new coTUIFloatSlider("ScaleSlider", moveTab->getID());
    moveObjectLabel = new coTUILabel("ObjectName:", moveTab->getID());
    moveObjectName = new coTUILabel("NoName", moveTab->getID());
    hoeheLabel = new coTUILabel("Hoehe:", moveTab->getID());
    breiteLabel = new coTUILabel("Breite:", moveTab->getID());
    tiefeLabel = new coTUILabel("Tiefe", moveTab->getID());
    hoeheEdit = new coTUIEditFloatField("hoehe", moveTab->getID());
    breiteEdit = new coTUIEditFloatField("breite", moveTab->getID());
    tiefeEdit = new coTUIEditFloatField("tiefe", moveTab->getID());
    aspectRatio = new coTUIToggleButton("KeepAspectRatio", moveTab->getID());
    allowX->setEventListener(this);
    allowY->setEventListener(this);
    allowZ->setEventListener(this);
    allowH->setEventListener(this);
    allowP->setEventListener(this);
    allowR->setEventListener(this);
    aspectRatio->setEventListener(this);
    Child->setEventListener(this);
    Parent->setEventListener(this);
    Undo->setEventListener(this);
    Redo->setEventListener(this);
    Reset->setEventListener(this);
    explicitTUIItem->setEventListener(this);
    moveEnabled->setEventListener(this);
    //ScaleField->setEventListener(this);
    ScaleSlider->setEventListener(this);
    moveObjectLabel->setEventListener(this);
    hoeheEdit->setEventListener(this);
    breiteEdit->setEventListener(this);
    tiefeEdit->setEventListener(this);
    ScaleSlider->setMin(0.1);
    ScaleSlider->setMax(10);
    ScaleSlider->setValue(1.0);
    hoeheEdit->setValue(0.0);
    breiteEdit->setValue(0.0);
    tiefeEdit->setValue(0.0);
    //ScaleField->setValue(1);
    allowX->setState(true);
    allowY->setState(false);
    allowZ->setState(true);
    allowH->setState(false);
    allowP->setState(false);
    allowR->setState(true);
    aspectRatio->setState(true);
    allowX->setPos(0, 0);
    allowY->setPos(0, 1);
    allowZ->setPos(0, 2);
    allowH->setPos(1, 0);
    allowP->setPos(1, 1);
    allowR->setPos(1, 2);
    Child->setPos(0, 3);
    Parent->setPos(1, 3);
    Undo->setPos(0, 4);
    Redo->setPos(1, 4);
    Reset->setPos(0, 5);
    explicitTUIItem->setPos(0, 6);
    moveEnabled->setPos(1, 6);
    //ScaleField->setPos(0,7);
    ScaleSlider->setPos(0, 7);
    moveObjectLabel->setPos(0, 8);
    moveObjectName->setPos(1, 8);

    hoeheLabel->setPos(0, 9);
    breiteLabel->setPos(0, 10);
    tiefeLabel->setPos(0, 11);
    hoeheEdit->setPos(1, 9);
    breiteEdit->setPos(1, 10);
    tiefeEdit->setPos(1, 11);
    aspectRatio->setPos(2, 10);

    pinboardEntry->setMenu(moveMenu);
    moveToggle = new coCheckboxMenuItem("Move", false);
    showNames = new coCheckboxMenuItem("Display Names", false);
    movex = new coCheckboxMenuItem("X", true);
    movey = new coCheckboxMenuItem("Y", false);
    movez = new coCheckboxMenuItem("Z", true);
    moveh = new coCheckboxMenuItem("H", false);
    movep = new coCheckboxMenuItem("P", false);
    mover = new coCheckboxMenuItem("R", true);
    local = new coCheckboxMenuItem("local", true);
    parentItem = new coButtonMenuItem("Parent");
    childItem = new coButtonMenuItem("Child");
    undoItem = new coButtonMenuItem("Undo");
    redoItem = new coButtonMenuItem("Redo");
    resetItem = new coButtonMenuItem("Reset");
    explicitItem = new coCheckboxMenuItem("MoveAll", !explicitMode);
    scaleItem = new coPotiMenuItem("Scale", 0.1, 10, 1.0);

    moveMenu->add(moveToggle);
    moveMenu->add(explicitItem);
    moveMenu->add(showNames);
    moveMenu->add(parentItem);
    moveMenu->add(childItem);
    moveMenu->add(movex);
    moveMenu->add(movey);
    moveMenu->add(movez);
    moveMenu->add(moveh);
    moveMenu->add(movep);
    moveMenu->add(mover);
    moveMenu->add(local);
    moveMenu->add(undoItem);
    moveMenu->add(redoItem);
    moveMenu->add(resetItem);
    moveMenu->add(scaleItem);
    moveToggle->setMenuListener(this);
    parentItem->setMenuListener(this);
    childItem->setMenuListener(this);
    resetItem->setMenuListener(this);
    undoItem->setMenuListener(this);
    redoItem->setMenuListener(this);
    explicitItem->setMenuListener(this);
    scaleItem->setMenuListener(this);

    boundingBoxNode = new osg::MatrixTransform();
    boundingBoxNode->addChild(createBBox());
    interactionA = new coTrackerButtonInteraction(coInteraction::ButtonA, "Move", coInteraction::Menu);
    interactionB = new coTrackerButtonInteraction(coInteraction::ButtonB, "Move", coInteraction::Menu);
    candidateNode = NULL;
    oldNode = NULL;
    selectedNodesParent = NULL;
    coVRSelectionManager::instance()->addListener(this);

    return true;
}

// this is called if the plugin is removed at runtime
Move::~Move()
{
    delete label;
    // we probably have to delete all move infos...

    delete moveTab;
    delete allowX;
    delete allowY;
    delete allowZ;
    delete allowH;
    delete allowP;
    delete allowR;
    delete Child;
    delete Parent;
    delete Undo;
    delete Redo;
    delete Reset;
    delete explicitTUIItem;
    delete moveEnabled;
    delete ScaleSlider;
    delete moveObjectLabel;
    delete moveObjectName;
    delete hoeheEdit;
    delete breiteLabel;
    delete tiefeLabel;
    delete hoeheLabel;
    delete breiteEdit;
    delete tiefeEdit;
    delete aspectRatio;

    delete redoItem;
    delete undoItem;
    delete resetItem;
    delete interactionA;
    delete interactionB;
    delete pinboardEntry;
    delete moveMenu;
    delete moveToggle;
    delete showNames;
    delete parentItem;
    delete childItem;
    delete movex;
    delete movey;
    delete movez;
    delete moveh;
    delete movep;
    delete mover;
    delete local;
    delete scaleItem;
    delete explicitItem;
    while (boundingBoxNode->getNumParents())
        boundingBoxNode->getParent(0)->removeChild(boundingBoxNode.get());

    coVRSelectionManager::instance()->removeListener(this);
    fprintf(stderr, "Move::~Move\n");
}

void Move::message(int type, int len, const void *buf)
{
    if (type == PluginMessageTypes::MoveAddMoveNode)
    {
    }
    else if (type == PluginMessageTypes::MoveMoveNode)
    {
        osg::Matrix m;
        std::string path;
        TokenBuffer tb((const char *)buf, len);
        tb >> path;
        selectedNodesParent = dynamic_cast<osg::Group *>(coVRSelectionManager::validPath(path));
        tb >> path;
        selectedNode = coVRSelectionManager::validPath(path);
        getMoveDCS();

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                tb >> m(i, j);
        moveDCS->setMatrix(m);
    }
}

void Move::selectLevel()
{
    //cerr << "numLevels: " << numLevels << endl;
    //cerr << "level: " << level << endl;
    if (level >= numLevels)
    {
        level = 0;
    }
}

bool Move::selectionChanged()
{

    std::list<osg::ref_ptr<osg::Node> > selectedNodeList = coVRSelectionManager::instance()->getSelectionList();
    std::list<osg::ref_ptr<osg::Group> > selectedParentList = coVRSelectionManager::instance()->getSelectedParentList();

    std::list<osg::ref_ptr<osg::Node> >::iterator nodeIter = selectedNodeList.end();
    std::list<osg::ref_ptr<osg::Group> >::iterator parentIter = selectedParentList.end();
    if (selectedNodeList.size() == 0)
    {
        moveDCS = NULL;
        moveObjectLabel->setLabel("None");
    }
    else
    {
        nodeIter--;
        parentIter--;
        const char *name = (*nodeIter)->getName().c_str();
        selectedNode = (*nodeIter).get();
        selectedNodesParent = (*parentIter).get();
        if (name)
            moveObjectName->setLabel(name);
        else
            moveObjectName->setLabel("NoName");
        if (info)
        {
            scaleItem->setValue(info->lastScaleY);
            ScaleSlider->setValue(info->lastScaleY);
        }
        updateScale();
    }
    return true;
}

bool Move::pickedObjChanged()
{
    return true;
}

void Move::preFrame()
{
    bool doUndo = false;
    osg::Node *node;
    coPointerButton *button = cover->getPointerButton();
    node = cover->getIntersectedNode();
    const osg::NodePath &intersectedNodePath = cover->getIntersectedNodePath();
    bool isSceneNode = false;
    if (node)
    {
        for (std::vector<osg::Node *>::const_iterator iter = intersectedNodePath.begin();
             iter != intersectedNodePath.end(); ++iter)
        {
            if ((*iter) == cover->getObjectsRoot())
            {
                isSceneNode = true;
                break;
            }
        }
    }
    if (isSceneNode && (!coVRSelectionManager::isHelperNode(node)))
    {
        if (showNames->getState())
        {
            if (node && !node->getName().empty())
            {

                label->setString(node->getName().c_str());
                label->setPosition(cover->getIntersectionHitPointWorld());
                label->show();
            }
            else
                label->hide();
        }
        else
            label->hide();

        bool isObject = false;
        //bool isNewObject=false;
        //select a node
        if (node && moveToggle->getState() && (node != oldNode) && (node != selectedNode))
        {
            // this might be a new candidate for a movable node

            osg::Matrix mat; //,dcsMat;
            mat.makeIdentity();
            osg::Node *currentNode;
            currentNode = node;
            //cerr << "test: " << node << endl;
            //cerr << "lev: " << level << endl;
            while (currentNode != NULL)
            {
                const char *nodeName = currentNode->getName().c_str();
                if (explicitMode)
                {
                    // do not touch nodes underneeth NoMove nodes
                    if (nodeName && strncmp(nodeName, "DoMove", 6) == 0)
                    {
                        isObject = true;
                        doUndo = true;
                    }
                    if (currentNode == cover->getObjectsRoot())
                    {
                        //isNewObject = true;
                        break;
                    }
                }
                else
                {
                    if (currentNode == cover->getObjectsRoot())
                    {
                        isObject = true;
                        //isNewObject = true;
                        doUndo = true;
                        break;
                    }
                }
                // do not touch nodes underneeth NoMove nodes
                if (nodeName && strncmp(nodeName, "NoMove", 6) == 0)
                {
                    isObject = false;
                    //isNewObject = false;
                    doUndo = false;
                    break;
                }
                if (currentNode == selectedNode)
                {
                    isObject = true; // already selected
                    //isNewObject = false;
                    doUndo = true;
                    break;
                }
                if (currentNode->getNumParents() > 0)
                {
                    std::vector<osg::Node *>::const_iterator iter = intersectedNodePath.end();
                    for (iter--; iter != intersectedNodePath.begin(); iter--)
                    {
                        if ((*iter) == currentNode)
                        {
                            iter--;
                            currentNode = *iter;
                        }
                    }
                }
                else
                    currentNode = NULL;
            }
            if (isObject /*|| isNewObject*/)
            {
                candidateNode = node;
            }
            else
            {
                candidateNode = NULL;
            }
        }
        if (node == NULL)
        {
            candidateNode = NULL;
        }
        oldNode = node;
    }
    else
    {
        if ((interactionA->getState() != coInteraction::Active) && (interactionB->getState() != coInteraction::Active))
        {
            oldNode = NULL;
            node = NULL;
            candidateNode = NULL;
            selectedNode = NULL;
            if (moveSelection)
            {
                coVRSelectionManager::instance()->clearSelection();
                moveSelection = false;
            }
        }
    }
    // if we point towards a selected or candidate Node
    if (node && moveToggle->getState() && ((node == candidateNode) || (node == selectedNode)))
    { // register the interactions
        if (!interactionA->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(interactionA);
        }
    }
    // if we don't point to a node or point to a non candidate Node
    if (!node || (((node != candidateNode) && (node != selectedNode)) && ((interactionB->isRegistered()) || (interactionA->isRegistered()))))
    { //unregister if possible;
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
        if (interactionB->isRegistered() && (interactionB->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionB);
        }
    }

    if (interactionA->wasStarted())
    {
        // select this node
        if (node)
        {
            bool isObject = false;
            bool isNewObject = false;
            osg::Matrix mat, dcsMat;
            mat.makeIdentity();
            osg::Node *currentNode;
            currentNode = node;
            //cerr << "test: " << node << endl;
            //cerr << "lev: " << level << endl;
            while (currentNode != NULL)
            {
                const char *nodeName = currentNode->getName().c_str();
                if (explicitMode)
                {
                    // do not touch nodes underneeth NoMove nodes
                    if (nodeName && strncmp(nodeName, "DoMove", 6) == 0)
                    {
                        isObject = true;
                        doUndo = true;
                    }
                    if (currentNode == cover->getObjectsRoot())
                    {
                        isNewObject = true;
                        break;
                    }
                }
                else
                {
                    if (currentNode == cover->getObjectsRoot())
                    {
                        isObject = true;
                        isNewObject = true;
                        doUndo = true;
                        break;
                    }
                }
                // do not touch nodes underneeth NoMove nodes
                if (nodeName && strncmp(nodeName, "NoMove", 6) == 0)
                {
                    isObject = false;
                    isNewObject = false;
                    doUndo = false;
                    break;
                }
                if (currentNode == selectedNode)
                {
                    isObject = true; // already selected
                    isNewObject = false;
                    cerr << "already selected: " << level << endl;
                    doUndo = true;
                    break;
                }
                if (currentNode->getNumParents() > 0)
                {
                    std::vector<osg::Node *>::const_iterator iter = intersectedNodePath.end();
                    for (iter--; iter != intersectedNodePath.begin(); iter--)
                    {
                        if ((*iter) == currentNode)
                        {
                            iter--;
                            currentNode = *iter;
                        }
                    }
                }
                else
                    currentNode = NULL;
            }
            if (isNewObject)
            {

                currentNode = node;
                numLevels = 0;
                while (currentNode != NULL)
                {
                    if (currentNode == cover->getObjectsRoot())
                    {
                        break;
                    }
                    const char *nodeName = currentNode->getName().c_str();
                    if (nodeName && (strncmp(nodeName, "Default", 7) == 0))
                    {
                        level = numLevels;
                    }
                    info = (MoveInfo *)OSGVruiUserDataCollection::getUserData(currentNode, "MoveInfo");

                    if (info)
                    {
                        scaleItem->setValue(info->lastScaleY);
                        ScaleSlider->setValue(info->lastScaleY);
                    }
                    if (info && !info->originalDCS)
                    {
                    }
                    else
                    {
                        nodes[numLevels] = currentNode;
                        numLevels++;
                    }
                    if (currentNode->getNumParents() > 0)
                    {
                        std::vector<osg::Node *>::const_iterator iter = intersectedNodePath.end();
                        for (iter--; iter != intersectedNodePath.begin(); iter--)
                        {
                            if ((*iter) == currentNode)
                            {
                                iter--;
                                currentNode = *iter;
                            }
                        }
                    }
                    else
                        currentNode = NULL;
                }

                osg::Group *parent = NULL;
                coVRSelectionManager::instance()->clearSelection();

                std::vector<osg::Node *>::const_iterator iter = intersectedNodePath.end();
                for (iter--; iter != intersectedNodePath.begin(); iter--)
                {
                    if ((*iter) == nodes[level])
                    {
                        if (iter != intersectedNodePath.begin())
                            iter--;
                        parent = (*iter)->asGroup();
                        while (parent && coVRSelectionManager::isHelperNode(parent))
                        {
                            iter--;
                            parent = (*iter)->asGroup();
                        }
                        selectedNode = nodes[level];
                        if (parent)
                        {
                            coVRSelectionManager::instance()->addSelection(parent, selectedNode);
                            coVRSelectionManager::instance()->pickedObjChanged();
                            moveSelection = true;
                        }
                        else
                        {
                            cerr << "parent not found" << endl;
                        }
                    }
                }

                if (!interactionB->isRegistered())
                {
                    coInteractionManager::the()->registerInteraction(interactionB);
                }
                allowMove = false;
            }
            if ((!isObject) && (selectedNode))
            {
                coVRSelectionManager::instance()->clearSelection();
                selectedNode = NULL;
                numLevels = 0;
                allowMove = false;
            }
        }
    }

    if ((selectedNode) && (!(interactionA->isRegistered() || interactionB->isRunning())) && button->getState())
    {
        if (moveSelection)
        {
            coVRSelectionManager::instance()->clearSelection();
            moveSelection = false;
        }
        selectedNode = NULL;
        numLevels = 0;
        allowMove = false;
    }

    // do the movement
    //check for move dcs
    osg::Node *currentNode;
    osg::Matrix dcsMat;

    if (selectedNode)
    {
        if (interactionA->wasStarted() || interactionB->wasStarted())
        {
            getMoveDCS();
            // start of interaction (button press)
            currentNode = moveDCS->getParent(0);
            startBaseMat.makeIdentity();
            while (currentNode != NULL)
            {
                if (dynamic_cast<osg::MatrixTransform *>(currentNode))
                {
                    dcsMat = ((osg::MatrixTransform *)currentNode)->getMatrix();
                    startBaseMat.postMult(dcsMat);
                }
                if (currentNode->getNumParents() > 0)
                    currentNode = currentNode->getParent(0);
                else
                    currentNode = NULL;
            }
            startMoveDCSMat = moveDCS->getMatrix();
            startCompleteMat = startBaseMat;
            startCompleteMat.preMult(startMoveDCSMat);

            if (doUndo)
            {
                didMove = true;
                addUndo(startMoveDCSMat, moveDCS.get());
            }

            if (!invStartCompleteMat.invert(startCompleteMat))
                fprintf(stderr, "Move::inv startCompleteMat is singular\n");
            if (!invStartHandMat.invert(cover->getPointerMat()))
                fprintf(stderr, "Move::inv getPointerMat is singular\n");
            startPickPos = cover->getIntersectionHitPointWorld();
            startTime = cover->frameTime();
        }
        if (interactionA->isRunning() && (moveDCS != NULL))
        { // ongoing interaction (left mousebutton)
            osg::Matrix moveMat, currentBaseMat, currentNewMat, newDCSMat, invcurrentBaseMat, localRot, tmpMat, tmp2Mat;
            moveMat.mult(invStartHandMat, cover->getPointerMat());

            if (moveDCS->getNumParents() > 0)
            {
                currentNode = moveDCS->getParent(0);
            }
            else
                currentNode = NULL;
            currentBaseMat.makeIdentity();
            while (currentNode != NULL)
            {
                if (dynamic_cast<osg::MatrixTransform *>(currentNode))
                {
                    dcsMat = ((osg::MatrixTransform *)currentNode)->getMatrix();
                    currentBaseMat.postMult(dcsMat);
                }
                if (currentNode->getNumParents() > 0)
                    currentNode = currentNode->getParent(0);
                else
                    currentNode = NULL;
            }
            if (!invcurrentBaseMat.invert(currentBaseMat))
                fprintf(stderr, "Move::inv currentBaseMat is singular\n");
            // frei
            if (!local->getState())
            {
                restrict(moveMat, true, false);
            }

            tmpMat.mult(startCompleteMat, moveMat);
            localRot.mult(tmpMat, invStartCompleteMat);
            if (local->getState())
            {
                restrict(localRot, true, false);
            }
            currentNewMat.mult(localRot, startCompleteMat);
            newDCSMat.mult(currentNewMat, invcurrentBaseMat);
            if (allowMove)
            {

                TokenBuffer tb;
                std::string path = coVRSelectionManager::generatePath(selectedNodesParent);
                tb << path;
                path = coVRSelectionManager::generatePath(selectedNode);
                tb << path;

                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        tb << newDCSMat(i, j);

                cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::MoveMoveNode, tb.get_length(), tb.get_data());
               
                vrui::vruiUserData  *info = OSGVruiUserDataCollection::getUserData(selectedNode, "RevitInfo");
                if(info!=NULL)
                {
                    cover->sendMessage(this, "Revit",
                        PluginMessageTypes::MoveMoveNode, tb.get_length(), tb.get_data());
                }

                if (printMode)
                {
                    osg::Matrix rotMat;
                    osg::Vec3 Trans;
                    Trans = newDCSMat.getTrans();
                    cerr << endl << "VRML Translation:" << endl;
                    osg::Quat quat;
                    quat.set(newDCSMat);
                    double angle, x, y, z;
                    quat.getRotate(angle, x, y, z);
                    cerr << "translation " << Trans[0] << " " << Trans[1] << " " << Trans[2] << endl;
                    cerr << "rotation " << x << " " << y << " " << z << " " << (angle / 180.0) * M_PI << endl;
                }
            }
            else
            {
                if ((cover->frameTime() - startTime) > 1)
                    allowMove = true;
            }
        }
        if (interactionB->isRunning() && (moveDCS != NULL))
        { // ongoing interaction (right mousebutton)
            osg::Matrix moveMat, currentBaseMat, currentNewMat, newDCSMat, invcurrentBaseMat, localRot, tmpMat, tmp2Mat;
            moveMat.mult(invStartHandMat, cover->getPointerMat());

            currentNode = moveDCS->getParent(0);
            currentBaseMat.makeIdentity();
            while (currentNode != NULL)
            {
                if (dynamic_cast<osg::MatrixTransform *>(currentNode))
                {
                    dcsMat = ((osg::MatrixTransform *)currentNode)->getMatrix();
                    currentBaseMat.postMult(dcsMat);
                }
                if (currentNode->getNumParents() > 0)
                    currentNode = currentNode->getParent(0);
                else
                    currentNode = NULL;
            }
            if (!invcurrentBaseMat.invert(currentBaseMat))
                fprintf(stderr, "Move::inv currentBaseMat is singular\n");
            // frei
            if (!local->getState())
            {
                restrict(moveMat, false, true);
            }

            tmpMat.mult(startCompleteMat, moveMat);
            localRot.mult(tmpMat, invStartCompleteMat);
            if (local->getState())
            {
                restrict(localRot, false, true);
            }
            currentNewMat.mult(localRot, startCompleteMat);
            newDCSMat.mult(currentNewMat, invcurrentBaseMat);

            if (allowMove)
            {

                TokenBuffer tb;
                std::string path = coVRSelectionManager::generatePath(selectedNodesParent);
                tb << path;
                path = coVRSelectionManager::generatePath(selectedNode);
                tb << path;

                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        tb << newDCSMat(i, j);

                cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::MoveMoveNode, tb.get_length(), tb.get_data());
            
                vrui::vruiUserData  *info = OSGVruiUserDataCollection::getUserData(selectedNode, "RevitInfo");
                if(info!=NULL)
                {
                    cover->sendMessage(this, "Revit",
                        PluginMessageTypes::MoveMoveNode, tb.get_length(), tb.get_data());
                }
                if (printMode)
                {
                    osg::Matrix rotMat;
                    osg::Vec3 Trans;
                    Trans = newDCSMat.getTrans();
                    cerr << endl << "VRML Translation:" << endl;
                    osg::Quat quat;
                    quat.set(newDCSMat);
                    double angle, x, y, z;
                    quat.getRotate(angle, x, y, z);
                    cerr << "translation " << Trans[0] << " " << Trans[1] << " " << Trans[2] << endl;
                    cerr << "rotation " << x << " " << y << " " << z << " " << (angle / 180.0) * M_PI << endl;
                }
            }
            else
            {
                if ((cover->frameTime() - startTime) > 1)
                    allowMove = true;
            }
        }
        if ((interactionA->wasStopped() || interactionB->wasStopped()) && (moveDCS != NULL))
        {
            if (allowMove)
            {
                osg::Matrix mat;
                mat = moveDCS->getMatrix();
                if (didMove)
                {
                    
                    TokenBuffer tb;
                    std::string path = coVRSelectionManager::generatePath(selectedNodesParent);
                    tb << path;
                    path = coVRSelectionManager::generatePath(selectedNode);
                    tb << path;

                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 4; j++)
                            tb << mat(i, j);
                    vrui::vruiUserData  *info = OSGVruiUserDataCollection::getUserData(selectedNode, "RevitInfo");
                    if(info!=NULL)
                    {
                        cover->sendMessage(this, "Revit",
                            PluginMessageTypes::MoveMoveNode, tb.get_length(), tb.get_data());
                    }
                    addUndo(mat, moveDCS.get());
                }
                didMove = false;
            }
            allowMove = true;
        }
    }
}

void Move::restrict(osg::Matrix &mat, bool noRot, bool noTrans)
{
    coCoord coord;
    coord = mat;
    if (noTrans)
    {
        coord.xyz[0] = coord.xyz[1] = coord.xyz[2] = 0;
    }
    else
    {
        if (!movex->getState())
        {
            coord.xyz[0] = 0;
        }
        if (!movey->getState())
        {
            coord.xyz[1] = 0;
        }
        if (!movez->getState())
        {
            coord.xyz[2] = 0;
        }
    }
    if (noRot)
    {
        coord.hpr[0] = coord.hpr[1] = coord.hpr[2] = 0;
    }
    else
    {
        if (!moveh->getState())
        {
            coord.hpr[0] = 0;
        }
        if (!movep->getState())
        {
            coord.hpr[1] = 0;
        }
        if (!mover->getState())
        {
            coord.hpr[2] = 0;
        }
    }
    coord.makeMat(mat);
}

void Move::getMoveDCS()
{
    // if this is a DCS, then use this one
    moveDCS = NULL;

    osg::Group *selectionHelperNode = coVRSelectionManager::getHelperNode(selectedNodesParent, selectedNode, coVRSelectionManager::MOVE);
    moveDCS = dynamic_cast<osg::MatrixTransform *>(selectionHelperNode);
    //cerr << "moveDCS=" << moveDCS.get() << endl;
    if (moveDCS.get() == NULL)
    {
        moveDCS = new osg::MatrixTransform();
        //cerr << "new moveDCS" << moveDCS.get() << endl;
        info = new MoveInfo();
        OSGVruiUserDataCollection::setUserData(moveDCS.get(), "MoveInfo", info);
        coVRSelectionManager::insertHelperNode(selectedNodesParent, selectedNode, moveDCS.get(), coVRSelectionManager::MOVE);
    }

    info = (MoveInfo *)OSGVruiUserDataCollection::getUserData(moveDCS.get(), "MoveInfo");
    if (info)
    {
        scaleItem->setValue(info->lastScaleY);
        ScaleSlider->setValue(info->lastScaleY);
    }
}

void Move::updateScale()
{
    if (info && selectedNode)
    {
        BBoxVisitor bbv;
        bbv.apply(*selectedNode);
        hoeheEdit->setValue((bbv.bbox.yMax() - bbv.bbox.yMin()) * info->lastScaleY);
        breiteEdit->setValue((bbv.bbox.xMax() - bbv.bbox.xMin()) * info->lastScaleX);
        tiefeEdit->setValue((bbv.bbox.zMax() - bbv.bbox.zMin()) * info->lastScaleZ);
    }
}

void Move::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == movex)
    {
        allowX->setState(movex->getState());
    }
    if (menuItem == movey)
    {
        allowY->setState(movey->getState());
    }
    if (menuItem == movez)
    {
        allowZ->setState(movez->getState());
    }
    if (menuItem == moveh)
    {
        allowH->setState(moveh->getState());
    }
    if (menuItem == movep)
    {
        allowP->setState(movep->getState());
    }
    if (menuItem == mover)
    {
        allowR->setState(mover->getState());
    }
    if (menuItem == parentItem)
    {
        if (level < numLevels - 1)
            level++;
        //selectLevel();
    }
    else if (menuItem == childItem)
    {
        if (level > 0)
            level--;
        //selectLevel();
    }
    else if (menuItem == undoItem)
    {
        undo();
    }
    else if (menuItem == redoItem)
    {
        redo();
    }
    else if (menuItem == explicitItem)
    {
        explicitMode = !explicitItem->getState();
        explicitTUIItem->setState(explicitItem->getState());
    }
    else if (menuItem == moveToggle)
    {
        if (moveToggle->getState())
            coIntersection::instance()->isectAllNodes(true);
        else
            coIntersection::instance()->isectAllNodes(false);

        moveEnabled->setState(moveToggle->getState());
    }
    else if (menuItem == scaleItem)
    {
        if (moveDCS.valid() && info)
        {
            osg::Matrix oldMat, netMat;
            float newScale = scaleItem->getValue();
            //ScaleSlider->setValue(newScale);
            //ScaleField->setValue(newScale);

            oldMat = moveDCS->getMatrix();
            osg::Vec3 v1 = osg::Vec3(oldMat(0, 0), oldMat(0, 1), oldMat(0, 2));
            osg::Vec3 v2 = osg::Vec3(oldMat(1, 0), oldMat(1, 1), oldMat(1, 2));
            osg::Vec3 v3 = osg::Vec3(oldMat(2, 0), oldMat(2, 1), oldMat(2, 2));
            float s1, s2, s3;
            float os1, os2, os3;
            s1 = v1.length();
            s2 = v2.length();
            s3 = v3.length();
            v1 = osg::Vec3(info->initialMat(0, 0), info->initialMat(0, 1), info->initialMat(0, 2));
            v2 = osg::Vec3(info->initialMat(1, 0), info->initialMat(1, 1), info->initialMat(1, 2));
            v3 = osg::Vec3(info->initialMat(2, 0), info->initialMat(2, 1), info->initialMat(2, 2));
            os1 = v1.length();
            os2 = v2.length();
            os3 = v3.length();
            oldMat.scale((os1 / s1) * newScale, (os2 / s2) * newScale, (os3 / s3) * newScale);
            netMat.preMult(oldMat);

            moveDCS->setMatrix(netMat);
            updateScale();
        }
    }
    else if (menuItem == resetItem)
    {
        osg::Matrix ident;
        ident.makeIdentity();
        if (moveDCS.get())
        {
            if (info)
            {
                moveDCS->setMatrix(info->initialMat);
                info->lastScaleX = 1;
                info->lastScaleY = 1;
                info->lastScaleZ = 1;
                scaleItem->setValue(info->lastScaleY);
                ScaleSlider->setValue(info->lastScaleY);
            }
            else
            {
                moveDCS->setMatrix(ident);
            }
            allowMove = false;
            updateScale();
            //cerr << "Reset " << endl;
        }
    }
    //cerr << "Level: " << level << endl;
}

void Move::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == allowX)
    {
        movex->setState(allowX->getState());
    }
    else if (tUIItem == allowY)
    {
        movey->setState(allowY->getState());
    }
    else if (tUIItem == allowZ)
    {
        movez->setState(allowZ->getState());
    }
    else if (tUIItem == allowH)
    {
        moveh->setState(allowH->getState());
    }
    else if (tUIItem == allowP)
    {
        movep->setState(allowP->getState());
    }
    else if (tUIItem == allowR)
    {
        mover->setState(allowR->getState());
    }
    else if (tUIItem == explicitTUIItem)
    {
        explicitMode = !explicitTUIItem->getState();
        explicitItem->setState(explicitTUIItem->getState());
    }
    else if (tUIItem == moveEnabled)
    {
        moveToggle->setState(moveEnabled->getState());
    }
    else if (tUIItem == hoeheEdit)
    {
        if (info && selectedNode)
        {
            BBoxVisitor bbv;
            bbv.apply(*selectedNode);
            float hoehe = (bbv.bbox.yMax() - bbv.bbox.yMin()) * info->lastScaleY;

            osg::Matrix netMat;

            netMat = moveDCS->getMatrix();

            float s = hoeheEdit->getValue() / hoehe;

            if (aspectRatio->getState())
            {
                netMat.preMult(osg::Matrix::scale(s, s, s));

                moveDCS->setMatrix(netMat);

                info->lastScaleX *= s;
                info->lastScaleY *= s;
                ;
                info->lastScaleZ *= s;
            }
            else
            {
                netMat.preMult(osg::Matrix::scale(1, s, 1));

                moveDCS->setMatrix(netMat);

                info->lastScaleY *= s;
            }
            updateScale();
        }
    }
    else if (tUIItem == breiteEdit)
    {
        if (info && selectedNode)
        {
            BBoxVisitor bbv;
            bbv.apply(*selectedNode);
            float breite = (bbv.bbox.xMax() - bbv.bbox.xMin()) * info->lastScaleX;

            osg::Matrix netMat;

            netMat = moveDCS->getMatrix();

            float s = breiteEdit->getValue() / breite;

            if (aspectRatio->getState())
            {
                netMat.preMult(osg::Matrix::scale(s, s, s));

                moveDCS->setMatrix(netMat);

                info->lastScaleX *= s;
                info->lastScaleY *= s;
                ;
                info->lastScaleZ *= s;
            }
            else
            {
                netMat.preMult(osg::Matrix::scale(s, 1, 1));

                moveDCS->setMatrix(netMat);

                info->lastScaleX *= s;
            }
            updateScale();
        }
    }
    else if (tUIItem == tiefeEdit)
    {
        if (info && selectedNode)
        {
            BBoxVisitor bbv;
            bbv.apply(*selectedNode);
            float tiefe = (bbv.bbox.zMax() - bbv.bbox.zMin()) * info->lastScaleZ;

            osg::Matrix netMat;

            netMat = moveDCS->getMatrix();

            float s = tiefeEdit->getValue() / tiefe;

            if (aspectRatio->getState())
            {
                netMat.preMult(osg::Matrix::scale(s, s, s));

                moveDCS->setMatrix(netMat);

                info->lastScaleX *= s;
                info->lastScaleY *= s;
                ;
                info->lastScaleZ *= s;
            }
            else
            {
                netMat.preMult(osg::Matrix::scale(1, 1, s));

                moveDCS->setMatrix(netMat);

                info->lastScaleZ *= s;
            }
            updateScale();
        }
    }
    else if (tUIItem == ScaleSlider)
    {
        if (moveDCS.get() && info)
        {
            osg::Matrix netMat;
            float newScale = ScaleSlider->getValue();
            scaleItem->setValue(newScale);

            netMat = moveDCS->getMatrix();

            float s = newScale / info->lastScaleY;
            netMat.preMult(osg::Matrix::scale(s, s, s));

            moveDCS->setMatrix(netMat);

            info->lastScaleX *= s;
            info->lastScaleY = newScale;
            info->lastScaleZ *= s;
            updateScale();
        }
    }
    //cerr << "Level: " << level << endl;
}

void Move::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == Parent)
    {
        if (level < numLevels - 1)
            level++;
        //selectLevel();
    }
    else if (tUIItem == Child)
    {
        if (level > 0)
            level--;
        //selectLevel();
    }
    else if (tUIItem == Undo)
    {
        undo();
    }
    else if (tUIItem == Redo)
    {
        redo();
    }
    else if (tUIItem == Reset)
    {
        osg::Matrix ident;
        ident.makeIdentity();
        if (moveDCS.get())
        {
            if (info)
            {
                moveDCS->setMatrix(info->initialMat);
                info->lastScaleX = 1;
                info->lastScaleY = 1;
                info->lastScaleZ = 1;
                scaleItem->setValue(info->lastScaleY);
                ScaleSlider->setValue(info->lastScaleY);
            }
            else
            {
                moveDCS->setMatrix(ident);
            }
            allowMove = false;
            updateScale();
            //cerr << "Reset " << endl;
        }
    }
    //cerr << "Level: " << level << endl;
}

osg::Node *Move::createBBox()
{
    osg::Geode *geodebox = new osg::Geode;
    osg::ShapeDrawable *sd;
    sd = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0.505, 0.504, 0.505), 1.01));
    osg::StateSet *ss = sd->getOrCreateStateSet();

    osg::PolygonMode *polymode = new osg::PolygonMode;
    polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
    ss->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    geodebox->addDrawable(sd);
    geodebox->setNodeMask(geodebox->getNodeMask() & ~Isect::Intersection);

    osg::Material *boxmat = new osg::Material;
    boxmat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    boxmat->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1.0));
    boxmat->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1.0));
    boxmat->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    boxmat->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1.0));
    boxmat->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    ss->setAttributeAndModes(boxmat, osg::StateAttribute::ON);
    return (geodebox);
}

void Move::addUndo(osg::Matrix &mat, osg::MatrixTransform *dcs)
{
    if (dcs == NULL)
        return;
    if (readPos != writePos)
    {
        decwrite();
        if ((undoDCS[writePos] == dcs) && (undoMat[writePos] == mat))
        {
            incwrite();
            return;
        }
        incwrite();
    }
    //cerr << "addUndo" << endl;
    undoDCS[writePos] = dcs;
    undoMat[writePos] = mat;
    //cerr << "a" << writePos << endl;
    incwrite();
    maxWritePos = writePos;
    if (readPos == writePos)
        incread();
}

void Move::incread()
{
    readPos++;
    if (readPos == MAX_UNDOS)
        readPos = 0;
}

void Move::decread()
{
    readPos--;
    if (readPos < 0)
        readPos = MAX_UNDOS - 1;
}

void Move::incwrite()
{
    writePos++;
    if (writePos == MAX_UNDOS)
        writePos = 0;
}

void Move::decwrite()
{
    writePos--;
    if (writePos < 0)
        writePos = MAX_UNDOS - 1;
}

void Move::undo()
{
    if (readPos == writePos)
        return;
    if (maxWritePos == writePos)
        decwrite();
    if (readPos == writePos)
        return;
    //cerr << "undo" << endl;
    decwrite();
    //cerr << "r" << writePos << endl;
    osg::Matrix mat;
    mat = undoDCS[writePos]->getMatrix();
    if (mat == undoMat[writePos])
    {
        undo();
    }
    else
    {
        undoDCS[writePos]->setMatrix(undoMat[writePos]);
    }
}

void Move::redo()
{
    if (writePos == maxWritePos)
        return;
    //cerr << "redo" << endl;
    incwrite();
    //cerr << "r" << writePos << endl;
    if (writePos == maxWritePos)
        return;
    //cerr << "r" << writePos << endl;

    osg::Matrix mat;
    mat = undoDCS[writePos]->getMatrix();
    if (mat == undoMat[writePos])
    {
        redo();
    }
    else
    {
        undoDCS[writePos]->setMatrix(undoMat[writePos]);
    }
}

COVERPLUGIN(Move)
