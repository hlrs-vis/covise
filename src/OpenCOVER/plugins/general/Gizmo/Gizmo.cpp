/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coIntersection.h>
#include <cover/RenderObject.h>
#include <cover/coInteractor.h>

#include <config/CoviseConfig.h>

#include "Gizmo.h"

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
#include <osg/io_utils>


#include <cover/coVRLabel.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <cover/coVRTui.h>
#include <OpenVRUI/osg/OSGVruiUserDataCollection.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <PluginUtil/PluginMessageTypes.h>
#include <net/tokenbuffer.h>

using covise::coCoviseConfig;
using covise::TokenBuffer;

typedef osg::fast_back_stack<osg::ref_ptr<osg::RefMatrix> > MatrixStack;

// has to match names from Transform module
const char *p_type_ = "Transform";
const char *p_scale_type_ = "scale_type";
const char *p_scale_scalar_ = "scaling_factor";
const char *p_scale_vertex_ = "new_origin";
const char *p_trans_vertex_ = "vector_of_translation";
const char *p_rotate_normal_ = "axis_of_rotation";
const char *p_rotate_vertex_ = "one_point_on_the_axis";
const char *p_rotate_scalar_ = "angle_of_rotation";

class BBoxVisitor : public osg::NodeVisitor
{
public:
    osg::BoundingBox bbox;
    osg::ref_ptr<osg::StateSet> _selectedHl;

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
        osg::PolygonMode *polymode = new osg::PolygonMode();
    polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);

    osg::Material *selMaterial = new osg::Material();
    selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(1.0, 0.3, 0.0, 0.2f));
    selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(1.0, 0.3, 0.0, 0.2f));
    selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 0.2f));
    selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
    selMaterial->setTransparency(osg::Material::FRONT_AND_BACK,0.2);

    selMaterial->setColorMode(osg::Material::OFF);  
    _selectedHl= new osg::StateSet();
    _selectedHl->setAttribute(selMaterial, osg::StateAttribute::ON | osg::StateAttribute::PROTECTED);

    _selectedHl->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);

        node.setStateSet(_selectedHl.get());
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

Move::Move() : ui::Owner("Gizmo_UI", cover->ui)
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
    moveTransformMode = false;
    printMode = coCoviseConfig::isOn("COVER.Plugin.Move.Print", false);
    readPos = writePos = maxWritePos = 0;
    // create the text
    osg::Vec4 fgcolor(0, 1, 0, 1);
    osg::Vec4 bgcolor(0.5, 0.5, 0.5, 0.5);
    float lineLen = 0.04 * cover->getSceneSize();
    float fontSize = 0.02 * cover->getSceneSize();
    label = new coVRLabel("test", fontSize, lineLen, fgcolor, bgcolor);
    label->hide();

    // setup menu
    _UIgizmoMenu.reset(new ui::Menu("NewGizmoMenu", this));

    _UImove.reset(new ui::Button(_UIgizmoMenu.get(), "Move"));
    _UImoveAll.reset(new ui::Button(_UIgizmoMenu.get(), "MoveAll"));
    _UItranslate.reset(new ui::Button(_UIgizmoMenu.get(), "Translate"));
    _UIrotate.reset(new ui::Button(_UIgizmoMenu.get(), "Rotate"));
    _UIscale.reset(new ui::Button(_UIgizmoMenu.get(), "Scale"));
    _UIdisplayNames.reset(new ui::Button(_UIgizmoMenu.get(), "DisplayNames"));
    _UIlocalCoords.reset(new ui::Button(_UIgizmoMenu.get(), "localCoords"));

    _UIparent.reset(new ui::Action(_UIgizmoMenu.get(), "Parent"));
    _UIchild.reset(new ui::Action(_UIgizmoMenu.get(), "Child"));
    _UIundo.reset(new ui::Action(_UIgizmoMenu.get(), "Undo"));
    _UIredo.reset(new ui::Action(_UIgizmoMenu.get(), "Redo"));
    _UIreset.reset(new ui::Action(_UIgizmoMenu.get(), "Reset"));

    _UIscaleFactor.reset(new ui::Slider(_UIgizmoMenu.get(), "Scalefactor"));

    _UImoveAll->setText("Move All");
    _UIdisplayNames->setText("Display Names");
    _UIlocalCoords->setText("Local Coords");
    _UIscaleFactor->setText("Scale Factor");

    
    _UImove->setState(false);
    _UImove->setCallback([this](bool state) {
        if (state)
            coIntersection::instance()->isectAllNodes(true);
        else
            coIntersection::instance()->isectAllNodes(false);
    });

    _UImoveAll->setState(false);
    _UImoveAll->setCallback([this](bool state) {
        explicitMode = !state;
    });

    _UItranslate->setState(true);
    _UItranslate->setCallback([this](bool state){

        _gizmo->setGizmoTypes(state,_UIrotate->state(),_UIscale->state());
    });
    _UIrotate->setState(true);
    _UIrotate->setCallback([this](bool state){

        _gizmo->setGizmoTypes(_UItranslate->state(),state,_UIscale->state());
    });
    _UIscale->setState(false);
    _UIscale->setCallback([this](bool state){
        
        _gizmo->setGizmoTypes(_UItranslate->state(),_UIrotate->state(),state);
    });
    // _displayNames->setCallback([]{});
    // _localCoords->setCallback([]{});

    _UIparent->setCallback([this]() {
        if (level < numLevels - 1)
            level++;
        //selectLevel();
    });

    _UIchild->setCallback([this]() {
        if (level > 0)
            level--;
        //selectLevel();
    });

    _UIundo->setCallback([this]() {
        undo();
    });

    _UIredo->setCallback([this]() {
        redo();
    });

    _UIreset->setCallback([this]() {
        osg::Matrix ident;
        ident.makeIdentity();
        if (moveDCS.get())
        {
            if (info)
            {
                moveDCS->setMatrix(info->initialMat);
                _gizmo->updateTransform(info->initialMat);

                info->lastScaleX = 1;
                info->lastScaleY = 1;
                info->lastScaleZ = 1;
                //scaleItem->setValue(info->lastScaleY);
                //ScaleSlider->setValue(info->lastScaleY);
            }
            else
            {
                moveDCS->setMatrix(ident);
                _gizmo->updateTransform(ident);
            }
            allowMove = false;
            updateScale();
            //cerr << "Reset " << endl;
        }
    });

    boundingBoxNode = new osg::MatrixTransform();
    boundingBoxNode->addChild(createBBox());
    interactionA = new coTrackerButtonInteraction(coInteraction::ButtonA, "Move", coInteraction::Menu);
    candidateNode = NULL;
    oldNode = NULL;
    selectedNodesParent = NULL;
    coVRSelectionManager::instance()->addListener(this);
    startTime=0;

    float interSize = cover->getSceneSize() / 50 ;
    osg::Matrix test;
    _gizmo.reset(new coVR3DGizmo(coVR3DGizmo::GIZMO_TYPE::ROTATE,_UItranslate->state(),_UIrotate->state(),_UIscale->state(), test, interSize, vrui::coInteraction::ButtonA, "hand", "Gizmo", vrui::coInteraction::Medium));
    _gizmo->hide();

    return true;
}

// this is called if the plugin is removed at runtime
Move::~Move()
{
    for (RoInteractorMap::iterator it = roInteractorMap.begin();
            it != roInteractorMap.end();
            ++it)
    {
        it->second->decRefCount();
    }
    roInteractorMap.clear();
    nodeRoMap.clear();

    delete label;
    // we probably have to delete all move infos...

    delete interactionA;

    while (boundingBoxNode->getNumParents())
        boundingBoxNode->getParent(0)->removeChild(boundingBoxNode.get());

    coVRSelectionManager::instance()->removeListener(this);
    fprintf(stderr, "Move::~Move\n");
}

void Move::message(int toWhom, int type, int len, const void *buf)
{
    if (type == PluginMessageTypes::MoveAddMoveNode)
    {
    }
    else if (type == PluginMessageTypes::MoveMoveNode)
    {
        osg::Matrix m;
        std::string path;
        TokenBuffer tb{ covise::DataHandle{(char*)buf, len, false} };
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
    std::cout <<"selection changed"<<std::endl;
    //coVRMSController::instance()->syncInt(3000);
    std::list<osg::ref_ptr<osg::Node> > selectedNodeList = coVRSelectionManager::instance()->getSelectionList();
    std::list<osg::ref_ptr<osg::Group> > selectedParentList = coVRSelectionManager::instance()->getSelectedParentList();

    std::list<osg::ref_ptr<osg::Node> >::iterator nodeIter = selectedNodeList.end();
    std::list<osg::ref_ptr<osg::Group> >::iterator parentIter = selectedParentList.end();
    if (selectedNodeList.size() == 0)
    {
        moveDCS = NULL;
        //moveObjectLabel->setLabel("None");
    //coVRMSController::instance()->syncInt(3001);
    }
    else
    {
    //coVRMSController::instance()->syncInt(3002);
        nodeIter--;
        parentIter--;
        const char *name = (*nodeIter)->getName().c_str();
        selectedNode = (*nodeIter).get();
        if (selectedNode == nullptr)
        {
            fprintf(stderr, "deselect\n");
        }
        selectedNodesParent = (*parentIter).get();
       /* if (name)
            moveObjectName->setLabel(name);
        else
            moveObjectName->setLabel("NoName");
        */if (info)
        {
            //scaleItem->setValue(info->lastScaleY);
            //ScaleSlider->setValue(info->lastScaleY);
        }
        updateScale();
    }
    return true;
}

bool Move::pickedObjChanged()
{
    return true;
}

/*osg::Matrix Move::scaleNode()
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
*/
void Move::doMove()
{
    osg::Matrix newDCSMat;
    // TODO: if scale Gizmo -> then...
    newDCSMat =  _gizmo->getMoveMatrix_o()*_startMoveDCSMat;
    
    TokenBuffer tb;
    std::string path = coVRSelectionManager::generatePath(selectedNodesParent);
    tb << path;
    path = coVRSelectionManager::generatePath(selectedNode);
    tb << path;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            tb << newDCSMat(i, j);
			

    cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                       PluginMessageTypes::MoveMoveNode, tb.getData().length(), tb.getData().data());
    
    vrui::vruiUserData  *info = OSGVruiUserDataCollection::getUserData(selectedNode, "RevitInfo");
    if(info!=NULL)
    {
        cover->sendMessage(this, "Revit",
            PluginMessageTypes::MoveMoveNode, tb.getData().length(), tb.getData().data());
    }

}

void Move::preFrame()
{
    _gizmo->preFrame();
    osg::Node *node;
    node = cover->getIntersectedNode();
    const osg::NodePath &intersectedNodePath = cover->getIntersectedNodePath();
    _isGizmoNode = _gizmo->isIntersected();

    if(_gizmo->wasStarted())
    {
        _startMoveDCSMat = moveDCS->getMatrix();
        addUndo(_startMoveDCSMat, moveDCS.get());
    }
    else if (_gizmo->wasStopped()) 
    {
        osg::Matrix stopMatrix = moveDCS->getMatrix();
        addUndo(stopMatrix,moveDCS.get());
    }
    else if(_gizmo->getState() == coInteraction::Active) // do the movement
        doMove();

    else if(!_isGizmoNode && _gizmo->getState() != coInteraction::Active) // check for nodes if gizmo is not active
    {
        bool is_SceneNode{false};
        if(node)
            is_SceneNode = isSceneNode(node, intersectedNodePath);
        
        if(is_SceneNode && !coVRSelectionManager::isHelperNode(node)) // Scene node but no helperNode -> do what ???
        {
            //std::cout<<"Scene node but no helperNode"<<std::endl;
            showOrhideName(node);
            bool isObject{false};
            bool notNeededAtThisPlace{false};
            bool notNeeded{false};
            if (node && _UImove->state() && (node != oldNode) && (node != selectedNode)) // Select a node
            {
                selectNode(node, intersectedNodePath, isObject, notNeededAtThisPlace,notNeeded);
                if(isObject)
                    candidateNode = node;
                else
                    candidateNode = NULL;
            }
            if(node == NULL)
                candidateNode = NULL;
            
            oldNode = node;
        }

        //**********************************************************    Register interactionA *******************************************************
        if (node && _UImove->state() && ((node == candidateNode) || (node == selectedNode))) //if we point towards a selected or candidate Node: necessarry to select the object
		{ 
		     if (!interactionA->isRegistered())
             {
                 std::cout <<"register interaction A" <<std::endl;
			     coInteractionManager::the()->registerInteraction(interactionA);
             }	
        }
        else if(_gizmoActive && !_isGizmoNode && !interactionA->isRegistered()) // necessary to unselect objects if gizmo is active, but interactionA was unregistered
        {
                 std::cout <<"register interaction A" <<std::endl;
			     coInteractionManager::the()->registerInteraction(interactionA);
        }    
        
        //********************************************************** Select or Unselect ***********************************************************************************
        if(interactionA->wasStarted() )
        {
            //Deselect
            if( _gizmoActive && 
            (!_isGizmoNode || ( !node || ((node != candidateNode) && (node != selectedNode))))
            )
            {
                std::cout<<"Deselection started"<<std::endl;
                deactivateGizmo();
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
            if(node) //Select
            {   
                std::cout<<"Selection started"<<std::endl;
                bool isObject = false;
                bool isNewObject = false;
                bool isAlreadySelected = false;
                selectNode(node, intersectedNodePath, isObject, isNewObject, isAlreadySelected);
                if(isNewObject)
                {                    
                    newObject(node, intersectedNodePath);//check exactly what happens !
                    // create Gizmo
                    // start of interaction (button press)
                    if(selectedNode)
                    {
                        _gizmoStartMat = calcStartMatrix();
                        //_gizmoStartMat.makeScale(osg::Vec3(1,1,1));
                        activateGizmo(_gizmoStartMat);
                    }
                }
            }
        } 
    }
    
    // if we don't point to a node or point to a non candidate or point to gizmo Node and gizmo is not active then unregister!
    if( _isGizmoNode && interactionA->isRegistered() ||
        (
            !_gizmoActive &&
            (!node || (((node != candidateNode) && (node != selectedNode)) && interactionA->isRegistered()))
        )
      )
    {   
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            std::cout<<"unregister InteractionA"<<std::endl;
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
    }
    //std::cout <<"doUndo: "<<doUndo <<" didMove: "<<didMove <<" allowMove: "<<allowMove<<std::endl; 
}

//remove global variable startCompleteMat
osg::Matrix Move::calcStartMatrix()
{
    
    osg::Matrix dcsMat;
    getMoveDCS();
    // start of interaction (button press)
    osg::Node *currentNode = NULL;
    if (moveDCS && moveDCS->getNumParents() > 0)
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
    _startMoveDCSMat = moveDCS->getMatrix();
    startCompleteMat = startBaseMat;
    startCompleteMat.preMult(_startMoveDCSMat);

    if (!invStartCompleteMat.invert(startCompleteMat))
        fprintf(stderr, "Move::inv startCompleteMat is singular\n");
    if (!invStartHandMat.invert(cover->getPointerMat()))
        fprintf(stderr, "Move::inv getPointerMat is singular\n");

    /*//remove Scale for gizmo Matrix:
    float newScale = 1.0;
    osg::Matrix oldMat,netMat;
    oldMat = startCompleteMat;
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
    return netMat * cover->getInvBaseMat();
    */
    return startCompleteMat* cover->getInvBaseMat();
}

void Move::selectNode(osg::Node* node, const osg::NodePath& intersectedNodePath, bool& isObject, bool& isNewObject, bool& isAlreadySelected) // check exactly what happens here !
{
    osg::Matrix mat;
    mat.makeIdentity();
    osg::Node* currentNode;
    currentNode = node;

    while(currentNode != NULL)
    {
        const char* nodeName = currentNode->getName().c_str();
        if (moveTransformMode)//-----------------------------------------------------this if block is missing in original Move plugin when it appears for the first time
        {
            NodeRoMap::iterator it = nodeRoMap.find(currentNode);
            if (it != nodeRoMap.end())
                isObject = true;

            if (currentNode == cover->getObjectsRoot())
                break;  
        }
        else if (explicitMode)//-------------------------------------------------------
        {
            // do not touch nodes underneeth NoMove nodes
            if (nodeName && (strncmp(nodeName, "DoMove", 6) == 0 || strncmp(nodeName, "Griff.", 6) == 0 || strncmp(nodeName, "Handle.", 7) == 0))
                isObject = true;
            
            if (currentNode == cover->getObjectsRoot())
            {
                isNewObject = true;
                break;
            }
        }
        else//-----------------------------------------------------------
        {
            if (currentNode == cover->getObjectsRoot())
            {
                isObject = true;
                isNewObject = true;
                break;
            }
        }
        // do not touch nodes underneeth NoMove nodes
        if (nodeName && strncmp(nodeName, "NoMove", 6) == 0)
        {
            isObject = false;
            isNewObject = false;
            break;
        }
        if (currentNode == selectedNode)
        {
            isObject = true; // already selected
            isNewObject = false;
            cerr << "already selected: " << level << endl; // maybe this is not the correct warning ?
            isAlreadySelected = true;
            break;
        }
        if (currentNode->getNumParents() > 0)
        {
            std::vector<osg::Node*>::const_iterator iter = intersectedNodePath.end();
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
}

void Move::newObject(osg::Node* node,const osg::NodePath& intersectedNodePath)
{
    osg::Node* currentNode = node; 
    numLevels = 0;

    while (currentNode != NULL)
    {
        if (currentNode == cover->getObjectsRoot())
            break;

        const char *nodeName = currentNode->getName().c_str();
        if (nodeName && (strncmp(nodeName, "Default", 7) == 0))
        {
            level = numLevels;
        }
        
        vrui::vruiUserData  *RevitInfo = OSGVruiUserDataCollection::getUserData(currentNode, "RevitInfo");
        if(RevitInfo!=NULL)
        {
            level = numLevels;
        }
        info = (MoveInfo *)OSGVruiUserDataCollection::getUserData(currentNode, "MoveInfo");
        if (info)
        {
            //scaleItem->setValue(info->lastScaleY);
            //ScaleSlider->setValue(info->lastScaleY);
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
    std::cout << "new object, clear selection"<< std::endl;

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
    
    allowMove = false;
    didMove = false;
}

void Move::showOrhideName(osg::Node *node)
{
    if (_UIdisplayNames->state())
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
}

bool Move::isSceneNode(osg::Node* node,const osg::NodePath& intersectedNodePath)const
{
    for (std::vector<osg::Node*>::const_iterator iter = intersectedNodePath.begin(); iter != intersectedNodePath.end(); ++iter)
    {
        if ((*iter) == cover->getObjectsRoot())
            return true;
    }
    return false;
}

// bool Move::isGizmoNode(osg::Node* node,const osg::NodePath& intersectedNodePath)const
// {
    // 
    //  for (std::vector<osg::Node*>::const_iterator iter = intersectedNodePath.begin(); iter != intersectedNodePath.end(); ++iter)
    //  {
        //  auto hitNode= _gizmo->getHitNode();
        //  if(hitNode)
        //  {
        //  
// 
        //  }
        //  if ((*iter) == _gizmo->getHitNode())
            //  return true;
    // 
    //  }
    //  return false;
// }

void Move::activateGizmo(const osg::Matrix& m)
{
    std::cout <<"activate gizmo"<<std::endl;

    _gizmo->updateTransform(m);
    _gizmo->enableIntersection();
    _gizmo->show();
    _gizmoActive = true;
}

void Move::deactivateGizmo()
{
    std::cout <<"deactivate gizmo"<<std::endl;
    _gizmo->disableIntersection();
    _gizmo->hide();
    _gizmoActive = false;
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
        //scaleItem->setValue(info->lastScaleY);
        //ScaleSlider->setValue(info->lastScaleY);
    }
}

void Move::updateScale()
{
    if (info && selectedNode)
    {
        BBoxVisitor bbv;
        bbv.apply(*selectedNode);
        //hoeheEdit->setValue((bbv.bbox.yMax() - bbv.bbox.yMin()) * info->lastScaleY);
        //breiteEdit->setValue((bbv.bbox.xMax() - bbv.bbox.xMin()) * info->lastScaleX);
        //tiefeEdit->setValue((bbv.bbox.zMax() - bbv.bbox.zMin()) * info->lastScaleZ);
    }
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

void Move::undo() // warum wird hier keine message an alle rausgeschickt ? funktioniert das im Kooperationsmodus ? 
{
    if (readPos == writePos)
        return;

    if (maxWritePos == writePos)
        decwrite();
    if (readPos == writePos)
    
        return;

    decwrite();
    osg::Matrix mat;
    mat = undoDCS[writePos]->getMatrix();
    if (mat == undoMat[writePos])
        undo();
    else
    {
        undoDCS[writePos]->setMatrix(undoMat[writePos]);
        _gizmo->updateTransform(calcStartMatrix());
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
        _gizmo->updateTransform(undoDCS[writePos]->getMatrix());
    }
}

void Move::newInteractor(const RenderObject *container, coInteractor *inter)
{
    if (!container || !inter)
        return;
    if (strcmp(inter->getPluginName(), "Move") != 0)
        return;

    if (roInteractorMap.insert(std::make_pair(container, inter)).second)
    {
        inter->incRefCount();
    }
}

void Move::addNode(osg::Node *node, const RenderObject *ro)
{
    if (roInteractorMap.find(ro) != roInteractorMap.end())
    {
        nodeRoMap[node] = ro;
    }
}

void Move::removeNode(osg::Node *node, bool isGroup, osg::Node *realNode)
{
    NodeRoMap::iterator it = nodeRoMap.find(node);
    if (it != nodeRoMap.end())
    {
        RoInteractorMap::iterator rit = roInteractorMap.find(it->second);
        if (rit != roInteractorMap.end())
        {
            rit->second->decRefCount();
            roInteractorMap.erase(rit);
        }
        nodeRoMap.erase(it);
    }
}

COVERPLUGIN(Move)
