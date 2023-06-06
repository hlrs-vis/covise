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
    _initialMat.makeIdentity();
    _originalDCS = false;
    _lastScaleX = 1;
    _lastScaleY = 1;
    _lastScaleZ = 1;
}

MoveInfo::~MoveInfo()
{
}

Move::Move() 
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("Gizmo_UI", cover->ui)
{
}

bool Move::init()
{
    //fprintf(stderr,"Move::Move\n");
    level = 0;
    numLevels = 0;
    _info = NULL;
    _moveSelection = false;
    _selectedNode = NULL;
    _explicitMode = coCoviseConfig::isOn("COVER.Plugin.Move.Explicit", true);
    readPos = writePos = maxWritePos = 0;
    // create the text
    osg::Vec4 fgcolor(0, 1, 0, 1);
    osg::Vec4 bgcolor(0.5, 0.5, 0.5, 0.5);
    float lineLen = 0.04 * cover->getSceneSize();
    float fontSize = 0.02 * cover->getSceneSize();
    _label = new coVRLabel("test", fontSize, lineLen, fgcolor, bgcolor);
    _label->hide();

    // setup menu
    _UIgizmoMenu.reset(new ui::Menu("Gizmo", this));

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
        _explicitMode = !state;
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
    });

    _UIchild->setCallback([this]() {
        if (level > 0)
            level--;
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
            if (_info)
            {
                moveDCS->setMatrix(_info->_initialMat);
                _gizmo->updateTransform(_info->_initialMat);

                _info->_lastScaleX = 1;
                _info->_lastScaleY = 1;
                _info->_lastScaleZ = 1;
                //scaleItem->setValue(_info->_lastScaleY);
                //ScaleSlider->setValue(_info->_lastScaleY);
            }
            else
            {
                moveDCS->setMatrix(ident);
                _gizmo->updateTransform(ident);
            }
            updateScale();
            //cerr << "Reset " << endl;
        }
    });

    boundingBoxNode = new osg::MatrixTransform();
    boundingBoxNode->addChild(createBBox());
    interactionA = new coTrackerButtonInteraction(coInteraction::ButtonA, "Move", coInteraction::Menu);
    _candidateNode = NULL;
    _oldNode = NULL;
    _selectedNodesParent = NULL;
    coVRSelectionManager::instance()->addListener(this);

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

    delete _label;
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
        _selectedNodesParent = dynamic_cast<osg::Group *>(coVRSelectionManager::validPath(path));
        tb >> path;
        _selectedNode = coVRSelectionManager::validPath(path);
        getMoveDCS();

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                tb >> m(i, j);
        moveDCS->setMatrix(m);
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
        _selectedNode = (*nodeIter).get();
        if (_selectedNode == nullptr)
        {
            fprintf(stderr, "deselect\n");
        }
        _selectedNodesParent = (*parentIter).get();
       /* if (name)
            moveObjectName->setLabel(name);
        else
            moveObjectName->setLabel("NoName");
        */if (_info)
        {
            //scaleItem->setValue(_info->_lastScaleY);
            //ScaleSlider->setValue(_info->_lastScaleY);
        }
        updateScale();
    }
    return true;
}

bool Move::pickedObjChanged()
{
    return true;
}

void Move::doMove()
{
    osg::Matrix newDCSMat;
    newDCSMat =  _gizmo->getMoveMatrix_o()*_startMoveDCSMat;
    
    TokenBuffer tb;
    std::string path = coVRSelectionManager::generatePath(_selectedNodesParent);
    tb << path;
    path = coVRSelectionManager::generatePath(_selectedNode);
    tb << path;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            tb << newDCSMat(i, j);
			

    cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                       PluginMessageTypes::MoveMoveNode, tb.getData().length(), tb.getData().data());
    
    vrui::vruiUserData  *_info = OSGVruiUserDataCollection::getUserData(_selectedNode, "RevitInfo");
    if(_info!=NULL)
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
    bool isGizmoNode = _gizmo->isIntersected();

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

    else if(!isGizmoNode && _gizmo->getState() != coInteraction::Active) // check for nodes if gizmo is not active
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
            if (node && _UImove->state() && (node != _oldNode) && (node != _selectedNode)) // Select a node
            {
                selectNode(node, intersectedNodePath, isObject, notNeededAtThisPlace,notNeeded);
                if(isObject)
                    _candidateNode = node;
                else
                    _candidateNode = NULL;
            }
            if(node == NULL)
                _candidateNode = NULL;
            
            _oldNode = node;
        }

        //**********************************************************    Register interactionA *******************************************************
        if (node && _UImove->state() && ((node == _candidateNode) || (node == _selectedNode))) //if we point towards a selected or candidate Node: necessarry to select the object
		{ 
		     if (!interactionA->isRegistered())
             {
                 std::cout <<"register interaction A" <<std::endl;
			     coInteractionManager::the()->registerInteraction(interactionA);
             }	
        }
        else if(_gizmoActive && !isGizmoNode && !interactionA->isRegistered()) // necessary to unselect objects if gizmo is active, but interactionA was unregistered
        {
                 std::cout <<"register interaction A" <<std::endl;
			     coInteractionManager::the()->registerInteraction(interactionA);
        }    
        
        //********************************************************** Select or Unselect ***********************************************************************************
        if(interactionA->wasStarted() )
        {
            //Deselect
            if( _gizmoActive && 
            (!isGizmoNode || ( !node || ((node != _candidateNode) && (node != _selectedNode))))
            )
            {
                std::cout<<"Deselection started"<<std::endl;
                deactivateGizmo();
                _oldNode = NULL;                                                                                                     
                node = NULL;                                                    
                _candidateNode = NULL;                                                   
                _selectedNode = NULL;                                                    
                if (_moveSelection)                                                  
                {                                                   
                    coVRSelectionManager::instance()->clearSelection();                                                 
                    _moveSelection = false;                                                  
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
                    if(_selectedNode)
                    {
                        osg::Matrix gizmoStartMat = calcStartMatrix();
                        //gizmoStartMat.makeScale(osg::Vec3(1,1,1));
                        activateGizmo(gizmoStartMat);
                    }
                }
            }
        } 
    }
    
    // if we don't point to a node or point to a non candidate or point to gizmo Node and gizmo is not active then unregister!
    if( isGizmoNode && interactionA->isRegistered() ||
        (
            !_gizmoActive &&
            (!node || (((node != _candidateNode) && (node != _selectedNode)) && interactionA->isRegistered()))
        )
      )
    {   
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            std::cout<<"unregister InteractionA"<<std::endl;
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
    }
}

osg::Matrix Move::calcStartMatrix()
{
    
    osg::Matrix dcsMat,startBaseMat,startCompleteMat;
    startBaseMat.makeIdentity();

    getMoveDCS();
    // start of interaction (button press)
    osg::Node *currentNode = NULL;
    if (moveDCS && moveDCS->getNumParents() > 0)
        currentNode = moveDCS->getParent(0);

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

    // remove Scale for gizmo Matrix
    osg::Vec3 decTrans, decScale;
    osg::Quat decRot,decSo;
    _startMoveDCSMat.decompose(decTrans,decRot,decScale,decSo);   
    return  osg::Matrix::rotate(decRot) * osg::Matrix::translate(decTrans) * startBaseMat * cover->getInvBaseMat();

    //return startCompleteMat* cover->getInvBaseMat();
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
        if (_explicitMode)//-------------------------------------------------------
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
        if (currentNode == _selectedNode)
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
        _info = (MoveInfo *)OSGVruiUserDataCollection::getUserData(currentNode, "MoveInfo");
        if (_info)
        {
            //scaleItem->setValue(_info->_lastScaleY);
            //ScaleSlider->setValue(_info->_lastScaleY);
        }
        if (_info && !_info->_originalDCS)
        {
        }
        else
        {
            _nodes[numLevels] = currentNode;
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
        if ((*iter) == _nodes[level])
        {
            if (iter != intersectedNodePath.begin())
                iter--;
            parent = (*iter)->asGroup();
            while (parent && coVRSelectionManager::isHelperNode(parent))
            {
                iter--;
                parent = (*iter)->asGroup();
            }
            _selectedNode = _nodes[level];
            if (parent)
            {
                coVRSelectionManager::instance()->addSelection(parent, _selectedNode);
                coVRSelectionManager::instance()->pickedObjChanged();
                _moveSelection = true;
            }
            else
            {
                cerr << "parent not found" << endl;
            }
        }
    }
}

void Move::showOrhideName(osg::Node *node)
{
    if (_UIdisplayNames->state())
    {
        if (node && !node->getName().empty())
        {
            _label->setString(node->getName().c_str());
            _label->setPosition(cover->getIntersectionHitPointWorld());
            _label->show();
        }
        else
            _label->hide();
    }
    else
            _label->hide();
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

    osg::Group *selectionHelperNode = coVRSelectionManager::getHelperNode(_selectedNodesParent, _selectedNode, coVRSelectionManager::MOVE);
    moveDCS = dynamic_cast<osg::MatrixTransform *>(selectionHelperNode);
    //cerr << "moveDCS=" << moveDCS.get() << endl;
    if (moveDCS.get() == NULL)
    {
        moveDCS = new osg::MatrixTransform();
        //cerr << "new moveDCS" << moveDCS.get() << endl;
        _info = new MoveInfo();
        OSGVruiUserDataCollection::setUserData(moveDCS.get(), "MoveInfo", _info);
        coVRSelectionManager::insertHelperNode(_selectedNodesParent, _selectedNode, moveDCS.get(), coVRSelectionManager::MOVE);
    }

    _info = (MoveInfo *)OSGVruiUserDataCollection::getUserData(moveDCS.get(), "MoveInfo");
    if (_info)
    {
        //scaleItem->setValue(_info->_lastScaleY);
        //ScaleSlider->setValue(_info->_lastScaleY);
    }
}

void Move::updateScale()
{
    if (_info && _selectedNode)
    {
        BBoxVisitor bbv;
        bbv.apply(*_selectedNode);
        //hoeheEdit->setValue((bbv.bbox.yMax() - bbv.bbox.yMin()) * _info->_lastScaleY);
        //breiteEdit->setValue((bbv.bbox.xMax() - bbv.bbox.xMin()) * _info->_lastScaleX);
        //tiefeEdit->setValue((bbv.bbox.zMax() - bbv.bbox.zMin()) * _info->_lastScaleZ);
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
        _gizmo->updateTransform(calcStartMatrix());

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
