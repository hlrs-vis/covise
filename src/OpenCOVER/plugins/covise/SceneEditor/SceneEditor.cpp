/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SceneEditor.h"

#define REPORT(x)                                                        \
    std::cout << __FILE__ << ":" << __LINE__ << ": " << #x << std::endl; \
    x;

#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Geometry>
#include <osg/Image>
#include <osgDB/ReadFile>
#include <osgShadow/ShadowedScene>

#include <cover/coVRFileManager.h>
#include <cover/coVRCollaboration.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <OpenVRUI/coNavInteraction.h>
#include <config/CoviseConfig.h>

#include "SceneUtils.h"
#include "ErrorCodes.h"

#include "Settings.h"

#include "Events/PreFrameEvent.h"
#include "Events/StartMouseEvent.h"
#include "Events/StopMouseEvent.h"
#include "Events/DoMouseEvent.h"
#include "Events/DoubleClickEvent.h"
#include "Events/SetTransformAxisEvent.h"
#include "Events/MouseEnterEvent.h"
#include "Events/MouseExitEvent.h"
#include "Events/SelectEvent.h"
#include "Events/DeselectEvent.h"
#include "Events/MountEvent.h"
#include "Events/UnmountEvent.h"
#include "Events/GetCameraEvent.h"
#include "Events/SetSizeEvent.h"
#include "Events/SwitchVariantEvent.h"
#include "Events/SetAppearanceColorEvent.h"
#include "Events/MoveObjectEvent.h"
#include "Events/SettingsChangedEvent.h"
#include "Events/InitKinematicsStateEvent.h"

#include <grmsg/coGRKeyWordMsg.h>
#include <grmsg/coGRObjVisMsg.h>
#include <grmsg/coGRObjDelMsg.h>
#include <grmsg/coGRObjMovedMsg.h>
#include <grmsg/coGRObjGeometryMsg.h>
#include <grmsg/coGRObjAddChildMsg.h>
#include <grmsg/coGRObjSetVariantMsg.h>
#include <grmsg/coGRObjSetAppearanceMsg.h>
#include <grmsg/coGRObjKinematicsStateMsg.h>

SceneEditor *SceneEditor::plugin = NULL;

static opencover::FileHandler coxmlFileHandlers[] = {
    { SceneEditor::loadCoxmlUrl, SceneEditor::loadCoxml,
      SceneEditor::unloadCoxml, "coxml" }
};

SceneEditor::SceneEditor()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    _sceneObjectManager = SceneObjectManager::instance();
    bool exists;
    std::string resourceDirectory = covise::coCoviseConfig::getEntry("value",
                                                                     "vr-prepare.Coxml.ResourceDirectory", &exists);
    if (exists)
    {
        _sceneObjectManager->setResourceDirectory(resourceDirectory);
    }

    _mouseOverNode = NULL;
    _mouseOverSceneObject = NULL;
    _selectedSceneObject = NULL;

    interactionA = new vrui::coNavInteraction(vrui::coInteraction::ButtonA, "Selection",
                                              vrui::coInteraction::NavigationHigh);
    interactionC = new vrui::coNavInteraction(vrui::coInteraction::ButtonC, "Selection",
                                              vrui::coInteraction::NavigationHigh);
}

SceneEditor::~SceneEditor()
{
    //    coVRSelectionManager::instance()->removeListener(this);
    delete interactionA;
    delete interactionC;
    delete _sceneObjectManager;
}

double
to_double(std::string s)
{
    double d;
    std::istringstream iss(s);
    iss >> d;
    return d;
}

template <class T>
void
operator>>(const std::string &s, T &converted)
{
    std::istringstream iss(s);
    iss >> converted;
    if ((iss.rdstate() & std::istringstream::failbit) != 0)
    {
        std::cerr << "Error in conversion from string \"" << s << "\" to type "
                  << typeid(T).name() << std::endl;
    }
}

bool
SceneEditor::init()
{
    if (plugin)
        return false;

    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "\nSceneEditorPlugin::SceneEditorPlugin\n");

    SceneEditor::plugin = this;

    // register for coxml files
    opencover::coVRFileManager::instance()->registerFileHandler(&coxmlFileHandlers[0]);

    // add shadowed scene over objets root
    osgShadow::ShadowedScene *shadowedScene = new osgShadow::ShadowedScene();
    //shadowedScene->setReceivesShadowTraversalMask(opencover::Isect::Intersection);
    //shadowedScene->setCastsShadowTraversalMask(opencover::Isect::Intersection);
    //fprintf(stderr, "castShadow %x\n", shadowedScene->getCastsShadowTraversalMask());
    osg::Group *parent = opencover::cover->getObjectsRoot()->getParent(0);
    for (unsigned int i = 0; i < parent->getNumChildren(); i++)
        shadowedScene->addChild(parent->getChild(i));
    parent->removeChildren(0, parent->getNumChildren());
    parent->addChild(shadowedScene);
    shadowedScene->cleanSceneGraph();

    return true;
}

void
SceneEditor::preFrame()
{

    osg::Node *oldMouseOverNode = _mouseOverNode;
    SceneObject *oldMouseOverSceneObject = _mouseOverSceneObject;

    // get new sceneobject if nescessary (node changed)
    _mouseOverNode = opencover::cover->getIntersectedNode();
    if (oldMouseOverNode != _mouseOverNode)
    {
        _mouseOverSceneObject = _sceneObjectManager->findSceneObject(
            _mouseOverNode);
    }
    if (_mouseOverSceneObject && (_mouseOverSceneObject->getType() == SceneObjectTypes::ROOM))
    {
        _mouseOverSceneObject = NULL;
    }
    if (opencover::coVRNavigationManager::instance()->getMode() == opencover::coVRNavigationManager::Measure)
    {
        // no interaction while measuring
        _mouseOverSceneObject = NULL;
    }

    // send enter/exit events
    if (oldMouseOverSceneObject != _mouseOverSceneObject)
    {
        if (oldMouseOverSceneObject)
        {
            MouseExitEvent mee;
            oldMouseOverSceneObject->receiveEvent(&mee);
        }
        if (_mouseOverSceneObject)
        {
            MouseEnterEvent mee;
            _mouseOverSceneObject->receiveEvent(&mee);
        }
    }

    // register or unregister the interactions
    if (_mouseOverSceneObject)
    {
        if (!interactionA->isRegistered())
        {
            vrui::coInteractionManager::the()->registerInteraction(interactionA);
            vrui::coInteractionManager::the()->registerInteraction(interactionC);
        }
    }
    else
    {
        if (interactionA->isRegistered() && !interactionA->isRunning()
            && !interactionA->wasStopped() && !interactionC->isRunning()
            && !interactionC->wasStopped())
        {
            vrui::coInteractionManager::the()->unregisterInteraction(interactionA);
            vrui::coInteractionManager::the()->unregisterInteraction(interactionC);
        }
    }

    // send deselect event if not clicked on any object
    if (opencover::cover->getPointerButton()->wasPressed() && !interactionA->wasStarted()
        && !interactionC->wasStarted())
    {
        if ((_mouseOverSceneObject == NULL) && (_selectedSceneObject != NULL))
        {
            // deselect
            DeselectEvent de;
            _selectedSceneObject->receiveEvent(&de);
            _selectedSceneObject = NULL;
            // select room
            Room *room = _sceneObjectManager->getRoom();
            if (room)
            {
                SelectEvent se;
                room->receiveEvent(&se);
                _selectedSceneObject = room;
            }
        }
    }

    if (interactionA->wasStarted() || interactionC->wasStarted())
    {
        // select/deselect event
        if (_selectedSceneObject != _mouseOverSceneObject)
        {
            if (_selectedSceneObject)
            {
                DeselectEvent de;
                _selectedSceneObject->receiveEvent(&de);
            }
            if (_mouseOverSceneObject)
            {
                SelectEvent se;
                _mouseOverSceneObject->receiveEvent(&se);
            }
        }
        _selectedSceneObject = _mouseOverSceneObject;
        // start mouse event
        if (_selectedSceneObject)
        {
            SceneObject *top = SceneUtils::followFixedMountsToParent(
                _selectedSceneObject);
            // reset axis
            SetTransformAxisEvent stae;
            stae.resetTranslate();
            stae.resetRotate();
            top->receiveEvent(&stae);
            // start
            StartMouseEvent smte;
            if (interactionC->wasStarted())
            {
                smte.setMouseButton(MouseEvent::TYPE_BUTTON_C);
            }
            else
            {
                smte.setMouseButton(MouseEvent::TYPE_BUTTON_A);
            }
            top->receiveEvent(&smte);
        }
    }
    else if (interactionA->isRunning() || interactionC->isRunning())
    {
        // mouse move event
        if (_selectedSceneObject)
        {
            DoMouseEvent dmte;
            if (interactionC->isRunning())
            {
                dmte.setMouseButton(MouseEvent::TYPE_BUTTON_C);
            }
            else
            {
                dmte.setMouseButton(MouseEvent::TYPE_BUTTON_A);
            }
            SceneUtils::followFixedMountsToParent(_selectedSceneObject)->receiveEvent(&dmte);
        }
    }
    else if (interactionA->wasStopped() || interactionC->wasStopped())
    {
        // stop mouse event
        if (_selectedSceneObject)
        {
            StopMouseEvent smte;
            if (interactionC->wasStopped())
            {
                smte.setMouseButton(MouseEvent::TYPE_BUTTON_C);
            }
            else
            {
                smte.setMouseButton(MouseEvent::TYPE_BUTTON_A);
            }
            SceneUtils::followFixedMountsToParent(_selectedSceneObject)->receiveEvent(&smte);
        }
    }

    PreFrameEvent pfe;
    _sceneObjectManager->broadcastEvent(&pfe);
}

void
SceneEditor::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "\n--- Plugin SceneEditor coVRGuiToRenderMsg [%s]\n",
                msg.getString().c_str());

    if (msg.isValid())
    {
        using namespace grmsg;
        switch (msg.getType())
        {
        case coGRMsg::KEYWORD:
        {
            auto &keywordMsg = msg.as<coGRKeyWordMsg>();
            const char *keyword = keywordMsg.getKeyWord();
            if (strcmp(keyword, "toggleOperatingRange") == 0)
            {
                Settings::instance()->toggleOperatingRangeVisible();
                SettingsChangedEvent sce;
                _sceneObjectManager->broadcastEvent(&sce);
            }
            else if (strcmp(keyword, "toggleGrid") == 0)
            {
                Settings::instance()->toggleGridVisible();
                Room *room = plugin->_sceneObjectManager->getRoom();
                if (room)
                {
                    SettingsChangedEvent sce;
                    room->receiveEvent(&sce);
                }
            }
        }
        break;
        case coGRMsg::GEO_VISIBLE:
        {
            auto &geometryVisibleMsg = msg.as<coGRObjVisMsg>();
            const char *objectName = geometryVisibleMsg.getObjName();

            if (opencover::cover->debugLevel(3))
                fprintf(stderr,
                        "in SceneEditor grmsg::coGRMsg::GEO_VISIBLE object=%s visible=%d\n",
                        objectName, geometryVisibleMsg.isVisible());

            SceneObject *so = _sceneObjectManager->findSceneObject(objectName);
            if (so)
            {
                osg::Node *n = so->getRootNode();
                if (geometryVisibleMsg.isVisible())
                    n->setNodeMask(n->getNodeMask() | opencover::Isect::Visible);
                else
                    n->setNodeMask(n->getNodeMask() & ~opencover::Isect::Visible);
            }
        }
        break;
        case coGRMsg::DELETE_OBJECT:
        {
            auto &geometryDeleteMsg = msg.as<coGRObjDelMsg>();
            const char *objectName = geometryDeleteMsg.getObjName();

            if (opencover::cover->debugLevel(3))
                fprintf(stderr,
                        "in SceneEditor grmsg::coGRMsg::DELETE_OBJECT object=%s \n",
                        objectName);

            SceneObject *so = _sceneObjectManager->findSceneObject(objectName);
            if (so)
            {
                _sceneObjectManager->deletingSceneObject(so);
            }
        }
        break;
        case coGRMsg::OBJECT_TRANSFORMED:
        {
            auto &geometryMovedMsg = msg.as<coGRObjMovedMsg>();
            const char *objectName = geometryMovedMsg.getObjName();

            osg::Quat rot(geometryMovedMsg.getRotX(), geometryMovedMsg.getRotY(),
                          geometryMovedMsg.getRotZ(), geometryMovedMsg.getRotAngle());
            osg::Matrix mt, mr;
            mt.makeTranslate(geometryMovedMsg.getTransX(),
                             geometryMovedMsg.getTransY(), geometryMovedMsg.getTransZ());
            mr.makeRotate(rot);

            if (opencover::cover->debugLevel(3))
                fprintf(stderr,
                        "in SceneEditor grmsg::coGRMsg::OBJECT_TRANSFORMED object=%s \n",
                        objectName);

            SceneObject *so = _sceneObjectManager->findSceneObject(objectName);
            if (so)
            {
                so->setTransform(mt, mr);
            }
        }
        break;
        case coGRMsg::GEOMETRY_OBJECT:
        {
            auto &geometryMsg = msg.as<coGRObjGeometryMsg>();
            const char *objectName = geometryMsg.getObjName();

            SetSizeEvent sse;
            sse.setWidth(geometryMsg.getWidth());
            sse.setHeight(geometryMsg.getHeight());
            sse.setLength(geometryMsg.getLength());

            if (opencover::cover->debugLevel(3))
                fprintf(stderr,
                        "in SceneEditor grmsg::coGRMsg::OBJECT_GEOMETRY object=%s \n",
                        objectName);

            SceneObject *so = _sceneObjectManager->findSceneObject(objectName);
            if (so)
                so->receiveEvent(&sse);
        }
        break;
        case coGRMsg::ADD_CHILD_OBJECT:
        {
            auto &childMsg = msg.as<coGRObjAddChildMsg>();
            const char *objectName = childMsg.getObjName();
            const char *childObjectName = childMsg.getChildObjName();

            SceneObject *so_father = _sceneObjectManager->findSceneObject(
                objectName);
            SceneObject *so_child = _sceneObjectManager->findSceneObject(
                childObjectName);

            if (so_father == NULL || so_child == NULL)
                return;

            if (opencover::cover->debugLevel(3))
                fprintf(stderr,
                        "in SceneEditor grmsg::coGRMsg::ADD_CHILD_OBJECT object=%s child=%s remove=%d \n",
                        objectName, childObjectName, childMsg.getRemove());

            if (so_child->getParent() == so_father)
                return;

            // mount
            if (childMsg.getRemove() == 0)
            {
                MountEvent me;
                me.setMaster(so_father);
                me.setForce(true);
                so_child->receiveEvent(&me);
            }
            // unmount
            else
            {
                UnmountEvent ue;
                ue.setMaster(so_father);
                so_child->receiveEvent(&ue);
            }
        }
        break;
        case coGRMsg::SET_VARIANT:
        {
            auto &variantMsg = msg.as<coGRObjSetVariantMsg>();
            const char *objectName = variantMsg.getObjName();
            const char *groupName = variantMsg.getGroupName();
            const char *variantName = variantMsg.getVariantName();

            SwitchVariantEvent sve;
            sve.setGroup(groupName);
            sve.setVariant(variantName);

            if (opencover::cover->debugLevel(3))
                fprintf(stderr, "in SceneEditor grmsg::coGRMsg::SET_VARIANT object=%s \n",
                        objectName);

            SceneObject *so = _sceneObjectManager->findSceneObject(objectName);
            if (so)
                so->receiveEvent(&sve);
        }
        break;
        case coGRMsg::SET_APPEARANCE:
        {
            auto &appearanceMsg = msg.as<coGRObjSetAppearanceMsg>();
            const char *objectName = appearanceMsg.getObjName();
            const char *scopeName = appearanceMsg.getScopeName();
            osg::Vec4 color = osg::Vec4(appearanceMsg.getR(), appearanceMsg.getG(),
                                        appearanceMsg.getB(), appearanceMsg.getA());

            if (objectName && scopeName)
            {
                SetAppearanceColorEvent sace;
                sace.setScope(scopeName);
                sace.setColor(color);

                if (opencover::cover->debugLevel(3))
                    fprintf(stderr,
                            "in SceneEditor grmsg::coGRMsg::SET_APPEARANCE object=%s \n",
                            objectName);

                SceneObject *so = _sceneObjectManager->findSceneObject(objectName);
                if (so)
                    so->receiveEvent(&sace);
            }
        }
        break;
        case coGRMsg::KINEMATICS_STATE:
        {
            auto &stateMsg = msg.as<coGRObjKinematicsStateMsg>();

            const char *objectName = stateMsg.getObjName();
            const char *state = stateMsg.getState();

            if (objectName && state)
            {
                SceneObject *so = _sceneObjectManager->findSceneObject(objectName);
                if (so)
                {
                    InitKinematicsStateEvent ikse;
                    ikse.setState(state);
                    so->receiveEvent(&ikse);
                }
            }
        }
        default:
            break;
        }
    }
}

int SceneEditor::loadCoxmlUrl(const opencover::Url &url, osg::Group *group, const char *ck)
{
    return loadCoxml(url.str().c_str(), group, ck);
}

int
SceneEditor::loadCoxml(const char *filename, osg::Group *group,
                       const char *ck)
{
    (void)group;

    if (plugin)
    {
        if (opencover::cover->debugLevel(3))
            std::cout << "Loading COXML " << filename << std::endl;

        if (plugin->_sceneObjectManager)
        {
            SceneObject *so = plugin->_sceneObjectManager->createSceneObjectFromCoxmlFile(
                filename);
            if (!so)
            {
                std::cerr << "Error loading COXML " << filename << std::endl;
                return 1;
            }
            so->setCoviseKey(ck);

            // for now just try to mount to room
            if (so->getType() != SceneObjectTypes::ROOM)
            {
                Room *room = plugin->_sceneObjectManager->getRoom();
                if (room)
                {
                    MountEvent me;
                    me.setMaster(room);
                    so->receiveEvent(&me);
                }
            }
            else
            {
                // immediately select room
                SelectEvent se;
                so->receiveEvent(&se);
                plugin->_selectedSceneObject = so;
            }
        }
        else
        {
            std::cerr << "Error loading COXML " << filename << std::endl;
        }
    }

    return 0;
}

int
SceneEditor::replaceCoxml(const char *filename, osg::Group *group,
                          const char *ck)
{
    (void)filename;
    (void)group;
    (void)ck;

    return 0;
}

int
SceneEditor::unloadCoxml(const char *filename, const char *ck)
{
    (void)filename;
    (void)ck;
    // just do nothing
    // file manager cannot handle diffrent input-objects
    //std::cerr << "unloadCoxml " << filename << std::endl;
    // find scene object and delete it
    /*if (plugin->_sceneObjectManager)
    {
    SceneObject *so =plugin->_sceneObjectManager->findSceneObject(ck);
    if (so)
    plugin->_sceneObjectManager->deleteSceneObject(so);
    }*/

    return 0;
}

void
SceneEditor::deselect(SceneObject *so)
{
    if (_selectedSceneObject == so)
    {
        _selectedSceneObject = NULL;
        _mouseOverSceneObject = NULL;
    }
}

void
SceneEditor::key(int type, int keySym, int mod)
{
    //fprintf(stderr, "sceneeditor key %d %d", keySym, mod);

    if (type == osgGA::GUIEventAdapter::DOUBLECLICK)
    {
        if (_mouseOverSceneObject != NULL)
        {
            DoubleClickEvent dce;
            if (mod == osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON)
            {
                dce.setMouseButton(MouseEvent::TYPE_BUTTON_A);
            }
            else
            {
                dce.setMouseButton(MouseEvent::TYPE_BUTTON_C);
            }
            _mouseOverSceneObject->receiveEvent(&dce);
        }
    }
    else if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        if (keySym == 'p')
        {
            if (opencover::coVRNavigationManager::instance()->isViewerPosRotationEnabled())
            {
                opencover::VRSceneGraph::instance()->getTransform()->setMatrix(
                    _oldSceneGraphTransform);
                opencover::coVRNavigationManager::instance()->enableViewerPosRotation(false);
            }
            else
            {
                if (_selectedSceneObject)
                {

                    // TODO: move this to CameraBehavior

                    GetCameraEvent gce;

                    if (_selectedSceneObject->receiveEvent(&gce)
                        == EventErrors::SUCCESS) // && _selectedSceneObject->receiveEvent(&gte) == EventErrors::SUCCESS)
                    {
                        opencover::coVRNavigationManager::instance()->enableViewerPosRotation(true);
                        _oldSceneGraphTransform = opencover::VRSceneGraph::instance()->getTransform()->getMatrix();

                        // set player position -> move/rotate world
                        osg::Matrix newTransform = osg::Matrix::identity();
                        osg::Matrix viewerMatrix = opencover::cover->getViewerMat();

                        double sF = opencover::VRSceneGraph::instance()->scaleFactor();
                        newTransform.postMult(
                            osg::Matrix::scale(1 / sF, 1 / sF, 1 / sF));

                        // Transform
                        osg::Vec3 t = _selectedSceneObject->getTranslate().getTrans();
                        osg::Quat r = _selectedSceneObject->getRotate().getRotate();
                        newTransform.postMult(osg::Matrix::translate(-t));
                        newTransform.postMult(osg::Matrix::rotate(r.inverse()));

                        // CameraBehavior
                        newTransform.postMult(
                            osg::Matrix::translate(-gce.getPosition()));
                        newTransform.postMult(
                            osg::Matrix::rotate(gce.getOrientation().inverse()));

                        newTransform.postMult(viewerMatrix);

                        opencover::VRSceneGraph::instance()->getTransform()->setMatrix(
                            newTransform);
                    }
                }
            }
        }
        else if (keySym == osgGA::GUIEventAdapter::KEY_Delete)
        {
            if (_selectedSceneObject && _selectedSceneObject != _sceneObjectManager->getRoom())
            {
                _sceneObjectManager->requestDeleteSceneObject(_selectedSceneObject);
                _selectedSceneObject = NULL;
                _mouseOverSceneObject = NULL;
            }
        }
        else if ((keySym == osgGA::GUIEventAdapter::KEY_Up)
                 || (keySym == osgGA::GUIEventAdapter::KEY_Down)
                 || (keySym == osgGA::GUIEventAdapter::KEY_Left)
                 || (keySym == osgGA::GUIEventAdapter::KEY_Right))
        {
            if (_selectedSceneObject)
            {
                osg::Vec3 direction;
                if (keySym == osgGA::GUIEventAdapter::KEY_Up)
                    direction = osg::Vec3(0.0f, 1.0f, 0.0);
                if (keySym == osgGA::GUIEventAdapter::KEY_Down)
                    direction = osg::Vec3(0.0f, -1.0f, 0.0);
                if (keySym == osgGA::GUIEventAdapter::KEY_Left)
                    direction = osg::Vec3(-1.0f, 0.0f, 0.0);
                if (keySym == osgGA::GUIEventAdapter::KEY_Right)
                    direction = osg::Vec3(1.0f, 0.0f, 0.0);
                MoveObjectEvent moe;
                moe.setDirection(direction);
                _selectedSceneObject->receiveEvent(&moe);
            }
        }

        //      else if (keySym == 'm')
        //      {
        //         Room * room = _sceneObjectManager->getRoom();
        //         if ((room != NULL) && (_selectedSceneObject != room))
        //         {
        //            _selectedSceneObject->setTranslate(osg::Matrix::translate(room->getIntersectionWorld()));
        //         }
        //      }

        //       else if (keySym == 'x')
        //       {
        //          if (_selectedSceneObject)
        //          {
        //             SetTransformAxisEvent stae;
        //             stae.setTransformAxis(osg::Vec3(1.0f,0.0f,0.0f));
        //             _selectedSceneObject->receiveEvent(&stae);
        //          }
        //       }
        //       else if (keySym == 'y')
        //       {
        //          if (_selectedSceneObject)
        //          {
        //             SetTransformAxisEvent stae;
        //             stae.setTransformAxis(osg::Vec3(0.0f,1.0f,0.0f));
        //             _selectedSceneObject->receiveEvent(&stae);
        //          }
        //       }
        //       else if (keySym == 'z')
        //       {
        //          if (_selectedSceneObject)
        //          {
        //             SetTransformAxisEvent stae;
        //             stae.setTransformAxis(osg::Vec3(0.0f,0.0f,1.0f));
        //             _selectedSceneObject->receiveEvent(&stae);
        //          }
        //       }
        //       else if (keySym == 'j')
        //       {
        //          if (_selectedSceneObject)
        //          {
        //             if (_mouseOverSceneObject)
        //             {
        //                MountEvent me;
        //                me.setMaster(_mouseOverSceneObject);
        //                _selectedSceneObject->receiveEvent(&me);
        //             }
        //             else
        //             {
        //                std::vector<SceneObject*> rooms = _sceneObjectManager->getSceneObjectsOfType(SceneObjectTypes::ROOM);
        //                if (rooms.size()>0)
        //                {
        //                   MountEvent me;
        //                   me.setMaster(rooms[0]);
        //                   _selectedSceneObject->receiveEvent(&me);
        //                }
        //             }
        //          }
        //       }
        //       else if (keySym == 'u')
        //       {
        //          if (_selectedSceneObject)
        //          {
        //             UnmountEvent ue;
        //             ue.setMaster(_selectedSceneObject->getParent());
        //             _selectedSceneObject->receiveEvent(&ue);
        //          }
        //       }
    }
}

void SceneEditor::userEvent(int mod)
{
    if (mod == 1) // drop event
    {
        SceneObject *so = SceneObjectManager::instance()->getLatestSceneObject();
        Room *room = SceneObjectManager::instance()->getRoom();
        if (so && room && (so != room))
        {
            // unmount
            UnmountEvent ue;
            so->receiveEvent(&ue);
            // new position
            osg::Vec3 isectPoint = opencover::cover->getIntersectionHitPointWorld();
            isectPoint = osg::Matrixd::inverse(opencover::cover->getXformMat()).preMult(isectPoint);
            isectPoint /= opencover::VRSceneGraph::instance()->scaleFactor();
            so->setTranslate(osg::Matrix::translate(isectPoint));
            // mount
            MountEvent me;
            so->receiveEvent(&me);
        }
    }
}

COVERPLUGIN(SceneEditor)
