/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VRVruiRenderInterface.h"

#include "coVRPluginSupport.h"
#include "VRVruiButtons.h"

#include <config/CoviseConfig.h>

#define WINW 800
#define WINH 600

#include <osg/Geode>
#include <osg/Math>
#include <osg/MatrixTransform>

#include "coActionUserData.h"
#include "coCollabInterface.h"
#include "coIntersection.h"
#include "coVRCollaboration.h"
#include "coVRCommunication.h"
#include "coVRFileManager.h"
#include "VRSceneGraph.h"
#include "coVRTouchTable.h"
#include "coVRConfig.h"

#undef START
#undef STOP

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coSquareButtonGeometry.h>
#include <OpenVRUI/coDefaultButtonGeometry.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coTextButtonGeometry.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <OpenVRUI/coTextureRectBackground.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coToggleButtonGeometry.h>
#include <OpenVRUI/coSlider.h>
#include <OpenVRUI/coValuePoti.h>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coJoystickManager.h>

#include <OpenVRUI/osg/OSGVruiColoredBackground.h>
#include <OpenVRUI/osg/OSGVruiTexturedBackground.h>
#include <OpenVRUI/osg/OSGVruiTextureRectBackground.h>
#include <OpenVRUI/osg/OSGVruiDefaultButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiTextButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiFlatButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiSquareButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiRectButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiFlatPanelGeometry.h>
#include <OpenVRUI/osg/OSGVruiFrame.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/osg/OSGVruiPanelGeometry.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiLabel.h>
#include <OpenVRUI/osg/OSGVruiToggleButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiSlider.h>
#include <OpenVRUI/osg/OSGVruiNull.h>
#include <OpenVRUI/osg/OSGVruiHit.h>
#include <OpenVRUI/osg/OSGVruiValuePoti.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiActionUserData.h>

#include <OpenVRUI/util/vruiLog.h>

#include <PluginUtil/PluginMessageTypes.h>

using namespace osg;
using namespace opencover;
using namespace vrui;
using covise::coCoviseConfig;

using std::string;

VRVruiRenderInterface::VRVruiRenderInterface()
{
    assert(!vruiRendererInterface::theInterface);

    groupNode = 0;
    sceneNode = 0;
    buttons = new VRVruiButtons(VRVruiButtons::Pointer);
    mouseButtons = new VRVruiButtons(VRVruiButtons::Mouse);
    relativeButtons = new VRVruiButtons(VRVruiButtons::Relative);

    handMatrix = new OSGVruiMatrix();
    headMatrix = new OSGVruiMatrix();
    mouseMatrix = new OSGVruiMatrix();
    relativeMatrix = new OSGVruiMatrix();

    handMatrix->makeIdentity();
    headMatrix->makeIdentity();
    mouseMatrix->makeIdentity();
    relativeMatrix->makeIdentity();

    vruiRendererInterface::theInterface = this;
    coIntersection::instance();

    look = coCoviseConfig::getEntry("COVER.LookAndFeel");
}

VRVruiRenderInterface::~VRVruiRenderInterface()
{
    delete groupNode;
    groupNode = 0;
    delete sceneNode;
    sceneNode = NULL;

    delete buttons;
    delete mouseButtons;
    delete relativeButtons;

    delete handMatrix;
    delete headMatrix;
    delete mouseMatrix;
    delete relativeMatrix;

    //delete coIntersection::instance();
    vruiRendererInterface::theInterface = 0;
}

vruiNode *VRVruiRenderInterface::getMenuGroup()
{
    if (!groupNode)
    {
        Group *group = cover->getMenuGroup();
        groupNode = new OSGVruiNode(group);
    }
    return groupNode;
}

vruiNode *VRVruiRenderInterface::getScene()
{
    if (!sceneNode)
    {
        Group *group = cover->getObjectsRoot();
        sceneNode = new OSGVruiNode(group);
    }
    return sceneNode;
}

vruiUIElementProvider *VRVruiRenderInterface::createUIElementProvider(coUIElement *element)
{

    string name(element->getClassName());

    if (name == "coFrame")
    {
        coFrame *frame = dynamic_cast<coFrame *>(element);
        if (frame)
        {
            //cerr << "VRVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiFrame(frame);
        }
    }

    if (name == "coBackground")
    {
        coBackground *back = dynamic_cast<coBackground *>(element);
        if (back)
        {
            //cerr << "VRVruiRenderInterface::createUIElementProvider info: creating (null) provider for " << name << endl;
            return new OSGVruiNull(back);
        }
    }

    if (name == "coColoredBackground")
    {
        coColoredBackground *back = dynamic_cast<coColoredBackground *>(element);
        if (back)
        {
            //cerr << "VRVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiColoredBackground(back);
        }
    }

    if (name == "coTexturedBackground")
    {
        coTexturedBackground *back = dynamic_cast<coTexturedBackground *>(element);
        if (back)
        {
            //cerr << "VRVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiTexturedBackground(back);
        }
    }

    if (name == "coTextureRectBackground")
    {
        coTextureRectBackground *back = dynamic_cast<coTextureRectBackground *>(element);
        if (back)
        {
            //cerr << "VRVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiTextureRectBackground(back);
        }
    }

    if (name == "coLabel")
    {
        coLabel *label = dynamic_cast<coLabel *>(element);
        if (label)
        {
            //cerr << "VRVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiLabel(label);
        }
    }

    if (name == "coSlider")
    {
        coSlider *slider = dynamic_cast<coSlider *>(element);
        if (slider)
        {
            //cerr << "VRVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiSlider(slider);
        }
    }

    if (name == "coValuePoti")
    {
        coValuePoti *poti = dynamic_cast<coValuePoti *>(element);
        if (poti)
        {
            //cerr << "VRVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiValuePoti(poti);
        }
    }

    if (name == "coSlopePoti")
    {
        coSlopePoti *poti = dynamic_cast<coSlopePoti *>(element);
        if (poti)
        {
            //cerr << "VRVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiValuePoti(poti);
        }
    }

    //VRUILOG("VRVruiRenderInterface::createUIElementProvider err: " << name << ": no provider found");
    return 0;
}

vruiButtonProvider *VRVruiRenderInterface::createButtonProvider(coButtonGeometry *element)
{

    string name(element->getClassName());

    if (name == "coDefaultButtonGeometry")
    {
        coDefaultButtonGeometry *button = dynamic_cast<coDefaultButtonGeometry *>(element);
        if (button)
        {
            //cerr << "VRVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new OSGVruiDefaultButtonGeometry(button);
        }
    }

    if (name == "coFlatButtonGeometry")
    {
        coFlatButtonGeometry *button = dynamic_cast<coFlatButtonGeometry *>(element);
        if (button)
        {
            //cerr << "VRVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new OSGVruiFlatButtonGeometry(button);
        }
    }

    if (name == "coSquareButtonGeometry")
    {
        coSquareButtonGeometry *button = dynamic_cast<coSquareButtonGeometry *>(element);
        if (button)
        {
            //cerr << "VRVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new OSGVruiSquareButtonGeometry(button);
        }
    }

    if (name == "coRectButtonGeometry")
    {
        coRectButtonGeometry *button = dynamic_cast<coRectButtonGeometry *>(element);
        if (button)
        {
            //cerr << "VRVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new OSGVruiRectButtonGeometry(button);
        }
    }

    if (name == "coToggleButtonGeometry")
    {
        coToggleButtonGeometry *button = dynamic_cast<coToggleButtonGeometry *>(element);
        if (button)
        {
            //cerr << "VRVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new OSGVruiToggleButtonGeometry(button);
        }
    }

    if (name == "coTextButtonGeometry")
    {
        coTextButtonGeometry *button = dynamic_cast<coTextButtonGeometry *>(element);
        if (button)
        {
            //cerr << "VRVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new OSGVruiTextButtonGeometry(button);
        }
    }

    VRUILOG("VRVruiRenderInterface::createButtonProvider err: " << name << ": no provider found");
    return 0;
}

vruiPanelGeometryProvider *VRVruiRenderInterface::createPanelGeometryProvider(coPanelGeometry *element)
{

    string name(element->getClassName());

    if (name == "coPanelGeometry")
    {

        coPanelGeometry *panel = dynamic_cast<coPanelGeometry *>(element);
        if (panel)
        {
            //cerr << "VRVruiRenderInterface::createPanelGeometryProvider info: creating provider for " << name << endl;
            return new OSGVruiPanelGeometry(panel);
        }
    }

    if (name == "coFlatPanelGeometry")
    {
        coFlatPanelGeometry *panel = dynamic_cast<coFlatPanelGeometry *>(element);
        if (panel)
        {
            //cerr << "VRVruiRenderInterface::createPanelGeometryProvider info: creating provider for " << name << endl;
            return new OSGVruiFlatPanelGeometry(panel);
        }
    }

    VRUILOG("VRVruiRenderInterface::createPanelGeometryProvider err: " << name << ": no provider found");
    return 0;
}

vruiTransformNode *VRVruiRenderInterface::createTransformNode()
{

    MatrixTransform *transform = new MatrixTransform();

    return new OSGVruiTransformNode(transform);
}

vruiMatrix *VRVruiRenderInterface::createMatrix()
{
    if (matrixStack.empty())
    {
        //VRUILOG("VRVruiRenderInterface::createMatrix: info: new matrix")
        return new OSGVruiMatrix();
    }
    else
    {
        //VRUILOG("VRVruiRenderInterface::createMatrix: info: stack matrix " << matrixStack.size())
        vruiMatrix *matrix = matrixStack.top();
        matrixStack.pop();
        return matrix;
    }
}

void VRVruiRenderInterface::deleteMatrix(vruiMatrix *matrix)
{
    if (matrix && matrixStack.size() <= 1024)
    {
        matrixStack.push(matrix);
        //VRUILOG("VRVruiRenderInterface::deleteMatrix: info: push stack " << matrixStack.size())
    }
    else
    {
        //VRUILOG("VRVruiRenderInterface::deleteMatrix: info: delete")
        delete matrix;
    }
}

string VRVruiRenderInterface::getName(const string &name) const
{
    const char *tmp = coVRFileManager::instance()->getName(name.c_str());
    if (tmp)
        return tmp;
    else
        return "";
}

vruiTexture *VRVruiRenderInterface::createTexture(const string &name)
{
    return new OSGVruiTexture(coVRFileManager::instance()->loadTexture(name.c_str()));
}

coAction::Result VRVruiRenderInterface::hit(coAction *action, vruiHit *)
{

    //FIXME: Catching too much...

    if (dynamic_cast<coButton *>(action) || dynamic_cast<coRotButton *>(action) || dynamic_cast<coSlider *>(action) || dynamic_cast<coValuePoti *>(action))
    {
        if (coVRCollaboration::instance()->getSyncMode() == coVRCollaboration::MasterSlaveCoupling
            && !coVRCollaboration::instance()->isMaster())
            return coAction::ACTION_DONE;
    }

    return coAction::ACTION_UNDEF;
}

void VRVruiRenderInterface::miss(coAction *)
{
}

vruiActionUserData *VRVruiRenderInterface::createActionUserData(coAction *action)
{
    coActionUserData *ud = new coActionUserData(action);
    return ud;
}

void VRVruiRenderInterface::deleteUserData(vruiUserData *userData)
{
    delete userData;
}

vruiUserData *VRVruiRenderInterface::createUserData()
{
    return 0;
}

coUpdateManager *VRVruiRenderInterface::getUpdateManager()
{
    if (cover)
        return cover->getUpdateManager();
    else
        return 0;
}

coJoystickManager *VRVruiRenderInterface::getJoystickManager()
{
    return coJoystickManager::instance();
}

void VRVruiRenderInterface::removePointerIcon(const string &name)
{
    OSGVruiNode *iconNode = dynamic_cast<OSGVruiNode *>(getIcon(name, true));
    if ((coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::MasterSlaveCoupling || coVRCollaboration::instance()->isMaster()) && iconNode && iconNode->getNodePtr())
    {
        VRSceneGraph::instance()->removePointerIcon(iconNode->getNodePtr());
    }
}

void VRVruiRenderInterface::addPointerIcon(const string &name)
{
    OSGVruiNode *iconNode = dynamic_cast<OSGVruiNode *>(getIcon(name, true));
    if ((coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::MasterSlaveCoupling || coVRCollaboration::instance()->isMaster()) && iconNode && iconNode->getNodePtr())
    {
        VRSceneGraph::instance()->addPointerIcon(iconNode->getNodePtr());
    }
}

// load an icon file looks in covise/icons/$LookAndFeel or covise/icons
// returns 0, if nothing found

vruiNode *VRVruiRenderInterface::getIcon(const string &iconName, bool shared)
{
    OSGVruiNode *node = 0;
    ref_ptr<Node> osgNode;
    //static bool haveIvPlugin = true;

    if (shared)
    {
        map<string, vruiNode *>::iterator nodeIterator = iconsList.find(iconName);

        if (nodeIterator != iconsList.end())
        {
            node = dynamic_cast<OSGVruiNode *>(nodeIterator->second);
        }
    }

    if (!node)
    {

        //std::string ivPluginName = osgDB::Registry::instance()->createLibraryNameForExtension("iv");
        //if (osgDB::findLibraryFile(ivPluginName).empty() && haveIvPlugin)
        //{
        //   haveIvPlugin = false;
        //   cerr << "Error: OpenSceneGraph's iv plugin not found" << endl;
        //}
        osgNode = coVRFileManager::instance()->loadIcon(iconName.c_str());

        if (osgNode.get())
        {
            node = new OSGVruiNode(osgNode.get());
            node->setName(iconName);
            iconsList[iconName] = node;
        }
    }

    return node;
}

vruiMatrix *VRVruiRenderInterface::getViewerMatrix() const
{
    headMatrix->setMatrix(cover->getViewerMat());
    return headMatrix;
}

vruiMatrix *VRVruiRenderInterface::getHandMatrix() const
{
    handMatrix->setMatrix(cover->getPointerMat());
    return handMatrix;
}

vruiMatrix *VRVruiRenderInterface::getMouseMatrix() const
{
    mouseMatrix->setMatrix(cover->getMouseMat());
    return mouseMatrix;
}

vruiMatrix *VRVruiRenderInterface::getRelativeMatrix() const
{
    relativeMatrix->setMatrix(cover->getRelativeMat());
    return relativeMatrix;
}

bool VRVruiRenderInterface::is2DInputDevice() const
{

    return coVRConfig::instance()->mouseTracking();
}

bool VRVruiRenderInterface::isMultiTouchDevice() const
{

    return (coVRTouchTable::instance()->ttInterface->isPlanar());
}

void VRVruiRenderInterface::sendCollabMessage(vruiCollabInterface *myinterface, const char *buffer, int length)
{
    if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::MasterSlaveCoupling
        || coVRCollaboration::instance()->isMaster())
    {
        coCOIM *coim = dynamic_cast<coCOIM *>(myinterface->getManager());
        cover->sendMessage(coim->getPlugin(),
                           coVRPluginSupport::TO_SAME_OTHERS,
                           (PluginMessageTypes::Type)myinterface->getType(),
                           length,
                           buffer);
    }
}

void VRVruiRenderInterface::remoteLock(int ID)
{
    coVRCommunication::instance()->RILock(ID);
}
void VRVruiRenderInterface::remoteUnLock(int ID)
{
    coVRCommunication::instance()->RIUnLock(ID);
}
bool VRVruiRenderInterface::isLocked(int ID)
{
    return coVRCommunication::instance()->isRILocked(ID);
}
bool VRVruiRenderInterface::isLockedByMe(int ID)
{
    return coVRCommunication::instance()->isRILockedByMe(ID);
}

double VRVruiRenderInterface::getFrameTime() const
{
    return cover->frameTime();
}
