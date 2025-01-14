/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvVruiRenderInterface.h"

#include "vvPluginSupport.h"
#include "vvVruiButtons.h"

#include <config/CoviseConfig.h>

#define WINW 800
#define WINH 600

#include <vsg/all.h>

#include "vvActionUserData.h"
#include "vvCollabInterface.h"
#include "vvIntersection.h"
#include "vvCollaboration.h"
#include "vvCommunication.h"
#include "vvFileManager.h"
//#include "vvTouchTable.h"
#include "vvConfig.h"
#include "vvSceneGraph.h"

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

#include <OpenVRUI/vsg/VSGVruiColoredBackground.h>
#include <OpenVRUI/vsg/VSGVruiTexturedBackground.h>
#include <OpenVRUI/vsg/VSGVruiTextureRectBackground.h>
#include <OpenVRUI/vsg/VSGVruiDefaultButtonGeometry.h>
#include <OpenVRUI/vsg/VSGVruiTextButtonGeometry.h>
#include <OpenVRUI/vsg/VSGVruiFlatButtonGeometry.h>
#include <OpenVRUI/vsg/VSGVruiSquareButtonGeometry.h>
#include <OpenVRUI/vsg/VSGVruiRectButtonGeometry.h>
#include <OpenVRUI/vsg/VSGVruiFlatPanelGeometry.h>
#include <OpenVRUI/vsg/VSGVruiFrame.h>
#include <OpenVRUI/vsg/VSGVruiMatrix.h>
#include <OpenVRUI/vsg/VSGVruiPanelGeometry.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiLabel.h>
#include <OpenVRUI/vsg/VSGVruiToggleButtonGeometry.h>
#include <OpenVRUI/vsg/VSGVruiSlider.h>
#include <OpenVRUI/vsg/VSGVruiNull.h>
#include <OpenVRUI/vsg/VSGVruiHit.h>
#include <OpenVRUI/vsg/VSGVruiValuePoti.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiActionUserData.h>

#include <OpenVRUI/util/vruiLog.h>

#include <PluginUtil/PluginMessageTypes.h>

using namespace vsg;
using namespace vive;
using namespace vrui;
using covise::coCoviseConfig;

using std::string;

vvVruiRenderInterface::vvVruiRenderInterface()
{
    assert(!vruiRendererInterface::theInterface);

    groupNode = 0;
    sceneNode = 0;
    buttons = new vvVruiButtons(vvVruiButtons::Pointer);
    mouseButtons = new vvVruiButtons(vvVruiButtons::Mouse);
    relativeButtons = new vvVruiButtons(vvVruiButtons::Relative);

    handMatrix = new VSGVruiMatrix();
    headMatrix = new VSGVruiMatrix();
    mouseMatrix = new VSGVruiMatrix();
    relativeMatrix = new VSGVruiMatrix();

    handMatrix->makeIdentity();
    headMatrix->makeIdentity();
    mouseMatrix->makeIdentity();
    relativeMatrix->makeIdentity();

    vruiRendererInterface::theInterface = this;
    vvIntersection::instance();

    look = coCoviseConfig::getEntry("COVER.LookAndFeel");
}

vvVruiRenderInterface::~vvVruiRenderInterface()
{
    delete alwaysVisibleNode;
    alwaysVisibleNode = nullptr;
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

    delete vvIntersection::instance();
    vruiRendererInterface::theInterface = 0;
}

void addToTransfer(vsg::Data* transfer)
{

}

vruiNode *vvVruiRenderInterface::getAlwaysVisibleGroup()
{
    if (!alwaysVisibleNode)
    {
        Group *group = vvSceneGraph::instance()->getAlwaysVisibleGroup();
        alwaysVisibleNode = new VSGVruiNode(group);
    }
    return alwaysVisibleNode;

}

vruiNode *vvVruiRenderInterface::getMenuGroup()
{
    if (!groupNode)
    {
        Group *group = vv->getMenuGroup();
        groupNode = new VSGVruiNode(group);
    }
    return groupNode;
}

vruiNode *vvVruiRenderInterface::getScene()
{
    if (!sceneNode)
    {
        Group *group = vv->getObjectsRoot();
        sceneNode = new VSGVruiNode(group);
    }
    return sceneNode;
}

vruiUIElementProvider *vvVruiRenderInterface::createUIElementProvider(coUIElement *element)
{

    string name(element->getClassName());

    if (name == "coFrame")
    {
        coFrame *frame = dynamic_cast<coFrame *>(element);
        if (frame)
        {
            //cerr << "vvVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new VSGVruiFrame(frame);
        }
    }

    if (name == "coBackground")
    {
        coBackground *back = dynamic_cast<coBackground *>(element);
        if (back)
        {
            //cerr << "vvVruiRenderInterface::createUIElementProvider info: creating (null) provider for " << name << endl;
            return new VSGVruiNull(back);
        }
    }

    if (name == "coColoredBackground")
    {
        coColoredBackground *back = dynamic_cast<coColoredBackground *>(element);
        if (back)
        {
            //cerr << "vvVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new VSGVruiColoredBackground(back);
        }
    }

    if (name == "coTexturedBackground")
    {
        coTexturedBackground *back = dynamic_cast<coTexturedBackground *>(element);
        if (back)
        {
            //cerr << "vvVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new VSGVruiTexturedBackground(back);
        }
    }

    if (name == "coTextureRectBackground")
    {
        coTextureRectBackground *back = dynamic_cast<coTextureRectBackground *>(element);
        if (back)
        {
            //cerr << "vvVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new VSGVruiTextureRectBackground(back);
        }
    }

    if (name == "coLabel")
    {
        coLabel *label = dynamic_cast<coLabel *>(element);
        if (label)
        {
            //cerr << "vvVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new VSGVruiLabel(label);
        }
    }

    if (name == "coSlider")
    {
        coSlider *slider = dynamic_cast<coSlider *>(element);
        if (slider)
        {
            //cerr << "vvVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new VSGVruiSlider(slider);
        }
    }

    if (name == "coValuePoti")
    {
        coValuePoti *poti = dynamic_cast<coValuePoti *>(element);
        if (poti)
        {
            //cerr << "vvVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new VSGVruiValuePoti(poti);
        }
    }

    if (name == "coSlopePoti")
    {
        coSlopePoti *poti = dynamic_cast<coSlopePoti *>(element);
        if (poti)
        {
            //cerr << "vvVruiRenderInterface::createUIElementProvider info: creating provider for " << name << endl;
            return new VSGVruiValuePoti(poti);
        }
    }

    //VRUILOG("vvVruiRenderInterface::createUIElementProvider err: " << name << ": no provider found");
    return 0;
}

vruiButtonProvider *vvVruiRenderInterface::createButtonProvider(coButtonGeometry *element)
{

    string name(element->getClassName());

    if (name == "coDefaultButtonGeometry")
    {
        coDefaultButtonGeometry *button = dynamic_cast<coDefaultButtonGeometry *>(element);
        if (button)
        {
            //cerr << "vvVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new VSGVruiDefaultButtonGeometry(button);
        }
    }

    if (name == "coFlatButtonGeometry")
    {
        coFlatButtonGeometry *button = dynamic_cast<coFlatButtonGeometry *>(element);
        if (button)
        {
            //cerr << "vvVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new VSGVruiFlatButtonGeometry(button);
        }
    }

    if (name == "coSquareButtonGeometry")
    {
        coSquareButtonGeometry *button = dynamic_cast<coSquareButtonGeometry *>(element);
        if (button)
        {
            //cerr << "vvVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new VSGVruiSquareButtonGeometry(button);
        }
    }

    if (name == "coRectButtonGeometry")
    {
        coRectButtonGeometry *button = dynamic_cast<coRectButtonGeometry *>(element);
        if (button)
        {
            //cerr << "vvVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new VSGVruiRectButtonGeometry(button);
        }
    }

    if (name == "coToggleButtonGeometry")
    {
        coToggleButtonGeometry *button = dynamic_cast<coToggleButtonGeometry *>(element);
        if (button)
        {
            //cerr << "vvVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new VSGVruiToggleButtonGeometry(button);
        }
    }

    if (name == "coTextButtonGeometry")
    {
        coTextButtonGeometry *button = dynamic_cast<coTextButtonGeometry *>(element);
        if (button)
        {
            //cerr << "vvVruiRenderInterface::createButtonProvider info: creating provider for " << name << endl;
            return new VSGVruiTextButtonGeometry(button);
        }
    }

    VRUILOG("vvVruiRenderInterface::createButtonProvider err: " << name << ": no provider found");
    return 0;
}

vruiPanelGeometryProvider *vvVruiRenderInterface::createPanelGeometryProvider(coPanelGeometry *element)
{

    string name(element->getClassName());

    if (name == "coPanelGeometry")
    {

        coPanelGeometry *panel = dynamic_cast<coPanelGeometry *>(element);
        if (panel)
        {
            //cerr << "vvVruiRenderInterface::createPanelGeometryProvider info: creating provider for " << name << endl;
            return new VSGVruiPanelGeometry(panel);
        }
    }

    if (name == "coFlatPanelGeometry")
    {
        coFlatPanelGeometry *panel = dynamic_cast<coFlatPanelGeometry *>(element);
        if (panel)
        {
            //cerr << "vvVruiRenderInterface::createPanelGeometryProvider info: creating provider for " << name << endl;
            return new VSGVruiFlatPanelGeometry(panel);
        }
    }

    VRUILOG("vvVruiRenderInterface::createPanelGeometryProvider err: " << name << ": no provider found");
    return 0;
}

vruiTransformNode *vvVruiRenderInterface::createTransformNode()
{

    MatrixTransform *transform = new MatrixTransform();

    return new VSGVruiTransformNode(transform);
}

vruiMatrix *vvVruiRenderInterface::createMatrix()
{
    if (matrixStack.empty())
    {
        //VRUILOG("vvVruiRenderInterface::createMatrix: info: new matrix")
        return new VSGVruiMatrix();
    }
    else
    {
        //VRUILOG("vvVruiRenderInterface::createMatrix: info: stack matrix " << matrixStack.size())
        vruiMatrix *matrix = matrixStack.top();
        matrixStack.pop();
        return matrix;
    }
}

void vvVruiRenderInterface::deleteMatrix(vruiMatrix *matrix)
{
    if (matrix && matrixStack.size() <= 1024)
    {
        matrixStack.push(matrix);
        //VRUILOG("vvVruiRenderInterface::deleteMatrix: info: push stack " << matrixStack.size())
    }
    else
    {
        //VRUILOG("vvVruiRenderInterface::deleteMatrix: info: delete")
        delete matrix;
    }
}

string vvVruiRenderInterface::getName(const string &name) const
{
    const char *tmp = vvFileManager::instance()->getName(name.c_str());
    if (tmp)
        return tmp;
    else
        return "";
}

string vvVruiRenderInterface::getFont(const string &name) const
{
    if (name.empty())
        return vvFileManager::instance()->getFontFile(nullptr);
    return vvFileManager::instance()->getFontFile(name.c_str());
}

vruiTexture *vvVruiRenderInterface::createTexture(const string &name)
{
    return new VSGVruiTexture(vvFileManager::instance()->loadTexture(name.c_str()));
}

coAction::Result vvVruiRenderInterface::hit(coAction *action, vruiHit *)
{

    //FIXME: Catching too much...

    if (dynamic_cast<coButton *>(action) || dynamic_cast<coRotButton *>(action) || dynamic_cast<coSlider *>(action) || dynamic_cast<coValuePoti *>(action))
    {
        if (vvCollaboration::instance()->getCouplingMode() == vvCollaboration::MasterSlaveCoupling
            && !vvCollaboration::instance()->isMaster())
            return coAction::ACTION_DONE;
    }

    return coAction::ACTION_UNDEF;
}

void vvVruiRenderInterface::miss(coAction *)
{
}

vruiActionUserData *vvVruiRenderInterface::createActionUserData(coAction *action)
{
    vvActionUserData *ud = new vvActionUserData(action);
    return ud;
}

void vvVruiRenderInterface::deleteUserData(vruiUserData *userData)
{
    delete userData;
}

vruiUserData *vvVruiRenderInterface::createUserData()
{
    return 0;
}

coUpdateManager *vvVruiRenderInterface::getUpdateManager()
{
    if (cover)
        return vv->getUpdateManager();
    else
        return 0;
}

coJoystickManager *vvVruiRenderInterface::getJoystickManager()
{
    return coJoystickManager::instance();
}

void vvVruiRenderInterface::removePointerIcon(const string &name)
{
    VSGVruiNode *iconNode = dynamic_cast<VSGVruiNode *>(getIcon(name, true));
    if ((vvCollaboration::instance()->getCouplingMode() != vvCollaboration::MasterSlaveCoupling || vvCollaboration::instance()->isMaster()) && iconNode && iconNode->getNodePtr())
    {
        vvSceneGraph::instance()->removePointerIcon(iconNode->getNodePtr());
    }
}

void vvVruiRenderInterface::addPointerIcon(const string &name)
{
    VSGVruiNode *iconNode = dynamic_cast<VSGVruiNode *>(getIcon(name, true));
    if ((vvCollaboration::instance()->getCouplingMode() != vvCollaboration::MasterSlaveCoupling || vvCollaboration::instance()->isMaster()) && iconNode && iconNode->getNodePtr())
    {
        vvSceneGraph::instance()->addPointerIcon(iconNode->getNodePtr());
    }
}

// load an icon file looks in covise/icons/$LookAndFeel or covise/icons
// returns 0, if nothing found

vruiNode *vvVruiRenderInterface::getIcon(const string &iconName, bool shared)
{
    VSGVruiNode *node = 0;
    ref_ptr<Node> osgNode;
    //static bool haveIvPlugin = true;

    if (shared)
    {
        map<string, vruiNode *>::iterator nodeIterator = iconsList.find(iconName);

        if (nodeIterator != iconsList.end())
        {
            node = dynamic_cast<VSGVruiNode *>(nodeIterator->second);
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
        osgNode = vvFileManager::instance()->loadIcon(iconName.c_str());

        if (osgNode.get())
        {
            node = new VSGVruiNode(osgNode.get());
            node->setName(iconName);
            iconsList[iconName] = node;
        }
    }

    return node;
}

vruiMatrix *vvVruiRenderInterface::getViewerMatrix() const
{
    headMatrix->matrix = (vv->getViewerMat());
    return headMatrix;
}

vruiMatrix *vvVruiRenderInterface::getHandMatrix() const
{
    handMatrix->matrix = (vv->getPointerMat());
    return handMatrix;
}

vruiMatrix *vvVruiRenderInterface::getMouseMatrix() const
{
    mouseMatrix->matrix = (vv->getMouseMat());
    return mouseMatrix;
}

vruiMatrix *vvVruiRenderInterface::getRelativeMatrix() const
{
    relativeMatrix->matrix = (vv->getRelativeMat());
    return relativeMatrix;
}

bool vvVruiRenderInterface::is2DInputDevice() const
{

    return vvConfig::instance()->mouseTracking();
}

bool vvVruiRenderInterface::isMultiTouchDevice() const
{

    return (vvTouchTable::instance()->ttInterface->isPlanar());
}

void vvVruiRenderInterface::sendCollabMessage(vruiCollabInterface *myinterface, const char *buffer, int length)
{
    if (vvCollaboration::instance()->getCouplingMode() != vvCollaboration::MasterSlaveCoupling
        || vvCollaboration::instance()->isMaster())
    {
        coCOIM *coim = dynamic_cast<coCOIM *>(myinterface->getManager());
        vv->sendMessage(coim->getPlugin(),
                           vvPluginSupport::TO_SAME_OTHERS,
                           (PluginMessageTypes::Type)myinterface->getType(),
                           length,
                           buffer);
    }
}

int vvVruiRenderInterface::getClientId()
{
    return vvCommunication::instance()->getID();
}

bool vvVruiRenderInterface::isRemoteBlockNececcary()
{
	//block in tightcoupling and as slave
	if (vvCollaboration::instance()->getCouplingMode() == vvCollaboration::LooseCoupling ||
		(vvCollaboration::instance()->getCouplingMode() == vvCollaboration::MasterSlaveCoupling && vvCollaboration::instance()->isMaster()))
	{
		return false;
	} 
	return true;
}

double vvVruiRenderInterface::getFrameTime() const
{
    return vv->frameTime();
}
