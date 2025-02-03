/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <vsg/core/Data.h>

#include <stack>

namespace vrui
{
class VSGVruiNode;
class VSGVruiMatrix;
class coAction;
}


namespace vive
{

class vvVruiRenderInterface : public vrui::vruiRendererInterface
{
public:
    vvVruiRenderInterface();
    virtual ~vvVruiRenderInterface();

    virtual vrui::vruiNode *getAlwaysVisibleGroup() override;
    virtual vrui::vruiNode *getScene() override;

    virtual vrui::vruiNode *getMenuGroup() override;
    virtual vrui::vruiUIElementProvider *createUIElementProvider(vrui::coUIElement *element) override;
    virtual vrui::vruiButtonProvider *createButtonProvider(vrui::coButtonGeometry *element) override;
    virtual vrui::vruiPanelGeometryProvider *createPanelGeometryProvider(vrui::coPanelGeometry *element) override;
    virtual void addToTransfer(vsg::BufferInfo* bi) override;

    virtual vrui::vruiTransformNode *createTransformNode() override;
    virtual vrui::vruiMatrix *createMatrix() override;
    virtual void deleteMatrix(vrui::vruiMatrix *matrix) override;

    virtual std::string getName(const std::string &name) const override;
    virtual std::string getFont(const std::string &name) const override;

    virtual vrui::vruiTexture *createTexture(const std::string &textureName) override;

    virtual vrui::coUpdateManager *getUpdateManager() override;
    virtual vrui::coJoystickManager *getJoystickManager() override;

    virtual vrui::vruiActionUserData *createActionUserData(vrui::coAction *action) override;
    virtual vrui::vruiUserData *createUserData() override;
    virtual void deleteUserData(vrui::vruiUserData *userData) override;

    virtual vrui::coAction::Result hit(vrui::coAction *action, vrui::vruiHit *hit) override;
    virtual void miss(vrui::coAction *action) override;

    // remove pointer indicator
    virtual void removePointerIcon(const std::string &name) override;
    // add    pointer indicator
    virtual void addPointerIcon(const std::string &name) override;

    virtual vrui::vruiNode *getIcon(const std::string &iconName, bool shared = false) override;

    virtual vrui::vruiMatrix *getViewerMatrix() const override;
    virtual vrui::vruiMatrix *getHandMatrix() const override;
    virtual vrui::vruiMatrix *getMouseMatrix() const override;
    virtual vrui::vruiMatrix *getRelativeMatrix() const override;

    virtual bool is2DInputDevice() const override;
    virtual bool isMultiTouchDevice() const override;

    virtual void sendCollabMessage(vrui::vruiCollabInterface *myinterface, const char *buffer, int length) override;

    virtual double getFrameTime() const override;

	virtual int getClientId() override;
	virtual bool isRemoteBlockNececcary() override;

    virtual bool compileNode(vrui::vruiNode*) override;
private:
    vrui::VSGVruiNode *alwaysVisibleNode = nullptr;
    vrui::VSGVruiNode *groupNode = nullptr;
    vrui::VSGVruiNode *sceneNode = nullptr;
    vrui::VSGVruiMatrix *handMatrix = nullptr;
    vrui::VSGVruiMatrix *headMatrix = nullptr;
    vrui::VSGVruiMatrix *mouseMatrix = nullptr;
    vrui::VSGVruiMatrix *relativeMatrix = nullptr;

    std::stack<vrui::vruiMatrix *> matrixStack;
    std::string look;
};
}
