/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_VRUI_RENDER_INTERFACE_H
#define VR_VRUI_RENDER_INTERFACE_H

/*! \file
 \brief  OpenVRUI interface to OpenCOVER

 \author Andreas Kopecki <kopecki@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date  2004
 */

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <stack>

namespace vrui
{
class OSGVruiNode;
class OSGVruiMatrix;
class coAction;
}

namespace osg
{
class Texture2D;
}

namespace opencover
{

class VRVruiRenderInterface : public vrui::vruiRendererInterface
{
public:
    VRVruiRenderInterface();
    virtual ~VRVruiRenderInterface();

    virtual vrui::vruiNode *getScene() override;

    virtual vrui::vruiNode *getMenuGroup() override;
    virtual vrui::vruiUIElementProvider *createUIElementProvider(vrui::coUIElement *element) override;
    virtual vrui::vruiButtonProvider *createButtonProvider(vrui::coButtonGeometry *element) override;
    virtual vrui::vruiPanelGeometryProvider *createPanelGeometryProvider(vrui::coPanelGeometry *element) override;

    virtual vrui::vruiTransformNode *createTransformNode() override;
    virtual vrui::vruiMatrix *createMatrix() override;
    virtual void deleteMatrix(vrui::vruiMatrix *matrix) override;

    virtual std::string getName(const std::string &name) const override;

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

    virtual void remoteLock(int) override;
    virtual void remoteUnLock(int) override;
    virtual bool isLocked(int) override;
    virtual bool isLockedByMe(int) override;

private:
    vrui::OSGVruiNode *groupNode = nullptr;
    vrui::OSGVruiNode *sceneNode = nullptr;
    vrui::OSGVruiMatrix *handMatrix = nullptr;
    vrui::OSGVruiMatrix *headMatrix = nullptr;
    vrui::OSGVruiMatrix *mouseMatrix = nullptr;
    vrui::OSGVruiMatrix *relativeMatrix = nullptr;

    std::stack<vrui::vruiMatrix *> matrixStack;
    std::string look;
};
}
#endif
