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

    virtual vrui::vruiNode *getScene();

    virtual vrui::vruiNode *getMenuGroup();
    virtual vrui::vruiUIElementProvider *createUIElementProvider(vrui::coUIElement *element);
    virtual vrui::vruiButtonProvider *createButtonProvider(vrui::coButtonGeometry *element);
    virtual vrui::vruiPanelGeometryProvider *createPanelGeometryProvider(vrui::coPanelGeometry *element);

    virtual vrui::vruiTransformNode *createTransformNode();
    virtual vrui::vruiMatrix *createMatrix();
    virtual void deleteMatrix(vrui::vruiMatrix *matrix);

    virtual std::string getName(const std::string &name) const;

    virtual vrui::vruiTexture *createTexture(const std::string &textureName);

    virtual vrui::coUpdateManager *getUpdateManager();
    virtual vrui::coJoystickManager *getJoystickManager();

    virtual vrui::vruiActionUserData *createActionUserData(vrui::coAction *action);
    virtual vrui::vruiUserData *createUserData();
    virtual void deleteUserData(vrui::vruiUserData *userData);

    virtual vrui::coAction::Result hit(vrui::coAction *action, vrui::vruiHit *hit);
    virtual void miss(vrui::coAction *action);

    // remove pointer indicator
    virtual void removePointerIcon(const std::string &name);
    // add    pointer indicator
    virtual void addPointerIcon(const std::string &name);

    virtual vrui::vruiNode *getIcon(const std::string &iconName, bool shared = false);

    virtual vrui::vruiMatrix *getViewerMatrix() const;
    virtual vrui::vruiMatrix *getHandMatrix() const;
    virtual vrui::vruiMatrix *getMouseMatrix() const;

    virtual bool is2DInputDevice() const;
    virtual bool isMultiTouchDevice() const;

    virtual void sendCollabMessage(vrui::vruiCollabInterface *myinterface, const char *buffer, int length);

    virtual double getFrameTime() const;

    virtual void remoteLock(int);
    virtual void remoteUnLock(int);
    virtual bool isLocked(int);
    virtual bool isLockedByMe(int);

private:
    vrui::OSGVruiNode *groupNode;
    vrui::OSGVruiNode *sceneNode;
    vrui::OSGVruiMatrix *handMatrix;
    vrui::OSGVruiMatrix *headMatrix;
    vrui::OSGVruiMatrix *mouseMatrix;

    std::stack<vrui::vruiMatrix *> matrixStack;
    std::string look;
};
}
#endif
