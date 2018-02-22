/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_RENDERERINTERFACE_H
#define VRUI_RENDERERINTERFACE_H

#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/sginterface/vruiMatrix.h>
#include <OpenVRUI/sginterface/vruiTexture.h>
#include <OpenVRUI/sginterface/vruiActionUserData.h>
#include <OpenVRUI/sginterface/vruiCollabInterface.h>

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coJoystickManager.h>

#include <string>
#include <map>

EXPORT_TEMPLATE2(template class OPENVRUIEXPORT std::map<std::string, vrui::vruiNode *>)

namespace vrui
{

class vruiButtonProvider;
class vruiPanelGeometryProvider;
class vruiUIElementProvider;

class vruiActionUserData;
class vruiButtons;
class vruiCollabInterface;
class vruiHit;
class vruiNode;
class vruiUserData;

class coButtonGeometry;
class coPanelGeometry;
class coUIElement;
class coUpdateManager;

class OPENVRUIEXPORT vruiRendererInterface
{

public:
    vruiRendererInterface();
    virtual ~vruiRendererInterface();

    virtual vruiNode *getMenuGroup() = 0;

    virtual vruiNode *getScene()
    {
        return NULL;
    }

    virtual vruiUIElementProvider *createUIElementProvider(coUIElement *element) = 0;
    virtual vruiButtonProvider *createButtonProvider(coButtonGeometry *button) = 0;
    virtual vruiPanelGeometryProvider *createPanelGeometryProvider(coPanelGeometry *panel) = 0;

    virtual vruiTransformNode *createTransformNode() = 0;
    virtual void deleteNode(vruiNode *node)
    {
        delete node;
    }

    virtual vruiMatrix *createMatrix() = 0;
    virtual void deleteMatrix(vruiMatrix *matrix)
    {
        delete matrix;
    }

    virtual vruiActionUserData *createActionUserData(coAction *action) = 0;
    virtual vruiUserData *createUserData() = 0;
    virtual void deleteUserData(vruiUserData *userData)
    {
        delete userData;
    }

    virtual vruiTexture *createTexture(const std::string &textureName) = 0;
    virtual void deleteTexture(vruiTexture *texture)
    {
        delete texture;
    }

    virtual vruiButtons *getButtons() const;
    virtual vruiButtons *getMouseButtons() const;
    virtual vruiButtons *getRelativeButtons() const;

    virtual double getFrameTime() const = 0; //{ return currentFrameTime; }
    //inline void setFrameTime(double t) { currentFrameTime = t; } // set frame start time by Renderer

    virtual std::string getName(const std::string &name) const = 0;

    virtual coUpdateManager *getUpdateManager() = 0;

    /// the application can have a joystick manager
    virtual coJoystickManager *getJoystickManager()
    {
        return NULL;
    }
    /// is menu pickable via ray
    virtual bool isRayActive()
    {
        return ray;
    }
    /// set menu pickable via ray
    virtual void setRayActive(bool b)
    {
        ray = b;
    }
    /// is menu selectable vie joystick
    virtual bool isJoystickActive();
    /// set menu selectable via joystick
    virtual void setJoystickActvie(bool b);

    virtual coAction::Result hit(coAction *action, vruiHit *hit) = 0;
    virtual void miss(coAction *action) = 0;

    // remove pointer indicator
    virtual void removePointerIcon(const std::string &name) = 0;
    // add    pointer indicator
    virtual void addPointerIcon(const std::string &name) = 0;

    virtual vruiNode *getIcon(const std::string &iconName, bool shared = false) = 0;

    virtual vruiMatrix *getViewerMatrix() const = 0;
    virtual vruiMatrix *getHandMatrix() const = 0;
    virtual vruiMatrix *getMouseMatrix() const = 0;
    virtual vruiMatrix *getRelativeMatrix() const = 0;

    virtual bool is2DInputDevice() const = 0;
    virtual bool isMultiTouchDevice() const
    {
        return false;
    }

    virtual void sendCollabMessage(vruiCollabInterface *myinterface, const char *buffer, int length) = 0;
    virtual void remoteLock(int)
    {
    }
    virtual void remoteUnLock(int)
    {
    }
    virtual bool isLocked(int)
    {
        return false;
    }
    virtual bool isLockedByMe(int)
    {
        return false;
    }

    static vruiRendererInterface *the();

    /// set the sensitivity oft thre rowMenuHandle interation scale
    void setInteractionScaleSensitivity(float f);
    /// get the sensitivity oft thre rowMenuHandle interation scale
    float getInteractionScaleSensitivity()
    {
        return interactionScaleSensitivity;
    }

    void setUpVector(coVector v)
    {
        upVector = v;
    }
    coVector getUpVector()
    {
        return upVector;
    }

    /* needed for RTT   
      virtual vruiMatrix *doBillboarding(vruiMatrix *invStartHandTrans, coVector pickPosition, coVector localPickPosition, float myScale) = 0;*/

protected:
    vruiButtons *buttons = nullptr;
    vruiButtons *mouseButtons = nullptr;
    vruiButtons *relativeButtons = nullptr;

    std::map<std::string, vruiNode *> iconsList;

    /// Sensitivity of the rowMenuHandle Interaction scale
    float interactionScaleSensitivity;

    /// Upvector
    coVector upVector;

    static vruiRendererInterface *theInterface;

    bool ray = false;
};
}
#endif
