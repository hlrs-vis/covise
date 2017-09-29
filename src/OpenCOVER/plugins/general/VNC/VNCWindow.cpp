/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2009 HLRS  **
 **                                                                        **
 ** Description: VNC Window Class                                          **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** Created on: Nov 17, 2008                                               **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

#include "VNCWindow.h"

#include <cover/coVRMSController.h>

#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coTextureRectBackground.h>

#include "VNCWindowActor.h"
#include "vnc/WindowBuffer.h"
#include "vnc/vncclient.h"

using opencover::coVRMSController;
using vrui::coTextureRectBackground;

VNCWindow::VNCWindow(const std::string &serverId, VNCClient *client, VNCWindowBuffer *windowBuffer)
    : windowBuffer(windowBuffer)
    , vncClient(client)
    , serverId(serverId)
{
    eventHandler = new VNCWindowActor(this);
    desktopTexture = 0;
    popupHandle = 0;

    lastMoveX = -1;
    lastMoveY = -1;
    lastClickX = -1;
    lastClickY = -1;

    init();
    show();
}

VNCWindow::~VNCWindow()
{
    delete desktopTexture;
    delete eventHandler;
    delete popupHandle;
} // VNCWindow::~VNCWindow

void VNCWindow::update()
{
    if (desktopTexture)
    {
        popupHandle->update();

        if (windowBuffer->synchronize())
        {
            desktopTexture->setUpdated(true);
            desktopTexture->setImage((uint *)windowBuffer->getFramebuffer(), 3, windowBuffer->getWidth(),
                                     windowBuffer->getHeight(), 0);
        }
    }
} // VNCWindow::update()

void VNCWindow::show()
{
    if (popupHandle)
        popupHandle->setVisible(true);
}

void VNCWindow::hide()
{
    if (popupHandle)
        popupHandle->setVisible(false);
}

void *VNCWindow::getFramebuffer() const
{
    return windowBuffer->getFramebuffer();
}

float VNCWindow::transformX(float x) const
{
    float scale = (float)(this->getScreenWidth())
                  / (float)(this->getTextureWidth());
    return x * this->getScreenWidth() / scale;
}

float VNCWindow::transformY(float y) const
{
    // if screen is 1024*768, and texture is 1024*1024
    // then we need to multiply y with a scaling factor...
    float scale = (float)(this->getScreenHeight())
                  / (float)(this->getTextureHeight());

    return (1.0f - y) * this->getScreenHeight() / scale;
}

/////////////////////////////////////////////////////////
//
// event handling
//
/////////////////////////////////////////////////////////

VNCWindowActor *VNCWindow::getActor() const
{
    return eventHandler;
}

void VNCWindow::mouseButtonPressed(float x, float y)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (lastClickX == x && lastClickY == y)
            return;

        lastClickX = x;
        lastClickY = y;

        // transform texture coordinates (x, y) to desktop coordinates
        //fprintf(stderr, "Click: %f, %f\n", x, y);
        const int16_t sx = (int16_t)(transformX(x));
        const int16_t sy = (int16_t)(transformY(y));
        const uint8_t button = 1;

        vncClient->sendPointerEvent(sx, sy, button);
    }
} // VNCWindow::mouseButtonPressed

void VNCWindow::mouseButtonReleased(float x, float y)
{
    if (coVRMSController::instance()->isMaster())
    {
        // transform texture coordinates (x, y) to desktop coordinates
        //fprintf(stderr, "Released: %f, %f\n", x, y);
        const int16_t sx = (int16_t)(transformX(x));
        const int16_t sy = (int16_t)(transformY(y));
        const uint8_t button = 0;

        vncClient->sendPointerEvent(sx, sy, button);
    }
}

void VNCWindow::mouseMoved(float x, float y)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (lastMoveX == x && lastMoveY == y)
            return;

        lastMoveX = x;
        lastMoveY = y;

        // transform x and y to screen coordinates
        //fprintf(stderr, "Moved: %f, %f\n", x, y);
        int16_t sx = (int16_t)(transformX(x));
        int16_t sy = (int16_t)(transformY(y));
        uint8_t button = 0;

        vncClient->sendPointerEvent(sx, sy, button);
    }
} // VNCWindow::mouseMoved

void VNCWindow::mouseDragged(float x, float y)
{
    mouseButtonPressed(x, y);
}

void VNCWindow::keyPressed(int keysym, int /* mod */)
{
    if (coVRMSController::instance()->isMaster())
    {
        vncClient->sendKeyDownEvent(keysym);
    }
} // VNCWindow::keypressed

void VNCWindow::keyReleased(int keysym, int /* mod */)
{
    if (coVRMSController::instance()->isMaster())
    {
        vncClient->sendKeyUpEvent(keysym);
    }
} // VNCWindow::keyReleased

void VNCWindow::sendText(const char *text)
{
    if (coVRMSController::instance()->isMaster())
    {
        size_t len = strlen(text);
        for (size_t i = 0; i < len; ++i)
        {
            vncClient->sendKeyPressEvent(text[i]);
        }
    }
} // VNCWindow::sendText

void VNCWindow::sendReturn()
{
    const int RETURN = 0xFF0D;
    keyPressed(RETURN, 0);
} // VNCWindow::sendReturn

void VNCWindow::sendCTRL_C()
{
    const int CTRL_C = 0x3;
    keyPressed(CTRL_C, 0);
}

void VNCWindow::sendCTRL_D()
{
    const int CTRL_D = 0x4;
    keyPressed(CTRL_D, 0);
}

void VNCWindow::sendCTRL_Z()
{
    const int CTRL_Z = 0x1A;
    keyPressed(CTRL_Z, 0);
}

void VNCWindow::sendESCAPE()
{
    const int ESC = 0xFF1B;
    keyPressed(ESC, 0);
}

int VNCWindow::getScreenWidth() const
{
    return windowBuffer->getWidth();
}

int VNCWindow::getScreenHeight() const
{
    return windowBuffer->getHeight();
}

int VNCWindow::getTextureWidth() const
{
    return getScreenWidth();
}

int VNCWindow::getTextureHeight() const
{
    return getScreenHeight();
}

VNCWindowBuffer *VNCWindow::createBuffer(int x, int y)
{
    windowBuffer = new VNCWindowBuffer(x, y, 0);
    return windowBuffer;
}

void VNCWindow::init()
{
    int width;
    int height;

    if (coVRMSController::instance()->isMaster())
    {
        width = vncClient->getFBWidth();
        height = vncClient->getFBHeight();

        coVRMSController::instance()->sendSlaves((void *)&width, sizeof(int));
        coVRMSController::instance()->sendSlaves((void *)&height, sizeof(int));

        int serverNameLength = (int)serverId.length() + 1;
        coVRMSController::instance()->sendSlaves((void *)&serverNameLength, sizeof(int));
        coVRMSController::instance()->sendSlaves((void *)serverId.c_str(), serverNameLength);
    }
    else
    {
        coVRMSController::instance()->readMaster((void *)&width, sizeof(int));
        coVRMSController::instance()->readMaster((void *)&height, sizeof(int));

        int serverNameLength = 0;
        coVRMSController::instance()->readMaster((void *)&serverNameLength, sizeof(int));
        char *tmpId = new char[serverNameLength];
        coVRMSController::instance()->readMaster((void *)tmpId, serverNameLength);
        serverId = tmpId;
        delete[] tmpId;

        createBuffer(width, height);
    }

    desktopTexture = new coTextureRectBackground(
        (uint *)windowBuffer->getFramebuffer(),
        3,
        width,
        height,
        0,
        eventHandler);

    desktopTexture->setSize((float)width, (float)height, 0);
    desktopTexture->setTexSize((float)width, (float)height);
    desktopTexture->setMinWidth((float)width);
    desktopTexture->setMinHeight((float)height);
    desktopTexture->setRepeat(true);

    popupHandle = new vrui::coPopupHandle(serverId.c_str());

    popupHandle->addElement(desktopTexture);
}
