/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2007 HLRS  **
 **                                                                        **
 ** Description: VNC Client Window                                         **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** History:                                                               **
 ** Oct-12 2007  v1                                                        **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

#include "../RemoteDT.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <cover/VRPinboard.h>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <OpenVRUI/coTextureRectBackground.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <cover/coVRMSController.h>
#include <sys/types.h>
#ifndef WIN32
#include <sys/ioctl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

#ifdef _WIN32
// for _mktemp_s
#include <io.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include "window.h"
#include "vncclient.hpp"
#include "vncauth.hpp"
#include "../RemoteDTActor.h"

VNCWindow::VNCWindow()
{
    server = 0;
    port = 0;
    password = 0;

    connected = false;

    framebuffer = 0;
    screenWidth = 0;
    screenHeight = 0;
    fbWidth = 0;
    fbHeight = 0;
    Bpp = 3;

    vncClient = 0;
    eventHandler = new RemoteDTActor(this);
    popupHandle = 0;
    desktopTexture = 0;

    lastMoveX = -1;
    lastMoveY = -1;
    lastClickX = -1;
    lastClickY = -1;
}

VNCWindow::~VNCWindow()
{
    if (vncClient)
    {
        if (connected)
        {
            if (!vncClient->VNCClose())
            {
                fprintf(stderr, "VNCWindow::~VNCWindow() err: Disconnecting failed!\n");
            }
        }
        delete vncClient;
    }

    if (eventHandler)
    {
        delete eventHandler;
    }

    if (popupHandle)
    {
        delete popupHandle;
    }

    if (desktopTexture)
    {
        delete desktopTexture;
    }

} // VNCWindow::~VNCWindow

/// WIN32-code is unchecked!
bool VNCWindow::makeConnection()
{
    fprintf(stderr,
            "VNCWindow::makeConnection(): Trying to connect to %s:%d\n", server, port);

    // connect to vnc server
    vncClient = new VNCClient(server, port, password);

    if (vncClient && !vncClient->VNCInit())
    {
        fprintf(stderr, "VNCWindow::makeConnection() err: VNC Init failed!\n");
        return false;
    }
    else
    {
        fprintf(stderr, "VNCWindow::makeConnection(): VNC Init successfull!\n");

        vncClient->sendFramebufferUpdateRequest(0, 0,
                                                vncClient->fbWidth,
                                                vncClient->fbHeight,
                                                false);
        framebuffer = (uint8_t *)vncClient->framebuffer;

        screenWidth = vncClient->fbWidth;
        screenHeight = vncClient->fbHeight;

        fbWidth = vncClient->fbWidth;
        fbHeight = vncClient->fbHeight;

        fprintf(stderr, "VNCWindow::makeConnection(): framebuffer: %d x %d, real screen %d x %d\n", fbWidth, fbHeight, screenWidth, screenHeight);

        vncClient->setUpdateQueue(&updateQueue);
    }

    return true;
}

bool VNCWindow::connectToServer(const char *server, unsigned port, const char *password)
{
    this->server = server;
    this->port = port;
    this->password = password;

    if (coVRMSController::instance()->isMaster())
    {
        connected = makeConnection();

        //fprintf(stderr, "Sending framebuffer information from master to slaves\n");

        coVRMSController::instance()->sendSlaves((char *)&connected, sizeof(bool));
        coVRMSController::instance()->sendSlaves((char *)&screenWidth, sizeof(screenWidth));
        coVRMSController::instance()->sendSlaves((char *)&screenHeight, sizeof(screenHeight));
        coVRMSController::instance()->sendSlaves((char *)&fbWidth, sizeof(fbWidth));
        coVRMSController::instance()->sendSlaves((char *)&fbHeight, sizeof(fbHeight));

        if (connected)
        {
            backbuffer = new char[fbWidth * fbHeight * Bpp];
        }
    }
    else
    {
        //fprintf(stderr, "Reading framebuffer information from master\n");

        coVRMSController::instance()->readMaster((char *)&connected, sizeof(bool));
        coVRMSController::instance()->readMaster((char *)&screenWidth, sizeof(screenWidth));
        coVRMSController::instance()->readMaster((char *)&screenHeight, sizeof(screenHeight));
        coVRMSController::instance()->readMaster((char *)&fbWidth, sizeof(fbWidth));
        coVRMSController::instance()->readMaster((char *)&fbHeight, sizeof(fbHeight));

        if (connected)
        {
            framebuffer = new unsigned char[fbWidth * fbHeight * Bpp];
            backbuffer = new char[fbWidth * fbHeight * Bpp];
        }
    }

    if (!connected)
    {
        fprintf(stderr, "VNCWindow::connectToServer() err: Connection failed, aborting!\n");
        return false;
    }

    desktopTexture = new coTextureRectBackground((uint *)framebuffer, Bpp, fbWidth, fbHeight, 0, eventHandler);

    desktopTexture->setSize(screenWidth, screenHeight, 0);
    desktopTexture->setTexSize(screenWidth, screenHeight);
    desktopTexture->setMinWidth(screenWidth);
    desktopTexture->setMinHeight(screenHeight);
    desktopTexture->setRepeat(true);

    char buf[1000];
    sprintf(buf, "VNC");
    popupHandle = new coPopupHandle(buf);
    //popupHandle->setScale(cover->getSceneSize()/2500);
    //popupHandle->setPos(-1300*cover->getSceneSize()/2500,0,-1200*cover->getSceneSize()/2500);
    popupHandle->addElement(desktopTexture);

    this->show();

    return true;
} // connectToServer

void VNCWindow::disconnectFromServer()
{
    if (vncClient)
    {
        if (connected)
        {
            if (!vncClient->VNCClose())
            {
                fprintf(stderr, "VNCWindow::~VNCWindow() err: Disconnecting failed!");
            }
        }
        delete vncClient;
        vncClient = 0;
    }
} // disconnectFromServer

bool VNCWindow::pollServer()
{
    // call VNC client status...
    bool updated = false;
    if (connected && vncClient)
    {
        fd_set rmask;
        struct timeval delay;
        int rfbsock = vncClient->getSock();
        delay.tv_sec = 0;
        delay.tv_usec = 10; // was 10
        FD_ZERO(&rmask);
        FD_SET(rfbsock, &rmask);

        if (select(rfbsock + 1, &rmask, NULL, NULL, &delay))
        {
            //fillRed((uint*)framebuffer, fbWidth, fbHeight);
            updated = true;
            if (FD_ISSET(rfbsock, &rmask))
            {
                if (!vncClient->handleRFBServerMessage())
                {
                    // TODO: Better error handling ?
                    fprintf(stderr, "VNCWindow::pollServer() err: can't handle RFB server message\n");
                    return false;
                }
            }
        }
    }
    return updated;
} // VNCWindow::pollServer()

void VNCWindow::update()
{
    if (desktopTexture && popupHandle)
    {
        bool updated = false;
        char c = 1;
        popupHandle->update();

        if (coVRMSController::instance()->isMaster())
        {
            updated = pollServer();
        }

        /*
         the update queue contains all rectangles that have been updated

         first we send the number of updates to the cluster (size of queue),
         then we send the x- and y- offsets, the width and height of the update rectangle
         finally we send the update data, which is then written to the slaves' own
         vnc-framebuffer
       */
        if (coVRMSController::instance()->isCluster())
        {
            if (coVRMSController::instance()->isMaster())
            {
                if (updated)
                {
                    // send the queue size!
                    c = updateQueue.size();
                    //fprintf(stderr, "Queue size: %d\n", (int)c);
                    coVRMSController::instance()->sendSlaves(&c, 1);

                    // now send all rectangle updates...
                    while (!updateQueue.empty())
                    {
                        rfbRectangle r = updateQueue.front();
                        updateQueue.pop();

                        // xoffset, yoffset, width and height, then data
                        coVRMSController::instance()->sendSlaves((char *)&r.x, sizeof(r.x));
                        coVRMSController::instance()->sendSlaves((char *)&r.y, sizeof(r.y));
                        coVRMSController::instance()->sendSlaves((char *)&r.w, sizeof(r.w));
                        coVRMSController::instance()->sendSlaves((char *)&r.h, sizeof(r.h));

                        int sz = 3 * (int)r.w * (int)r.h;
                        int it = 0;

                        for (int iy = r.y; iy < r.y + r.h; ++iy)
                        {
                            for (int ix = r.x; ix < r.x + r.w; ++ix)
                            {
                                backbuffer[it++] = ((char *)framebuffer)[3 * iy * fbWidth + 3 * ix];
                                backbuffer[it++] = ((char *)framebuffer)[3 * iy * fbWidth + 3 * ix + 1];
                                backbuffer[it++] = ((char *)framebuffer)[3 * iy * fbWidth + 3 * ix + 2];
                            }
                        }

                        coVRMSController::instance()->sendSlaves(backbuffer, sz);

                    } // while queue not empty
                }
                else
                {
                    c = 0;
                    coVRMSController::instance()->sendSlaves(&c, 1);
                }
            }
            else
            {
                coVRMSController::instance()->readMaster(&c, 1);

                updated = c != 0;

                for (int i = 0; i < c; ++i)
                {
                    rfbRectangle r;
                    coVRMSController::instance()->readMaster((char *)&r.x, sizeof(r.x));
                    coVRMSController::instance()->readMaster((char *)&r.y, sizeof(r.y));
                    coVRMSController::instance()->readMaster((char *)&r.w, sizeof(r.w));
                    coVRMSController::instance()->readMaster((char *)&r.h, sizeof(r.h));

                    const int sz = 3 * (int)r.w * (int)r.h;
                    coVRMSController::instance()->readMaster(backbuffer, sz);

                    //write data to framebuffer!
                    int it = 0;
                    for (int iy = r.y; iy < r.y + r.h; ++iy)
                    {
                        for (int ix = r.x; ix < r.x + r.w; ++ix)
                        {
                            ((char *)framebuffer)[3 * iy * fbWidth + 3 * ix] = backbuffer[it++];
                            ((char *)framebuffer)[3 * iy * fbWidth + 3 * ix + 1] = backbuffer[it++];
                            ((char *)framebuffer)[3 * iy * fbWidth + 3 * ix + 2] = backbuffer[it++];
                        }
                    }
                } // for all rectangle updates!
            }
        }

        if (updated)
        {
            desktopTexture->setUpdated(true);
            desktopTexture->setImage((uint *)framebuffer, Bpp, fbWidth, fbHeight, 0);
        }
    }
} // VNCWindow::update()

bool VNCWindow::isConnected() const
{
    return connected;
}

int VNCWindow::getScreenWidth() const
{
    return vncClient ? screenWidth : 0;
}

int VNCWindow::getScreenHeight() const
{
    return vncClient ? screenHeight : 1;
}

int VNCWindow::getTextureWidth() const
{
    return vncClient ? fbWidth : 0;
}

int VNCWindow::getTextureHeight() const
{
    return vncClient ? fbHeight : 1;
}

void VNCWindow::show()
{
    if (popupHandle)
        popupHandle->setVisible(true);
}

void VNCWindow::hide()
{
    // disconnect?
    if (popupHandle)
        popupHandle->setVisible(false);
}

void *VNCWindow::getFramebuffer() const
{
    return framebuffer;
}

float VNCWindow::transformX(float x) const
{
    float scale = (float)(this->getScreenWidth()) / (float)(this->getTextureWidth());
    return x * this->getScreenWidth() / scale;
}

float VNCWindow::transformY(float y) const
{
    // if screen is 1024*768, and texture is 1024*1024
    // then we need to multiply y with a scaling factor...
    float scale = (float)(this->getScreenHeight()) / (float)(this->getTextureHeight());

    return (1.0f - y) * this->getScreenHeight() / scale;
}

/////////////////////////////////////////////////////////
//
// event handling
//
/////////////////////////////////////////////////////////

RemoteDTActor *VNCWindow::getActor() const
{
    return eventHandler;
}

void VNCWindow::mouseButtonPressed(float x, float y)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (connected)
        {
            if (lastClickX == x && lastClickY == y)
                return;

            lastClickX = x;
            lastClickY = y;

            // transform texture coordinates (x, y) to desktop coordinates
            //fprintf(stderr, "Click: %f, %f\n", x, y);
            int16_t sx = (int16_t)(transformX(x));
            int16_t sy = (int16_t)(transformY(y));
            uint8_t button = 1;

            uint32_t NUM = 4;
            const char *params[4 /*NUM*/];
            params[0] = "ptr";
            char p1[16], p2[16], p3[16];

            sprintf(p1, "%d", sx);
            params[1] = p1;
            sprintf(p2, "%d", sy);
            params[2] = p2;
            sprintf(p3, "%d", button);
            params[3] = p3;

            vncClient->sendRFBEvent((char **)params, (uint32_t *)&NUM);
        }
    }
} // VNCWindow::mouseButtonPressed

void VNCWindow::mouseButtonReleased(float x, float y)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (connected)
        {
            // transform texture coordinates (x, y) to desktop coordinates
            //fprintf(stderr, "Released: %f, %f\n", x, y);
            int16_t sx = (int16_t)(transformX(x));
            int16_t sy = (int16_t)(transformY(y));
            uint8_t button = 0;

            uint32_t NUM = 4;
            const char *params[4 /*NUM*/];
            params[0] = "ptr";
            char p1[16], p2[16], p3[16];

            sprintf(p1, "%d", sx);
            params[1] = p1;
            sprintf(p2, "%d", sy);
            params[2] = p2;
            sprintf(p3, "%d", button);
            params[3] = p3;

            vncClient->sendRFBEvent((char **)params, (uint32_t *)&NUM);
        }
    }
}

void VNCWindow::mouseMoved(float x, float y)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (connected)
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

            uint32_t NUM = 4;
            const char *params[4 /*NUM*/];
            params[0] = "ptr";
            char p1[16], p2[16], p3[16];

            sprintf(p1, "%d", sx);
            params[1] = p1;
            sprintf(p2, "%d", sy);
            params[2] = p2;
            sprintf(p3, "%d", button);
            params[3] = p3;

            vncClient->sendRFBEvent((char **)params, (uint32_t *)&NUM); // send to VNC
        }
    }
} // VNCWindow::mouseMoved

void VNCWindow::mouseDragged(float x, float y)
{
    if (coVRMSController::instance()->isMaster())
    {
        //x = x;
        //y = y;
        mouseButtonPressed(x, y);
    }
}

void VNCWindow::keyPressed(int keysym, int /* mod */)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (connected)
        {
            uint32_t NUM = 2;
            const char *params[2 /*NUM*/];
            params[0] = "keydown";
            params[1] = (char *)&keysym;

            vncClient->sendRFBEvent((char **)params, (uint32_t *)&NUM);
        }
    }
} // VNCWindow::keypressed

void VNCWindow::keyReleased(int keysym, int /* mod */)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (connected)
        {
            uint32_t NUM = 2;
            const char *params[2 /*NUM*/];
            params[0] = "keyup";
            params[1] = (char *)&keysym;

            vncClient->sendRFBEvent((char **)params, (uint32_t *)&NUM);
        }
    }
} // VNCWindow::keyReleased

void VNCWindow::sendText(const char *text)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (connected)
        {
            int len = strlen(text);
            //fprintf(stderr, "Sending: %s\n", text);
            uint32_t NUM = 2;
            const char *params[2 /*NUM*/];

            for (int i = 0; i < len; ++i)
            {
                params[0] = "keydown";
                int c = text[i];
                params[1] = (char *)&c;
                vncClient->sendRFBEvent((char **)params, (uint32_t *)&NUM);
            }
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
