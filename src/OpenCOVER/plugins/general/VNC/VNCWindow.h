/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VNCWINDOW_H_
#define VNCWINDOW_H_

#include <queue>
#include <string>

#include "vnc/rfbproto.h"

class VNCClient;
class VNCWindowActor;
class VNCWindowBuffer;

namespace vrui
{
class coPopupHandle;
class coTextureRectBackground;
}

class VNCWindow
{
public:
    typedef std::queue<rfbRectangle> UpdateQueue;

    VNCWindow(const std::string &serverId = "", VNCClient *client = 0, VNCWindowBuffer *windowBuffer = 0);
    virtual ~VNCWindow();

    void update();
    void show();
    void hide();

    int getScreenWidth() const;
    int getScreenHeight() const;
    int getTextureWidth() const;
    int getTextureHeight() const;

    float transformX(float x) const;
    float transformY(float y) const;

    VNCWindowActor *getActor() const;
    void *getFramebuffer() const;

    void mouseButtonPressed(float x, float y);
    void mouseButtonReleased(float x, float y);
    void mouseMoved(float x, float y);
    void mouseDragged(float x, float y);
    void keyPressed(int keysym, int mod);
    void keyReleased(int keysym, int mod);
    void sendText(const char *text);
    void sendReturn();
    void sendCTRL_C();
    void sendCTRL_D();
    void sendCTRL_Z();
    void sendESCAPE();

    VNCWindowBuffer *createBuffer(int x, int y);
    VNCWindowBuffer *getBuffer()
    {
        return windowBuffer;
    }

private:
    void init();

    vrui::coTextureRectBackground *desktopTexture; ///< widget onto which the remote desktop is drawn
    vrui::coPopupHandle *popupHandle;

    VNCWindowBuffer *windowBuffer; ///< stores pixel data for this window
    VNCClient *vncClient;
    VNCWindowActor *eventHandler;

    char *transferbuffer; ///< buffer for transferring updates from Master to Slaves

    float lastMoveX, lastMoveY;
    float lastClickX, lastClickY;

    std::string serverId;
}; // class VNCWindow

#endif /* VNCWINDOW_H_ */
