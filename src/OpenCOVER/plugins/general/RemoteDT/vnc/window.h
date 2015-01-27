/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VNC_WINDOW_H
#define _VNC_WINDOW_H

#include "rfbproto.hpp"
#include <queue>

#include "../Interface.h"

class VNCClient;
class VNCClientTextured;
class RemoteDTActor;

namespace vrui
{
class coPopupHandle;
class coTexturedBackground;
class coTextureRectBackground;
}

class VNCWindow : virtual public IRemoteDesktop
{
public:
    typedef std::queue<rfbRectangle> UpdateQueue;

    VNCWindow();
    virtual ~VNCWindow();

    virtual bool connectToServer(const char *server, unsigned port, const char *password);
    virtual void disconnectFromServer();

    virtual void update();
    virtual void show();
    virtual void hide();

    virtual bool isConnected() const;
    virtual int getScreenWidth() const;
    virtual int getScreenHeight() const;
    virtual int getTextureWidth() const;
    virtual int getTextureHeight() const;
    virtual void *getFramebuffer() const;
    virtual float transformX(float x) const;
    virtual float transformY(float y) const;

    virtual RemoteDTActor *getActor() const;

    virtual void mouseButtonPressed(float x, float y);
    virtual void mouseButtonReleased(float x, float y);
    virtual void mouseMoved(float x, float y);
    virtual void mouseDragged(float x, float y);
    virtual void keyPressed(int keysym, int mod);
    virtual void keyReleased(int keysym, int mod);
    virtual void sendText(const char *text);
    virtual void sendReturn();
    virtual void sendCTRL_C();
    virtual void sendCTRL_D();
    virtual void sendCTRL_Z();
    virtual void sendESCAPE();

protected:
    /** 
    * called by connectToServer on the master
    * saves password into a temporary (encrypted) file
    * connects to a server, authenticates and receives
    * the framebuffer
    */
    bool makeConnection();

    const char *server; ///< server (IP-address), ex.: '127.0.0.1'
    unsigned port; ///< port of the server
    const char *password; ///< password to the server, will be cleared after connection try

    bool connected; ///< whether a connection exists

    unsigned char *framebuffer; ///< pointer to VNCClientTextured's framebuffer (guaranteed to be power of 2)
    int screenWidth; ///< the remote server' screen width
    int screenHeight; ///< the remote server' screen height
    int fbWidth; ///< framebuffer width
    int fbHeight; ///< framebuffer height
    int Bpp; ///< framebuffer Bytes per pixel

    //VNCClientTextured* vncClient;
    VNCClient *vncClient;
    RemoteDTActor *eventHandler;
    vrui::coPopupHandle *popupHandle;
    //coTexturedBackground* desktopTexture; ///< widget onto which the remote desktop is drawn
    vrui::coTextureRectBackground *desktopTexture; ///< widget onto which the remote desktop is drawn

    bool pollServer(); ///< checks for updates on the remote server

    UpdateQueue updateQueue; ///< contains a list of dirty rectangles

    char *backbuffer; ///< backbuffer for updating the framebuffer

private:
    float lastMoveX, lastMoveY;
    float lastClickX, lastClickY;

}; // class VNCWindow

#endif // _VNC_WINDOW_H
