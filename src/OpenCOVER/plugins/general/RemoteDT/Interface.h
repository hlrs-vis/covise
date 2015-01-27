/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _REMOTEDT_INTERFACE_H
#define _REMOTEDT_INTERFACE_H

class RemoteDTActor;

/*!
 * IRemoteDesktop contains the interface, which any remote
 * tool needs to implement
 */
class IRemoteDesktop
{
public:
    virtual ~IRemoteDesktop(){}; // segfault when pure virtual

    /// connects to the remote server
    virtual bool connectToServer(const char *server, unsigned port, const char *password) = 0;

    /// disconnects from the remote server
    virtual void disconnectFromServer() = 0;

    /// updates the texture/UI elements
    virtual void update() = 0;

    /// shows this window
    virtual void show() = 0;

    /// hides this window
    virtual void hide() = 0;

    /// returns whether a connection to a server exists
    virtual bool isConnected() const = 0;

    /// returns the screen width of the remote server
    virtual int getScreenWidth() const = 0;

    /// returns the screen height of the remote server
    virtual int getScreenHeight() const = 0;

    /// returns the texture width of the internal buffer
    virtual int getTextureWidth() const = 0;

    /// returns the texture height of the internal buffer
    virtual int getTextureHeight() const = 0;

    /// returns the pointer to the frame buffer
    virtual void *getFramebuffer() const = 0;

    /// transforms x from 0..1 to screen coordinate
    virtual float transformX(float x) const = 0;

    /// transforms y from 0..1 to screen coordinate
    virtual float transformY(float y) const = 0;

    /// returns the event handler of this object
    virtual RemoteDTActor *getActor() const = 0;

    /// mouse button pressed
    virtual void mouseButtonPressed(float x, float y) = 0;

    /// mouse button released
    virtual void mouseButtonReleased(float x, float y) = 0;

    /// mouse moved
    virtual void mouseMoved(float x, float y) = 0;

    /// mouse moved with button pressed
    virtual void mouseDragged(float x, float y) = 0;

    /// keyboard key pressed
    virtual void keyPressed(int keysym, int mod) = 0;

    /// keyboard key released
    virtual void keyReleased(int keysym, int mod) = 0;

    /// send text
    virtual void sendText(const char *text) = 0;

    /// send RETURN key event
    virtual void sendReturn() = 0;

    /// send CTRL+C (abort) key event
    virtual void sendCTRL_C() = 0;

    /// send CTRL+D (end of file) key event
    virtual void sendCTRL_D() = 0;

    /// send CTRL+Z (suspend process) key event
    virtual void sendCTRL_Z() = 0;

    /// send ESCAPE key event
    virtual void sendESCAPE() = 0;

}; // IRemoteDesktop

#endif // _REMOTEDT_INTERFACE_H
