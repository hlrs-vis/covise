/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//---------------------------------------------------------------------------
// VREng (Virtual Reality Engine)	http://vreng.enst.fr/
//
// Copyright (C) 1997-2007 Ecole Nationale Superieure des Telecommunications
//
// VREng is a free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public Licence as published by
// the Free Software Foundation; either version 2, or (at your option)
// any later version.
//
// VREng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//---------------------------------------------------------------------------
#ifndef VNCCLIENT_HPP
#define VNCCLIENT_HPP

#include <list>
#include <map>
#include "rfbproto.h"

class VNCWindowBuffer;

#define RFB_BUF_SIZE 640 * 480

/**
 * class VNCRGB
 * this is what our framebuffer is made of 24 bits per pixel
 * 8bits for Red, 8 for Green, 8 for Blue
 */
class VNCRGB
{
public:
    VNCRGB();
    ///< this constructor is only used to handle faster 32bits pixels we get from the server
    VNCRGB(const CARD32 &pixel);
    VNCRGB(CARD8 red, CARD8 green, CARD8 blue);

    CARD8 Red;
    CARD8 Green;
    CARD8 Blue;
};

/**
 * class VNCClient
 * object will be used in VReng:
 * - to connect to the VNC server
 * - to handle its messages
 * - to forward pointer and keyboard events to the server
 * framebuffer is public
 */
class VNCClient
{

private:
    /// this buffer is used to get Server MSGs
    char rfbbuffer[RFB_BUF_SIZE];

    /// do we want to forward events? (no if true)
    bool viewonly;

    /// the server supports 'seamless windows'
    bool sharedAppSupport;

    /// handling Rectangles we got from server
    void fillRect(int rx, int ry, int rw, int rh, VNCRGB pixel);

    CARD8 rescalePixValue(CARD32 Pix, CARD8 Shift, CARD16 Max);
    VNCRGB cardToVNCRGB(CARD32 Pixel);

    bool handleRAW32(int rx, int ry, int rw, int rh);
    bool handleCR(int srcx, int srcy, int rx, int ry, int rw, int rh);
    bool handleRRE32(int rx, int ry, int rw, int rh);
    bool handleCoRRE32(int rx, int ry, int rw, int rh);
    bool handleHextile32(int rx, int ry, int rw, int rh);

    /**
    * the VNCRFBproto object implements all we need to to send and receive
    * messages specified in the RFB protocol
    */
    VNCRFBproto rfbproto;

    std::map<int, VNCWindowBuffer *> windows;

    /// list of fresh generated windows
    std::list<int> newWindowIds;

    /// here is our current framebuffer, width and height.
    VNCRGB *framebuffer;
    CARD16 fbWidth;
    CARD16 fbHeight;

public:
    /// constructor
    VNCClient(const char *server, int port, const char *passFile);

    void setWindowBuffer(VNCWindowBuffer *wb)
    {
        this->wb = wb;
    }

    /// for now, public attributes we keep to handle server cut texts
    char *serverCutText;
    bool newServerCutText;

    /// we might want to get the socket descriptor to listen to it
    int getSock();

    // set of functions to initialize the connection
    enum
    {
        BPP8,
        BPP16,
        BPP32
    };

    bool VNCInit();
#if 0 //not used
   /// flag can be BPP8, BPP16 or BPP16 to indicate how many bits per pixel we want
   bool VNCInit(int flag);
#endif

    /// VNCInit has to allocate memory for the framebuffer if everything goes fine
    /// VNCClose has to free it
    bool VNCClose();

    /* Remote Frame Buffer Protocol v3.3 */

    /// sends a key down event to remote host
    void sendKeyDownEvent(int keysym);

    /// sends a key up event to remote host
    void sendKeyUpEvent(int keysym);

    /// sends a key down/up event to remote host
    void sendKeyPressEvent(int keysyn);

    /// sends a pointer event to remote host
    void sendPointerEvent(int x, int y, int buttons);

    bool sendIncrementalFramebufferUpdateRequest();

    /// messages from client to server
    bool sendFramebufferUpdateRequest(int x, int y, int w, int h,
                                      bool incremental);

    /// handle the messages from the server
    bool handleRFBServerMessage();

    /// handle an update message for 'the' framebuffer
    bool handleFramebufferUpdateMessage();

    /// handle an update message for a specific 'seamless' window
    bool handleSharedAppUpdateMessage();

    /// handle all update rects from a server message
    bool handleRFBUpdateRects(int num);

    CARD16 getFBWidth() const
    {
        return fbWidth;
    }
    CARD16 getFBHeight() const
    {
        return fbHeight;
    }

    /// check for updates on the VNC server
    bool pollServer();

    /**
    * Returns an existing VNCWindowBuffer if ID already exists; returns a
    * fresh VNCWindowBuffer if ID didn't exist.
    * The whole desktop has always ID 0, shared apps have IDs != 0
    */
    VNCWindowBuffer *getWindow(int id);
    VNCWindowBuffer *wb;

    /**
    * closes window with specific ID (does nothing if ID does not exist)
    */
    void closeWindow(int id);

    /**
    * returns whether new windows were generated in the last update.
    */
    bool checkNewWindows() const;

    /**
    * returns a list of the IDs of the new windows
    */
    const std::list<int> &getNewWindowList() const;

    /**
    * cleares the new window list
    */
    void clearNewWindowList();
};

#endif // VNCCLIENT_HPP
