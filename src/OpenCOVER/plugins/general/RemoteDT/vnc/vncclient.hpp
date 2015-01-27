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

#include <queue>
#include "rfbproto.hpp"

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
    char rfbbuffer[RFB_BUF_SIZE];
    ///< this buffer is used to get Server MSGs

    bool viewonly;
    ///< do we want to forward events? (no if true)

    void fillRect(int rx, int ry, int rw, int rh, VNCRGB pixel);
    ///< handling Rectangles we got from server

    CARD8 rescalePixValue(CARD32 Pix, CARD8 Shift, CARD16 Max);
    VNCRGB cardToVNCRGB(CARD32 Pixel);

    bool handleRAW32(int rx, int ry, int rw, int rh);
    bool handleCR(int srcx, int srcy, int rx, int ry, int rw, int rh);
    bool handleRRE32(int rx, int ry, int rw, int rh);
    bool handleCoRRE32(int rx, int ry, int rw, int rh);
    bool handleHextile32(int rx, int ry, int rw, int rh);

protected:
    VNCRFBproto rfbproto;
    /**< the VNCRFBproto object implements all we need to to send and receive
   * messages specified in the RFB protocol
   */

    std::queue<rfbRectangle> *updateQueue;

public:
    VNCClient(const char *server, int port, const char *passFile);
    ///< constructor

    char *serverCutText;
    bool newServerCutText;
    ///< for now, public attributes we keep to handle server cut texts

    VNCRGB *framebuffer;
    CARD16 fbWidth;
    CARD16 fbHeight;
    ///< here is our framebuffer

    int getSock();
    ///< we might want to get the socket descriptor to listen to it

    void setUpdateQueue(std::queue<rfbRectangle> *uq)
    {
        updateQueue = uq;
    }

    // set of functions to initialize the connection
    enum
    {
        BPP8,
        BPP16,
        BPP32
    };

    bool VNCInit();
#if 0 //not used
  bool VNCInit(int flag);
  ///< flag can be BPP8, BPP16 or BPP16 to indicate how many bits per pixel we want
#endif
    ///< VNCInit has to allocate memory for the framebuffer if everything goes fine
    bool VNCClose();
    ///< VNCClose has to free it

    /* Remote Frame Buffer Protocol v3.3 */

    /**
   sendRFBEvent should be replaced by sendKeyEvent and sendPointerEvent!
   */
    void sendRFBEvent(char **params, unsigned int *num_params);
    bool sendIncrementalFramebufferUpdateRequest();
    bool sendFramebufferUpdateRequest(int x, int y, int w, int h, bool incremental);
    ///< messages from client to server

    bool handleRFBServerMessage();
    ///< handle the messages from the server
};

/**
 * class VNCClientTextured
 * inherits everything from VNCClient
 * the framebuffer allocated is bigger than the real VNC server screen size
 * and has dimensions that are a power of 2
 * needed by OpenGL to map the texture
 */
class VNCClientTextured : public VNCClient
{
public:
    VNCClientTextured(const char *server, int port, const char *pass);

    CARD16 realScreenWidth;
    CARD16 realScreenHeight;
    ///< here we keep what the real server screen size is

    bool VNCInit();
};

#endif // VNCCLIENT_HPP
