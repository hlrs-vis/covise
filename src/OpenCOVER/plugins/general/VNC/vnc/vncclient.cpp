/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2009 HLRS  **
 **                                                                        **
 ** Description: VNC Client                                                **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** based on vnclient.cpp from VREng                                       **
 **                                                                        **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

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

#include "vncclient.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#if !defined(_WIN32) && !defined(__APPLE__)
#include <unistd.h>
#include <X11/X.h>
#endif

#include "util/common.h"

#include "WindowBuffer.h"

//#define TRACE(x) x
#define TRACE(x)

/*
 * Constructors
 */
VNCRGB::VNCRGB()
{
    Red = 0;
    Green = 0;
    Blue = 0;
}

VNCRGB::VNCRGB(const uint32_t &pixel)
{
    Red = (uint8_t)(pixel);
    Green = (uint8_t)(pixel >> 8);
    Blue = (uint8_t)(pixel >> 16);
}

VNCRGB::VNCRGB(uint8_t r, uint8_t g, uint8_t b)
{
    Red = r;
    Green = g;
    Blue = b;
}

VNCClient::VNCClient(const char *server, int port, const char *pswdfile)
    : rfbproto(server, port, pswdfile)
{
    fbWidth = 0;
    fbHeight = 0;
    framebuffer = NULL;
    viewonly = false;
    serverCutText = NULL;
    newServerCutText = false;
    sharedAppSupport = false;
}

bool VNCClient::VNCInit()
{
    if (rfbproto.connectToRFBServer() && rfbproto.initialiseRFBConnection())
    {
        rfbproto.setVisual32();

        if (!rfbproto.setFormatAndEncodings())
        {
            fprintf(stderr,
                    "VNCClient::VNCInit() err: unable to set default PixelFormat\n");
            VNCClose();
            return false;
        }

        fbWidth = rfbproto.si.framebufferWidth;
        fbHeight = rfbproto.si.framebufferHeight;

        VNCWindowBuffer *window = getWindow(0);
        window->resize(fbWidth, fbHeight);

        framebuffer = reinterpret_cast<VNCRGB *>(window->getFramebuffer());

        //framebuffer = new VNCRGB[fbWidth * fbHeight];
        if (!framebuffer)
        {
            fprintf(stderr,
                    "VNCClient::VNCInit() err: unable to allocate memory for framebuffer\n");
            VNCClose();
            return false;
        }
        return true;
    }
    else
    {
        fprintf(stderr,
                "VNCClient::VNCInit() err: connection or initialization impossible\n");
        VNCClose();
    }
    return false;
}

bool VNCClient::VNCClose()
{
#ifdef _WIN32
    closesocket(getSock());
#else
    close(getSock());
#endif
    if (framebuffer)
        delete[] framebuffer;
    framebuffer = NULL;
    return true;
}

void VNCClient::sendKeyDownEvent(int keysym)
{
    rfbproto.sendKeyEvent(keysym, true);
}

void VNCClient::sendKeyUpEvent(int keysym)
{
    rfbproto.sendKeyEvent(keysym, false);
}

void VNCClient::sendKeyPressEvent(int keysym)
{
    sendKeyDownEvent(keysym);
    sendKeyUpEvent(keysym);
}

void VNCClient::sendPointerEvent(int x, int y, int buttons)
{
    rfbproto.sendPointerEvent(x, y, buttons);
}

uint8_t VNCClient::rescalePixValue(uint32_t Pix, uint8_t Shift, uint16_t Max)
{
    return (uint8_t)(((Pix >> Shift) & Max) * (256 / (Max + 1)));
}

// can be used with a uint8_t, uint16_t or uint32_t Pixel
VNCRGB VNCClient::cardToVNCRGB(uint32_t Pixel)
{
    rfbPixelFormat PF = rfbproto.pixFormat;

    return VNCRGB(rescalePixValue(Pixel, PF.redShift, PF.redMax),
                  rescalePixValue(Pixel, PF.greenShift, PF.greenMax), rescalePixValue(
                                                                          Pixel, PF.blueShift, PF.blueMax));
}

// All the methods we need to handle a given rect message
// And update the framebuffer

bool VNCClient::handleRAW32(int rx, int ry, int rw, int rh)
{
    uint32_t *src = (uint32_t *)rfbbuffer;
    VNCRGB *dest = (framebuffer + ry * fbWidth + rx);

    for (int h = 0; h < rh; h++)
    {
        for (int w = 0; w < rw; w++)
        {
            *dest++ = cardToVNCRGB(swap32IfLE(*src)); //BUG! segfault under Linux
            src++;
        }
        dest += fbWidth - rw;
    }
    return true;
}

bool VNCClient::handleCR(int srcx, int srcy, int rx, int ry, int rw, int rh)
{
    VNCRGB *src;
    VNCRGB *dest;
    VNCRGB *buftmp = new VNCRGB[rh * rw];

    src = framebuffer + (srcy * fbWidth + srcx);
    dest = buftmp;
    for (int h = 0; h < rh; h++)
    {
        memcpy(dest, src, rw * sizeof(VNCRGB));
        src += fbWidth;
        dest += rw;
    }

    src = buftmp;
    dest = framebuffer + (ry * fbWidth + rx);
    for (int h = 0; h < rh; h++)
    {
        memcpy(dest, src, rw * sizeof(VNCRGB));
        src += rw;
        dest += fbWidth;
    }
    delete[] buftmp;
    return true;
}

void VNCClient::fillRect(int rx, int ry, int rw, int rh, VNCRGB pixel)
{
    VNCRGB *dest = framebuffer + (ry * fbWidth + rx);

    // was trace
    TRACE(fprintf(stderr, "fillRect: rx=%d ry=%d rw=%d rh=%d\n", rx, ry, rw, rh);)

    for (int h = 0; h < rh; h++)
    {
        for (int w = 0; w < rw; w++)
            *dest++ = pixel;
        dest += fbWidth - rw;
    }
}

bool VNCClient::handleRRE32(int rx, int ry, int rw, int rh)
{
    rfbRREHeader hdr;
    uint32_t pix;
    rfbRectangle subrect;
    VNCRGB pixel;

    // was trace
    TRACE(fprintf(stderr, "handleRRE32: rx=%d ry=%d rw=%d rh=%d\n", rx, ry, rw, rh);)

    if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&hdr, sz_rfbRREHeader))
        return false;

    hdr.nSubrects = swap32IfLE(hdr.nSubrects);

    if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&pix, sizeof(pix)))
        return false;

    pixel = cardToVNCRGB(swap32IfLE(pix));
    fillRect(rx, ry, rw, rh, pixel);

    for (unsigned int i = 0; i < hdr.nSubrects; i++)
    {
        if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&pix, sizeof(pix)))
            return false;

        if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&subrect,
                                                 sz_rfbRectangle))
            return false;

        subrect.x = swap16IfLE(subrect.x);
        subrect.y = swap16IfLE(subrect.y);
        subrect.w = swap16IfLE(subrect.w);
        subrect.h = swap16IfLE(subrect.h);

        pixel = cardToVNCRGB(swap32IfLE(pix));
        fillRect(subrect.x, subrect.y, subrect.w, subrect.h, pixel);
    }
    return true;
}

bool VNCClient::handleCoRRE32(int rx, int ry, int rw, int rh)
{
    rfbRREHeader hdr;
    uint32_t pix;
    uint8_t *ptr;
    int x, y, ww, hh;
    VNCRGB pixel;

    // was trace
    TRACE(fprintf(stderr, "handleCoRRE32: rx=%d ry=%d rw=%d rh=%d\n", rx, ry, rw, rh);)

    if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&hdr, sz_rfbRREHeader))
        return false;

    hdr.nSubrects = swap32IfLE(hdr.nSubrects);

    if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&pix, sizeof(pix)))
        return false;

    pixel = cardToVNCRGB(swap32IfLE(pix));
    fillRect(rx, ry, rw, rh, pixel);

    if (!rfbproto.VNC_sock.ReadFromRFBServer(rfbbuffer, hdr.nSubrects * 8))
        return false;

    ptr = (uint8_t *)rfbbuffer;

    for (unsigned int i = 0; i < hdr.nSubrects; i++)
    {
        pix = *(uint32_t *)ptr;
        ptr += 4;
        x = *ptr++;
        y = *ptr++;
        ww = *ptr++;
        hh = *ptr++;

        pixel = cardToVNCRGB(swap32IfLE(pix));
        fillRect(rx + x, ry + y, ww, hh, pixel);
    }
    return true;
}

#define GET_PIXEL32(pix, ptr) (((uint8_t *)&(pix))[0] = *(ptr)++, \
                               ((uint8_t *)&(pix))[1] = *(ptr)++, \
                               ((uint8_t *)&(pix))[2] = *(ptr)++, \
                               ((uint8_t *)&(pix))[3] = *(ptr)++)

bool VNCClient::handleHextile32(int rx, int ry, int rw, int rh)
{
    uint32_t bg, fg;
    int i;
    uint8_t *ptr;
    int x, y, w, h;
    int sx, sy, sw, sh;
    uint8_t subencoding;
    uint8_t nSubrects;
    VNCRGB pixel;

    // was trace
    TRACE(fprintf(stderr, "handleHextile32: rx=%d ry=%d rw=%d rh=%d\n", rx, ry, rw, rh);)

    for (y = ry; y < ry + rh; y += 16)
    {
        for (x = rx; x < rx + rw; x += 16)
        {
            w = h = 16;
            if (rx + rw - x < 16)
                w = rx + rw - x;
            if (ry + rh - y < 16)
                h = ry + rh - y;

            if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&subencoding, 1))
                return false;
            if (subencoding & rfbHextileRaw)
            {
                if (!rfbproto.VNC_sock.ReadFromRFBServer(rfbbuffer, w * h * 4))
                    return false;

                handleRAW32(x, y, w, h);
                continue;
            }

            if (subencoding & rfbHextileBackgroundSpecified)
                if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&bg, sizeof(bg)))
                    return false;

            pixel = VNCRGB(swap32IfLE(bg));
            fillRect(x, y, w, h, pixel);

            if (subencoding & rfbHextileForegroundSpecified)
                if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&fg, sizeof(fg)))
                    return false;
            if (!(subencoding & rfbHextileAnySubrects))
                continue;

            if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&nSubrects, 1))
                return false;

            ptr = (uint8_t *)rfbbuffer;

            if (subencoding & rfbHextileSubrectsColoured)
            {
                if (!rfbproto.VNC_sock.ReadFromRFBServer(rfbbuffer, nSubrects * 6))
                    return false;

                for (i = 0; i < nSubrects; i++)
                {
                    GET_PIXEL32(fg, ptr);
                    sx = rfbHextileExtractX(*ptr);
                    sy = rfbHextileExtractY(*ptr);
                    ptr++;
                    sw = rfbHextileExtractW(*ptr);
                    sh = rfbHextileExtractH(*ptr);
                    ptr++;
                    pixel = VNCRGB(swap32IfLE(fg));
                    fillRect(x + sx, y + sy, sw, sh, pixel);
                }
            }
            else
            {
                if (!rfbproto.VNC_sock.ReadFromRFBServer(rfbbuffer, nSubrects * 2))
                    return false;

                for (i = 0; i < nSubrects; i++)
                {
                    sx = rfbHextileExtractX(*ptr);
                    sy = rfbHextileExtractY(*ptr);
                    ptr++;
                    sw = rfbHextileExtractW(*ptr);
                    sh = rfbHextileExtractH(*ptr);
                    ptr++;
                    pixel = VNCRGB(swap32IfLE(fg));
                    fillRect(x + sx, y + sy, sw, sh, pixel);
                }
            }
        }
    }
    return true;
}

bool VNCClient::handleRFBServerMessage()
{
    rfbServerToClientMsg msg;

    if (!framebuffer)
    {
        return false;
    }

    if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&msg, 1))
    {
        return false;
    }

    switch (msg.type)
    {
    case rfbSetColourMapEntries:
        fprintf(
            stderr,
            "VNCClient::handleRFBServerMessage(): rfbSetColourMapEntries not supported yet\n");
        uint16_t rgb[3];

        if (!rfbproto.VNC_sock.ReadFromRFBServer(((char *)&msg) + 1,
                                                 sz_rfbSetColourMapEntriesMsg - 1))
        {
            return false;
        }

        msg.scme.firstColour = swap16IfLE(msg.scme.firstColour);
        msg.scme.nColours = swap16IfLE(msg.scme.nColours);

        for (int i = 0; i < msg.scme.nColours; i++)
        {
            if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)rgb, 6))
            {
                return false;
            }
        }
        break;

    case rfbFramebufferUpdate:
        TRACE(fprintf(stderr, "VNCClient::handleRFBServerMessage(): rfbFramebufferUpdate\n");)
        return handleFramebufferUpdateMessage();

    case rfbBell:
        TRACE(fprintf(stderr, "VNCClient::handleRFBServerMessage(): Bell!\n");)
        //XBell(dpy,100);
        break;

    case rfbServerCutText:
        if (!rfbproto.VNC_sock.ReadFromRFBServer(((char *)&msg) + 1,
                                                 sz_rfbServerCutTextMsg - 1))
        {
            TRACE(fprintf(stderr, "VNCClient::handleRFBServerMessage(): rfbServerCutText 0!\n");)
            return false;
        }

        msg.sct.length = swap32IfLE(msg.sct.length);

        if (serverCutText)
        {
            delete[] serverCutText;
            serverCutText = NULL;
        }

        serverCutText = new char[msg.sct.length + 1];

        if (!rfbproto.VNC_sock.ReadFromRFBServer(serverCutText, msg.sct.length))
        {
            TRACE(fprintf(stderr, "VNCClient::handleRFBServerMessage(): rfbServerCutText 1!\n");)
            return false;
        }

        serverCutText[msg.sct.length] = 0;
        newServerCutText = true;
        break;

    case rfbSharedAppUpdate:
        fprintf(stderr,
                "VNCClient::handleRFBServerMessage() info: Shared App update package!");
        return handleSharedAppUpdateMessage();

    default:
        fprintf(
            stderr,
            "VNCClient::handleRFBServerMessage(): Unknown message type %d from VNC server\n",
            msg.type);
        return false;
    }
    return true;
}

bool VNCClient::handleFramebufferUpdateMessage()
{
    rfbServerToClientMsg msg;
    if (!rfbproto.VNC_sock.ReadFromRFBServer(((char *)&msg.fu) + 1,
                                             sz_rfbFramebufferUpdateMsg - 1))
    {
        return false;
    }

    msg.fu.nRects = swap16IfLE(msg.fu.nRects);

    VNCWindowBuffer *wb = getWindow(0); // the 'desktop' has id 0
    framebuffer = reinterpret_cast<VNCRGB *>(wb->getFramebuffer());
    fbWidth = wb->getWidth();
    fbHeight = wb->getHeight();

    return handleRFBUpdateRects(msg.fu.nRects);
}

bool VNCClient::handleSharedAppUpdateMessage()
{
    sharedAppSupport = true;

    rfbSharedAppUpdateMsg msg;
    if (!rfbproto.VNC_sock.ReadFromRFBServer(((char *)&msg) + 1,
                                             sz_rfbSharedAppUpdateMsg - 1))
    {
        fprintf(
            stderr,
            "VNCClient::handleSharedAppUpdateMessage() err: Error while trying to read a SharedAppUpdate message!\n");
        return false;
    }
    msg.numRects = swap16IfLE(msg.numRects);
    msg.windowId = swap32IfLE(msg.windowId);
    msg.parentId = swap32IfLE(msg.parentId);
    msg.updateRect.x = swap16IfLE(msg.updateRect.x);
    msg.updateRect.y = swap16IfLE(msg.updateRect.y);
    msg.updateRect.w = swap16IfLE(msg.updateRect.w);
    msg.updateRect.h = swap16IfLE(msg.updateRect.h);
    msg.cursorOffsetX = swap16IfLE(msg.cursorOffsetX);
    msg.cursorOffsetY = swap16IfLE(msg.cursorOffsetY);

    if (msg.updateRect.x == 0 && msg.updateRect.y == 0 && msg.updateRect.w
        && msg.updateRect.h == 0)
    {
        // close specific window...
        closeWindow(msg.windowId);
        return true;
    }

    VNCWindowBuffer *wb = getWindow(msg.windowId);

    if (wb->getWidth() < msg.updateRect.w || wb->getHeight() < msg.updateRect.h)
    {
        wb->resize(msg.updateRect.w, msg.updateRect.h);
    }

    framebuffer = reinterpret_cast<VNCRGB *>(wb->getFramebuffer());
    fbWidth = wb->getWidth();
    fbHeight = wb->getHeight();

    return handleRFBUpdateRects(msg.numRects);
#if 0
   for (int i = 0; i < msg.numRects; ++i)
   {
      // read framebufferUpdateRectHdr
   }
   return true;
#endif
}

int VNCClient::getSock()
{
    return rfbproto.getSock();
}

bool VNCClient::sendIncrementalFramebufferUpdateRequest()
{
    return rfbproto.sendIncrementalFramebufferUpdateRequest();
}

bool VNCClient::sendFramebufferUpdateRequest(int x, int y, int w, int h,
                                             bool incremental)
{
    return rfbproto.sendFramebufferUpdateRequest(x, y, w, h, incremental);
}

VNCWindowBuffer *VNCClient::getWindow(int id)
{
    (void)id;
    return wb;
#if 0
   std::map<int, VNCWindowBuffer*>::iterator it = windows.find(id);
   if (it == windows.end())
   {
      VNCWindowBuffer* wb = new VNCWindowBuffer(1, 1, id, 0);
      windows.insert(std::pair<int, VNCWindowBuffer*>(id, wb));
      return wb;
   }
   return it->second;
#endif
}

void VNCClient::closeWindow(int id)
{
    std::map<int, VNCWindowBuffer *>::iterator it = windows.find(id);
    if (it == windows.end())
    {
        fprintf(stderr,
                "VNCClient::closeWindow() err: Window id %d does not exist!\n", id);
    }
}

bool VNCClient::handleRFBUpdateRects(int num)
{
    rfbFramebufferUpdateRectHeader rect;
    int linesToRead;
    int bytesPerLine;
    int i;

    for (i = 0; i < num; i++)
    {
        if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&rect,
                                                 sz_rfbFramebufferUpdateRectHeader))
        {
            return false;
        }

        rect.r.x = swap16IfLE(rect.r.x);
        rect.r.y = swap16IfLE(rect.r.y);
        rect.r.w = swap16IfLE(rect.r.w);
        rect.r.h = swap16IfLE(rect.r.h);

        rect.encoding = swap32IfLE(rect.encoding);

        if ((rect.r.x + rect.r.w > rfbproto.si.framebufferWidth) || (rect.r.y
                                                                     + rect.r.h > rfbproto.si.framebufferHeight))
        {
            fprintf(
                stderr,
                "VNCClient::handleRFBServerMessage() err: Rect too large: %dx%d at (%d, %d)\n",
                rect.r.w, rect.r.h, rect.r.x, rect.r.y);
            return false;
        }

        if ((rect.r.h * rect.r.w) == 0)
        {
            fprintf(stderr,
                    "VNCClient::handleRFBServerMessage(): Zero rfbproto.size rect - ignoring\n");
            continue;
        }

        wb->getUpdateQueue()->push(rect.r);

        switch (rect.encoding)
        {
        case rfbEncodingRaw:
            bytesPerLine = rect.r.w * rfbproto.pixFormat.bitsPerPixel / 8;
            linesToRead = sizeof(rfbbuffer) / bytesPerLine;

            fprintf(
                stderr,
                "VNCClient::handleRFBServerMessage(): rfbFramebufferUpdate rfbEncodingRaw bytesPerLine=%d linesToRead=%d\n",
                bytesPerLine, linesToRead);

            while (rect.r.h > 0)
            {
                if (linesToRead > rect.r.h)
                {
                    linesToRead = rect.r.h;
                }

                if (!rfbproto.VNC_sock.ReadFromRFBServer(rfbbuffer, bytesPerLine
                                                                    * linesToRead))
                {
                    return false;
                }

                if (!handleRAW32(rect.r.x, rect.r.y, rect.r.w, rect.r.h))
                {
                    return false;
                }

                rect.r.h -= linesToRead;
                rect.r.y += linesToRead;
            }
            break;

        case rfbEncodingCopyRect:
            rfbCopyRect cr;
            if (!rfbproto.VNC_sock.ReadFromRFBServer((char *)&cr,
                                                     sz_rfbCopyRect))
            {
                return false;
            }

            cr.srcX = swap16IfLE(cr.srcX);
            cr.srcY = swap16IfLE(cr.srcY);

            if (!handleCR(cr.srcX, cr.srcY, rect.r.x, rect.r.y, rect.r.w,
                          rect.r.h))
            {
                return false;
            }
            break;

        case rfbEncodingRRE:
            if (!handleRRE32(rect.r.x, rect.r.y, rect.r.w, rect.r.h))
            {
                return false;
            }
            break;

        case rfbEncodingCoRRE:
            if (!handleCoRRE32(rect.r.x, rect.r.y, rect.r.w, rect.r.h))
            {
                return false;
            }
            break;

        case rfbEncodingHextile:
            if (!handleHextile32(rect.r.x, rect.r.y, rect.r.w, rect.r.h))
            {
                return false;
            }
            break;

        default:
            fprintf(
                stderr,
                "VNCClient::handleRFBServerMessage(): Unknown rect encoding %d\n",
                (int)rect.encoding);
            return false;
        }
    }

    if (!sendIncrementalFramebufferUpdateRequest())
    {
        return false;
    }
    return true;
}

bool VNCClient::pollServer()
{
    // call VNC client status...
    bool updated = false;
    fd_set rmask;
    struct timeval delay;
    int rfbsock = getSock();
    delay.tv_sec = 0;
    delay.tv_usec = 10; // was 10
    FD_ZERO(&rmask);
    FD_SET(rfbsock, &rmask);

    if (select(rfbsock + 1, &rmask, NULL, NULL, &delay))
    {
        updated = true;
        if (FD_ISSET(rfbsock, &rmask))
        {
            if (!handleRFBServerMessage())
            {
                fprintf(
                    stderr, "VNCClient::pollServer() err: can't handle RFB server message\n");
                return false;
            }
        }
    }
    return updated;
}

bool VNCClient::checkNewWindows() const
{
    return false;
}

void VNCClient::clearNewWindowList()
{
}
