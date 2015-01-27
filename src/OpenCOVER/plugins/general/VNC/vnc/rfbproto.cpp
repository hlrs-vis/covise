/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2008 HLRS  **
 **                                                                        **
 ** Description: RFB Protocol                                              **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** based on rfbproto.cpp from VREng                                       **
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
/*
 *  Copyright (C) 1999 AT&T Laboratories Cambridge.  All Rights Reserved.
 *
 *  This is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This software is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this software; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307,
 *  USA.
 */
/*
 * rfbproto.cpp - functions to deal with client side of RFB protocol.
 */

#include "rfbproto.h"

#ifndef WIN32
#include <stdint.h>
#endif
#include <string.h>
#include <stdio.h>

int endianTest = 1;
char *serverCutText = NULL;
bool newServerCutText = false;

/* note that the CoRRE encoding uses this buffer and assumes it is big enough
 to hold 255 * 255 * 32 bits -> 260100 bytes.  640*480 = 307200 bytes */
/* also hextile assumes it is big enough to hold 16 * 16 * 32 bits */

/** Constructors */
VNCRFBproto::VNCRFBproto(const char *Servername, int Port, const char *pswdFile)
    : VNC_sock(Servername, Port)
{
    ShareDesktop = true;
    buffered = 0;
    strcpy(passwordFile, pswdFile);
#if 0 //not used
   Encodings = NULL; // not used
#endif //not used
}

/** ConnectToRFBServer */
bool VNCRFBproto::connectToRFBServer()
{
    int sock = VNC_sock.ConnectToTcpAddr();
    if (sock < 0)
    {
        // was error
        fprintf(stderr,
                "VNCRFBproto::connectToRFBServer() err: unable to connect to VNC server\n");
        return false;
    }
    return VNC_sock.SetNonBlocking();
}

bool VNCRFBproto::initialiseRFBConnection()
{
    rfbProtocolVersionMsg pv;
    rfbClientInitMsg ci;
    int major, minor;
    char *reason; //, *passwd;
    reason = NULL;
    //passwd = NULL;
    CARD32 authScheme, reasonLen, authResult;
    CARD8 challenge[CHALLENGESIZE];

    // was trace
    fprintf(stderr,
            "VNCRFBproto::initialiseRFBConnection(): initializing RFB connection\n");

    if (!VNC_sock.ReadFromRFBServer(pv, sz_rfbProtocolVersionMsg))
        return false;

    pv[sz_rfbProtocolVersionMsg] = 0;

    if (sscanf(pv, rfbProtocolVersionFormat, &major, &minor) != 2)
    {
        // was error
        fprintf(stderr,
                "VNCRFBproto::initialiseRFBConnection() err: not a valid VNC server\n");
        return false;
    }

    // was trace
    fprintf(
        stderr,
        "VNCRFBproto::initialiseRFBConnection(): VNC server supports protocol version %d.%d (viewer %d.%d)\n",
        major, minor, rfbProtocolMajorVersion, rfbProtocolMinorVersion);

    major = rfbProtocolMajorVersion;
    minor = rfbProtocolMinorVersion;
    sprintf(pv, rfbProtocolVersionFormat, major, minor);

    if (!VNC_sock.WriteExact(pv, sz_rfbProtocolVersionMsg))
        return false;
    if (!VNC_sock.ReadFromRFBServer((char *)&authScheme, 4))
        return false;

    authScheme = (CARD32)swap32IfLE(authScheme);
    switch (authScheme)
    {

    case rfbConnFailed:
        if (!VNC_sock.ReadFromRFBServer((char *)&reasonLen, 4))
            return false;
        reasonLen = swap32IfLE(reasonLen);
        reason = new char[reasonLen];

        if (!VNC_sock.ReadFromRFBServer(reason, reasonLen))
            return false;
        // error
        fprintf(
            stderr,
            "VNCRFBproto::initialiseRFBConnection() err: VNC connection failed: %.*s\n",
            (int)reasonLen, reason);
        delete[] reason;
        return false;

    case rfbNoAuth:
        // was trace
        fprintf(stderr,
                "VNCRFBproto::initialiseRFBConnection(): No authentication needed\n");
        break;

    case rfbVncAuth:
        if (!VNC_sock.ReadFromRFBServer((char *)challenge, CHALLENGESIZE))
            return false;

        if (strlen(passwordFile) > 8)
            passwordFile[8] = '\0';

        vncEncryptBytes(challenge, passwordFile);

        if (!VNC_sock.WriteExact((char *)challenge, CHALLENGESIZE))
            return false;
        if (!VNC_sock.ReadFromRFBServer((char *)&authResult, 4))
            return false;

        authResult = swap32IfLE(authResult);
        switch (authResult)
        {
        case rfbVncAuthOK:
            // was trace
            fprintf(stderr,
                    "VNCRFBproto::initialiseRFBConnection(): VNC authentication succeeded\n");
            break;
        case rfbVncAuthFailed:
            // was error
            fprintf(stderr,
                    "VNCRFBproto::initialiseRFBConnection() err: VNC authentication failed\n");
            return false;
        case rfbVncAuthTooMany:
            // was error
            fprintf(
                stderr,
                "VNCRFBproto::initialiseRFBConnection() err: VNC authentication failed - too many tries\n");
            return false;
        default:
            // was error
            fprintf(
                stderr,
                "VNCRFBproto::initialiseRFBConnection() err: Unknown VNC authentication result: %d\n",
                (int)authResult);
            return false;
        }
        break;

    default:
        // was error
        fprintf(
            stderr,
            "VNCRFBproto::initialiseRFBConnection() err: Unknown authentication scheme from VNC server: %d\n",
            (int)authScheme);
        return false;
    }

    ci.shared = (ShareDesktop ? 1 : 0);

    if (!VNC_sock.WriteExact((char *)&ci, sz_rfbClientInitMsg))
        return false;
    if (!VNC_sock.ReadFromRFBServer((char *)&si, sz_rfbServerInitMsg))
        return false;

    si.framebufferWidth = swap16IfLE(si.framebufferWidth);
    si.framebufferHeight = swap16IfLE(si.framebufferHeight);
    si.format.redMax = swap16IfLE(si.format.redMax);
    si.format.greenMax = swap16IfLE(si.format.greenMax);
    si.format.blueMax = swap16IfLE(si.format.blueMax);
    si.nameLength = swap32IfLE(si.nameLength);

    desktopName = new char[si.nameLength + 1];
    if (!VNC_sock.ReadFromRFBServer(desktopName, si.nameLength))
    {
        delete[] desktopName;
        desktopName = NULL;
        return false;
    }
    desktopName[si.nameLength] = 0;
    // was notice
    fprintf(stderr,
            "VNCRFBproto::initialiseRFBConnection(): Using desktop: %s\n",
            desktopName);
    delete[] desktopName;
    desktopName = NULL;

    // was 2x trace
    fprintf(
        stderr,
        "VNCRFBproto::initialiseRFBConnection(): connected to VNC server, using protocol version %d.%d\n",
        rfbProtocolMajorVersion, rfbProtocolMinorVersion);
    fprintf(stderr,
            "VNCRFBproto::initialiseRFBConnection(): VNC server default format:\n");
    printPixelFormat(&si.format);

    return true;
}

bool VNCRFBproto::setFormatAndEncodings()
{
    rfbSetPixelFormatMsg spf;
    char buf[sz_rfbSetEncodingsMsg + MAX_ENCODINGS * 4];
    rfbSetEncodingsMsg *se = (rfbSetEncodingsMsg *)buf;
    CARD32 *encs = (CARD32 *)(&buf[sz_rfbSetEncodingsMsg]);
    int len = 0;

    spf.type = rfbSetPixelFormat;
    spf.format = pixFormat;
    spf.format.redMax = swap16IfLE(spf.format.redMax);
    spf.format.greenMax = swap16IfLE(spf.format.greenMax);
    spf.format.blueMax = swap16IfLE(spf.format.blueMax);

    if (!VNC_sock.WriteExact((char *)&spf, sz_rfbSetPixelFormatMsg))
        return false;

    se->type = rfbSetEncodings;
    se->nEncodings = 0;

    //
    // this was commented out, as there was a segfault, when using raw encoding
    // on linux

    //if (VNC_sock.SameMachine()) {
    // was error
    //fprintf(stderr, "Same machine: preferring raw encoding\n");
    //encs[se->nEncodings++] = swap32IfLE(rfbEncodingRaw);
    //}
    encs[se->nEncodings++] = swap32IfLE(rfbEncodingCopyRect);
    encs[se->nEncodings++] = swap32IfLE(rfbEncodingHextile);
    encs[se->nEncodings++] = swap32IfLE(rfbEncodingCoRRE);
    encs[se->nEncodings++] = swap32IfLE(rfbEncodingRRE);

    len = sz_rfbSetEncodingsMsg + se->nEncodings * 4;
    se->nEncodings = swap16IfLE(se->nEncodings);

    if (!VNC_sock.WriteExact(buf, len))
        return false;

    return true;
}

bool VNCRFBproto::sendIncrementalFramebufferUpdateRequest()
{
    return sendFramebufferUpdateRequest(0, 0, si.framebufferWidth,
                                        si.framebufferHeight, true);
}

bool VNCRFBproto::sendFramebufferUpdateRequest(int x, int y, int w, int h,
                                               bool incremental)
{
    rfbFramebufferUpdateRequestMsg fur;

    fur.type = rfbFramebufferUpdateRequest;
    fur.incremental = incremental ? 1 : 0;
    fur.x = swap16IfLE(x);
    fur.y = swap16IfLE(y);
    fur.w = swap16IfLE(w);
    fur.h = swap16IfLE(h);

    if (!VNC_sock.WriteExact((char *)&fur, sz_rfbFramebufferUpdateRequestMsg))
        return false;

    return true;
}

bool VNCRFBproto::sendPointerEvent(int x, int y, int buttonMask)
{
    rfbPointerEventMsg pe;

    pe.type = rfbPointerEvent;
    pe.buttonMask = buttonMask;
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    pe.x = swap16IfLE(x);
    pe.y = swap16IfLE(y);
    return VNC_sock.WriteExact((char *)&pe, sz_rfbPointerEventMsg);
}

bool VNCRFBproto::sendKeyEvent(CARD32 key, bool down)
{
    rfbKeyEventMsg ke;

    ke.type = rfbKeyEvent;
    ke.down = down ? 1 : 0;
    ke.key = swap32IfLE(key);
    return VNC_sock.WriteExact((char *)&ke, sz_rfbKeyEventMsg);
}

bool VNCRFBproto::sendClientCutText(char *str, int len)
{
    rfbClientCutTextMsg cct;

    if (serverCutText)
        delete[] serverCutText;
    serverCutText = NULL;

    cct.type = rfbClientCutText;
    cct.length = swap32IfLE(len);

    return (VNC_sock.WriteExact((char *)&cct, sz_rfbClientCutTextMsg)
            && VNC_sock.WriteExact(str, len));
}

/**
 * initialise pixFormat qui donne le format de pixel que l'on veut que
 * le serveur nous envoie.
 * initialise pour demander le RGB
 * sur 3*8 bits = 24 bits
 * A APPELER DANS LE CONSTRUCTEUR (mto)
 */
void VNCRFBproto::setVisual32()
{
    // was trace
    //fprintf(stderr, "SetVisual32\n");
    pixFormat.bitsPerPixel = 32;
    pixFormat.depth = 24;
    pixFormat.trueColour = 1;
    pixFormat.bigEndian = 1;
    pixFormat.redMax = 255;
    pixFormat.greenMax = 255;
    pixFormat.blueMax = 255;
    pixFormat.redShift = 0;
    pixFormat.greenShift = 8;
    pixFormat.blueShift = 16;
}

void VNCRFBproto::printPixelFormat(rfbPixelFormat *format)
{
    if (format->bitsPerPixel == 1)
    {
        // was 2x trace
        fprintf(stderr, "VNCRFBproto::printPixelFormat(): Single bit per pixel\n");
        fprintf(
            stderr,
            "VNCRFBproto::printPixelFormat(): %s significant bit in each byte is leftmost on the screen\n",
            (format->bigEndian ? "Most" : "Least"));
    }
    else
    {
        // was trace
        fprintf(stderr, "VNCRFBproto::printPixelFormat(): %d bits per pixel\n",
                format->bitsPerPixel);
        if (format->bitsPerPixel != 8)
        {
            fprintf(
                stderr,
                "VNCRFBproto::printPixelFormat(): %s significant byte first in each pixel\n",
                (format->bigEndian ? "Most" : "Least"));
        }
        if (format->trueColour)
        {
            fprintf(
                stderr,
                "VNCRFBproto::printPixelFormat(): True color: max red %d green %d blue %d, shift red %d green %d blue %d\n",
                format->redMax, format->greenMax, format->blueMax,
                format->redShift, format->greenShift, format->blueShift);
        }
        else
            fprintf(stderr,
                    "VNCRFBproto::printPixelFormat(): Colour map (not true colour)\n");
    }
}

int VNCRFBproto::getSock()
{
    return VNC_sock.GetSock();
}
