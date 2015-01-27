/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_clientsocket.h
 *
 *  Project : AudioServer
 *
 *  Package : AudioServer prototype
 *
 *  Author  : Marc Schreier                              Date: 05/05/2002
 *
 *  Purpose : Header file
 *
 *********************************************************************
 */

#ifndef AS_CLIENTSOCKET_H_
#define AS_CLIENTSOCKET_H_

class as_ClientSocket
{
public:
    int s_handle; // connection socket handle
    struct sockaddr_in address; // socket address
    HANDLE thread; // thread handle
    unsigned long threadId; // thread ID
    bool master; // first one becomes master
    HWND hWnd; // handle of main application window
};
#endif
