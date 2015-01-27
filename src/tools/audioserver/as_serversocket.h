/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_serversocket.h
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

#ifndef __AS_SOCKSERVER_H_
#define __AS_SOCKSERVER_H_

//#include "as_ClientSocket.h"

#define MAX_CLIENTS 16
#define MAX_BUFLEN 32767

#define SOCKPORT_DEFAULT 31231

class as_ServerSocket
{
public:
    int s_handle; ///< socket handle
    int cl_handle; ///< socket handle
    char hostname[1024]; // host name
    unsigned short port; // socket port
    struct sockaddr_in address; // socket address
    struct hostent *host; // host structure
    int clients_connected; // number of connected clients
    //	as_ClientSocket		* clients[MAX_CLIENTS];	// array of client structure
    //	as_ClientSocket		* currentClient;	// current acting client
    HWND hWnd; // handle of main application window
    HANDLE hThread;
    unsigned long lpThreadId;

    //	int addClient(as_ClientSocket *Client);
    int checkVersion(void);
    void cleanup(void);
    void eventHandler(WPARAM wParam, LPARAM lParam);
    int create(HWND hWindow);
    void printError(void);
    int waitForClient(void);
    //	int disconnectClient(as_ClientSocket *Client);
    int disconnectClient(void);
    int socketHandler(SOCKET s);
    int sendMessage(char *buffer);
    unsigned short GetPort(void);
    void SetPort(unsigned short newport);
};

extern as_ServerSocket *AS_Server;
DWORD WINAPI socketThread(as_ServerSocket *pServerSocket);
#endif
