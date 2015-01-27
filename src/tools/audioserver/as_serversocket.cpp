/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_ServerSocket.cpp
 *
 *  Project : AudioServer
 *
 *  Package : AudioServer prototype
 *
 *  Author  : Marc Schreier                           Date: 05/05/2002
 *
 *  Purpose : Network server socket functions
 *
 *********************************************************************
 */

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif

#include <windows.h>

#include <assert.h>
#include <errno.h>
#include <iostream>
#include <string>

// #include <process.h>

#include <mmsystem.h>

#include "as_ServerSocket.h"
#include "as_client_api.h"
//#include "as_gui.h"
#include "as_comm.h"
#include "as_Control.h"

#include "common.h"

using namespace std;

DWORD threadId;

/*
int as_ServerSocket::addClient(as_ClientSocket *Client)
{
   int count_clients = 0;
   int i = 0;
   char msg[MSG_LEN];
//	DWORD exitCode;   // thread termination status

   // check all previous connections
   for (i = 0; i< MAX_CLIENTS; i++) {
      if (NULL != clients[i]) {
sprintf(msg, "Checking client %d", i);
AddLogMsg(msg);
if (NULL == clients[i]->thread) {
sprintf(msg, "Thread info == NULL!");
AddLogMsg(msg);
continue;
}

if (0 == GetExitCodeThread(
clients[i]->thread,
&exitCode
)){
sprintf(msg, "Could not get thread information (error %d)",
GetLastError());
AddLogMsg(msg);
} else {
if (exitCode != STILL_ACTIVE) {
sprintf(msg, "Client %d terminated!", i);
AddLogMsg(msg);
clients[i] = NULL;
continue;
} else {
sprintf(msg, "Client %d is still active!", i);
AddLogMsg(msg);
}
}

count_clients++;
}
}
clients_connected = count_clients;

sprintf(msg, "Currently %d client(s) connected.", count_clients);
AddLogMsg(msg);

if (count_clients == MAX_CLIENTS) {
sprintf(msg, "Already maximum number (%d) of clients connected!",
MAX_CLIENTS);
AddLogMsg(msg);
return -1;
}

// search for free place in list
for (i = 0; i < MAX_CLIENTS; i++) {
if (NULL==clients[i]) break;
}

// is it really free?
if ((MAX_CLIENTS == i) || (NULL != clients[i])) {
sprintf(msg, "Already maximum number (%d) of clients connected!",
MAX_CLIENTS);
AddLogMsg(msg);
return -1;
}

if (1 == count_clients) {
Client->master = 1;
} else {
Client->master = 0;
}
clients[i] = Client;
currentClient = Client;
return i;
}
*/

int as_ServerSocket::checkVersion(void)
{
    int err;
    WORD wVersionRequested;
    WSADATA wsaData;
    /*
      // initialize clients array
      for (int i = 0; i < MAX_CLIENTS; i++) {
         clients[i] = NULL;
      }
   */
    // request specific Winsock DLL version
    wVersionRequested = MAKEWORD(2, 2);
    err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0)
    {
        // No WinSock DLL found
        cout << "No WinSock.DLL available!" << endl;
        return -1;
    }

    /* Confirm that the WinSock DLL supports 2.2.
    * Note that if the DLL supports versions greater
    * than 2.2 in addition to 2.2, it will still return
    * 2.2 in wVersion since that is the version we
    * requested.
    */
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
    {

        // wrong version
        cout << "Wrong WinSock version (not 2.2)!" << endl;
        WSACleanup();
        return -1;
    }

    // good version
    return 0;
}

int as_ServerSocket::create(HWND hWindow)
{
    DWORD optval = TRUE;

    // check socket version
    if (0 != checkVersion())
    {
        MessageBox(NULL, "WinSock", "Wrong socket DLL or not available!", MB_ICONSTOP);
        return -1;
    }

    this->hThread = CreateThread(
        NULL,
        NULL,
        (LPTHREAD_START_ROUTINE)socketThread,
        this,
        0,
        &this->lpThreadId);

    /*
      // create socket
      s_handle = socket(AF_INET, SOCK_STREAM, 0);

      // =========================

      wMsg = WM_APP;
      if(SOCKET_ERROR == WSAAsyncSelect (
         s_handle,
         hWindow,
         wMsg,
   FD_READ|FD_ACCEPT|FD_CONNECT|FD_CLOSE
   )) {
   printError();
   WSACleanup( );
   return -1;
   }

   if (INVALID_SOCKET  == s_handle) {
   printError();
   WSACleanup( );
   return -1;
   }

   // set socket options
   if(SOCKET_ERROR == setsockopt(
   s_handle,
   SOL_SOCKET,
   SO_REUSEADDR,
   (const char*)&optval,
   sizeof(optval)
   )) {
   printError();
   WSACleanup();
   return -1;
   }

   // get host info
   gethostname(hostname, 1024);
   host = gethostbyname(hostname);
   if (NULL == host) {
   printError();
   WSACleanup();
   return -1;
   }
   if ((port < 22) || (port > 65535)) {
   port = SOCKPORT_DEFAULT;
   AddLogMsg("Invalid socket port, setting to default!");
   }
   sprintf(msg, "Host name: %s, port: %d", host->h_name, this->port);
   AddLogMsg(msg);

   // set address family and options
   address.sin_family = AF_INET;
   address.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
   address.sin_family = host->h_addrtype;
   address.sin_port = htons(port);

   // bind socket
   if (SOCKET_ERROR == bind(s_handle, (struct sockaddr*)&address, sizeof(address))) {
   printError();
   WSACleanup();
   return -1;
   }

   // listen socket
   if (SOCKET_ERROR == listen(s_handle,1)) {
   printError();
   WSACleanup();
   return -1;
   }
   */
    this->hWnd = hWindow;
    //	clients_connected = 0;

    return 0;
}

/*
int as_ServerSocket::disconnectClient(as_ClientSocket *Client)
{
   int current_client = 0;
   char msg[MSG_LEN];

   if ((NULL == Client) && (NULL == currentClient)) {
      sprintf(msg, "as_ServerSocket::disconnectClient error: wrong parameter");
      AddLogMsg(msg);
      return -1;
   }

if ((NULL == Client) && (NULL != currentClient)) {
Client = currentClient;
}

while (current_client < MAX_CLIENTS) {
if (Client == clients[current_client]) break;
current_client++;
}

shutdown(Client->s_handle, SD_SEND);
closesocket(Client->s_handle);
clients_connected--;
sprintf(msg, "Now %d client(s) connected!", clients_connected);
AddLogMsg(msg);

if (Client) delete(Client);

clients[current_client] = NULL;

if (0 == clients_connected) AS_Control->Panic();
return 0;
}
*/

int as_ServerSocket::disconnectClient(void)
{
    //   shutdown(this->cl_handle, SD_SEND);
    closesocket(this->cl_handle);

    // in case of playing sounds
    AS_Control->Panic();
    return 0;
}

/*	
int as_ServerSocket::waitForClient(void)
{
   as_ClientSocket *Client = new as_ClientSocket;
   char msg[MSG_LEN];
   long wMsg;

   int len = sizeof(struct sockaddr_in);
   int currentClient = -1;
   SECURITY_ATTRIBUTES threadAttributes;

Client->s_handle = accept(this->s_handle, (struct sockaddr *)&Client->address, &len);
if (INVALID_SOCKET == Client->s_handle) {
printError();
return -1;
}
sprintf(msg, "Server socket: %d, Client socket: %d", this->s_handle, Client->s_handle);
AddLogMsg(msg);

sprintf(msg, "accept successful, client address = %s",
inet_ntoa(Client->address.sin_addr));
AddLogMsg(msg);

currentClient = addClient(Client);
if ( -1 == currentClient) {
// too many clients !
sprintf(msg, "Error: could not register client %d", clients_connected+1);
AddLogMsg(msg);

shutdown(Client->s_handle, SD_SEND);
return -1;
}

clients_connected++;
sprintf(msg, "waitForClient: now %d client(s) connected.", clients_connected);
AddLogMsg(msg);

// start client handler thread
Client->thread = _beginthread(clientHandler, 0, &clients[currentClient]);
if (-1 == Client->thread) {
perror("Could not create client handler thread");
Client = NULL;
return -1;
}

threadAttributes.lpSecurityDescriptor = NULL;
threadAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);

// forward main window handle for socket event handlers
Client->hWnd = this->hWnd;

Client->thread = CreateThread(
&threadAttributes,			// pointer to security attributes
0,							// initial thread stack size
clientHandler,				// pointer to thread function
clients[currentClient],		// argument for new thread
0,							// creation flags
&Client->threadId			// pointer to receive thread ID
);
if (NULL == Client->thread) {
sprintf(msg, "Could not create client handler thread (error %d)",
GetLastError());
MessageBox(NULL, msg, "waitForClient", MB_OK);
Client = NULL;
return -1;
}
sprintf(msg, "[%08lxh] Thread created.", Client->threadId);
AddLogMsg(msg);

clients[currentClient] = Client;
return 0;
}
*/

void as_ServerSocket::cleanup(void)
{
    WSAAsyncSelect(this->s_handle, this->hWnd, 0, 0);
    closesocket(this->s_handle);
    WSACleanup();
}

void as_ServerSocket::printError(void)
{
    int err;
    char msg[MSG_LEN];

    err = WSAGetLastError();

    sprintf(msg, "Socket error %d:", err);
    switch (err)
    {
    case WSANOTINITIALISED:
        strcat(msg, " A successful WSAStartup must occur "
                    "before using this function.");
        break;
    case WSAENETDOWN:
        strcat(msg, " The network subsystem or the associated "
                    "service provider has failed.");
        break;
    case WSAEAFNOSUPPORT:
        strcat(msg, " The specified address family is not supported.");
        break;
    case WSAEINPROGRESS:
        strcat(msg, " A blocking Windows Sockets call "
                    "is in progress, or the service provider is "
                    "still processing a callback function.");
        break;
    case WSAEMFILE:
        strcat(msg, " No more socket descriptors are available.");
        break;
    case WSAENOBUFS:
        strcat(msg, " No buffer space is available. "
                    "The socket cannot be created.");
        break;
    case WSAEPROTONOSUPPORT:
        strcat(msg, " The specified protocol is not supported.");
        break;
    case WSAEPROTOTYPE:
        strcat(msg, " The specified protocol is the "
                    "wrong type for this socket.");
        break;
    case WSAESOCKTNOSUPPORT:
        strcat(msg, " The specified socket type is not "
                    "supported in this address family.");
        break;
    default:
        strcat(msg, " Not WSA a error code!");
        break;
    }
    AddLogMsg(msg);
    MessageBox(NULL, "Socket error", msg, MB_ICONSTOP);
}

/*
void as_ServerSocket::eventHandler( WPARAM wParam, LPARAM lParam )
{
   SOCKET s = wParam;
   int socket_error = WSAGETSELECTERROR(lParam);
   int socket_event = WSAGETSELECTEVENT(lParam);

   if ( 0 != socket_error) {
      char msg[MSG_LEN];

      switch (socket_error) {
case WSAENETDOWN:
sprintf(msg, "ServerSocket %d: WSAENETDOWN", s);
break;
case WSAECONNRESET:
sprintf(msg, "ServerSocket %d: Connection reset by peer.", s);
AS_Control->Panic();
break;
case WSAECONNABORTED:
sprintf(msg, "ServerSocket %d: Connection aborted.", s);
AS_Control->Panic();
break;
default:
sprintf(msg, "ServerSocket %d: Event %d, error %d",
s, socket_event, socket_error);
break;
}
AddLogMsg(msg);
return;
}

switch ( socket_event) {
case FD_ACCEPT:
this->waitForClient();
break;
case FD_CLOSE:
AddLogMsg("Socket: closed");
AS_Control->Panic();
break;
case FD_CONNECT:
break;
case FD_READ:
socketHandler(s);
break;
case FD_WRITE:
break;
default:
{
char msg[128];
sprintf(msg, "Socket event %Xh", socket_event);
AddLogMsg(msg);
}
break;
}
}
*/

// ===================================================================================

int as_ServerSocket::socketHandler(SOCKET s)
{
    int err;
    static char inBuf[MAX_BUFLEN];
    static char strBuf[MAX_BUFLEN];
    static long bufLen = 0;
    static long cmdLen = 0;
    static long dataLenExpected = 0; // signals data transmission
    static long dataLenRecv = 0;
    long dataLenRecvMax = 0;

    char c[2];

    char msg[MSG_LEN];

    // data transmission
    if (0 < dataLenExpected)
    {

        // adjust max reception bytes to size of expected data:
        //
        // expected data length:    [###########################]
        // buffer for recv step 1:  [MAX_BUFLEN]
        // buffer for recv step 2:             [MAX_BUFLEN]
        // buffer for recv step 3:                        [ ??? ]
        //

        if (dataLenExpected > MAX_BUFLEN)
        {
            dataLenRecvMax = MAX_BUFLEN;
        }
        else
        {
            dataLenRecvMax = dataLenExpected;
        }

        //		sprintf(msg, "(Data) Socket receiving %d blocks", dataLenRecvMax);
        //		AddLogMsg(msg);

        dataLenRecv = recv(s, inBuf, dataLenRecvMax, 0);
        if (SOCKET_ERROR == dataLenRecv)
        {
            // reset vars
            dataLenRecv = 0;
            dataLenExpected = 0;
            // get error
            err = WSAGetLastError();
            switch (err)
            {
            case WSAESHUTDOWN:
                AddLogMsg("# Error while receving data: Socket was shut down!");
                return -1;
            case WSAENOTCONN:
                AddLogMsg("# Error while receving data: Socket not connected!");
                return -1;
            case WSAEMSGSIZE:
                AddLogMsg("# Error while receving data: Socket buffer too small: Data in queue");
                return 0;
            case WSAEWOULDBLOCK:
                AddLogMsg("# Error while receving data: Socket function would Block. Trying again...");
                return 0;
            default:
                sprintf(msg, "# Error while receving data: Socket error %d", err);
                AddLogMsg(msg);
                return 0;
            }
        }
        else if (0 == dataLenRecv)
        {
            // reception of 0 byte message may be sign of link shutdown
            AddLogMsg("Received Nothing");
            // reset vars
            dataLenExpected = 0;
            return -1;
        }

        //		sprintf(msg, "Data transmission: %d blocks received", dataLenRecv);
        //		AddLogMsg(msg);

        // process data
        dataLenExpected = AS_Communication->processData(inBuf, dataLenRecv);

        //		sprintf(msg, "Data transmission: now %d blocks expected", dataLenExpected);
        //		AddLogMsg(msg);

        cmdLen = 0;

        // return and wait for next data block
        return 0;
    }

    // initialize buffers
    if (0 == cmdLen)
    {
        inBuf[0] = '\0';
        strBuf[0] = '\0';
        c[0] = '\0';
        c[1] = '\0';
    }

    // command message received char by char
    bufLen = recv(s, c, 1, 0);
    if (SOCKET_ERROR == bufLen)
    {
        // reset buffer
        cmdLen = 0;
        inBuf[0] = '\0';
        strBuf[0] = '\0';
        c[0] = '\0';
        c[1] = '\0';
        // reset vars
        dataLenRecv = 0;
        dataLenExpected = 0;
        // get error
        err = WSAGetLastError();
        switch (err)
        {
        case WSAECONNRESET:
            AddLogMsg("Client disconnected");
            return -1;
        case WSAECONNABORTED:
            AddLogMsg("# Error while receiving message: Connection was aborted!");
            return -1;
        case WSAESHUTDOWN:
            AddLogMsg("# Error while receiving message: Socket was shut down!");
            return -1;
        case WSAENOTCONN:
            AddLogMsg("# Error while receiving message: Socket not connected!");
            return 0;
        case WSAEMSGSIZE:
            AddLogMsg("# Error while receiving message: Socket receive buffer too small: Data in queue");
            return 0;
        case WSAEWOULDBLOCK:
            //				AddLogMsg("# Error while receiving message: Socket function would Block. Trying again...");
            return 0;
        case WSAENOTSOCK:
            AddLogMsg("# Error while receiving message: Invalid Socket!");
            return -1;
        default:
            sprintf(msg, "# Error while receiving message: Socket error %d", err);
            AddLogMsg(msg);
            return 0;
        }
    }
    else if (0 == bufLen)
    {
        // reception of 0 byte message may be sign of link shutdown
        AddLogMsg("# No data read: Socket connection closed!");
        // reset buffer
        cmdLen = 0;
        inBuf[0] = '\0';
        strBuf[0] = '\0';
        c[0] = '\0';
        c[1] = '\0';
        // reset vars
        dataLenRecv = 0;
        dataLenExpected = 0;
        return -1;
    }

    c[1] = '\0';
    if ((1 == bufLen) && ('\n' != c[0]) && ('\r' != c[0]))
    {
        // 1 character received, add to strbuf
        strcat(strBuf, c);
        cmdLen++;
    }

    if ((0 < cmdLen) && ('\n' == c[0]))
    {

        // message received

        //		sprintf(msg, "Message of %d blocks received: '%s'", cmdLen, strBuf);
        //		AddLogMsg(msg);

        // decode and execute command with parameters
        dataLenExpected = AS_Communication->tokenizeMessage(strBuf, cmdLen);

        // reset buffer
        strBuf[0] = '\0';
        c[0] = '\0';
        c[1] = '\0';
        cmdLen = 0;
        // reset vars
        dataLenRecv = 0;
        //dataLenExpected = 0;
    }

    return dataLenExpected;
}

int as_ServerSocket::sendMessage(char *buffer)
{
    int bufLen;
    int rc;

    if (NULL == buffer)
    {
        AddLogMsg("as_ServerSocket::sendBuffer error: invalid buffer");
        return -1;
    }

    bufLen = strlen(buffer);
    if (MAX_BUFLEN < bufLen)
    {
        AddLogMsg("as_ServerSocket::sendBuffer error: buffer too large");
        return -1;
    }

    rc = send(this->cl_handle, buffer, bufLen, 0);
    if (rc != bufLen)
    {
        char msg[MSG_LEN];
        sprintf(msg, "as_ServerSocket::sendBuffer error: could not send buffer, error %d", WSAGetLastError());
        AddLogMsg(msg);
        return -1;
    }
    return 0;
}

void as_ServerSocket::SetPort(unsigned short newport)
{
    char msg[MSG_LEN];
    this->port = newport;
    sprintf(msg, "Setting socket port to %d, please restart app!", newport);
    AddLogMsg(msg);
}

unsigned short as_ServerSocket::GetPort()
{
    return this->port;
}

DWORD WINAPI socketThread(as_ServerSocket *pServerSocket)
{
    DWORD optval = TRUE;
    char msg[MSG_LEN];
    int addrLen;
    int rc;

    // #######################################################################################
    // ### IMPORTANT                                                                       ###
    // ### Initialize COM interface for each thread, else DirectX won't work correctly!!!  ###

    CoInitialize(NULL);

    // ###                                                                                 ###
    // #######################################################################################

    // create socket
    pServerSocket->s_handle = socket(AF_INET, SOCK_STREAM, 0);

    // set socket options
    if (SOCKET_ERROR == setsockopt(
                            pServerSocket->s_handle,
                            SOL_SOCKET,
                            SO_REUSEADDR,
                            (const char *)&optval,
                            sizeof(optval)))
    {
        pServerSocket->printError();
        WSACleanup();
        return -1;
    }

    // get host info
    gethostname(pServerSocket->hostname, 1024);
    pServerSocket->host = gethostbyname(pServerSocket->hostname);
    if (NULL == pServerSocket->host)
    {
        pServerSocket->printError();
        WSACleanup();
        return -1;
    }
    if ((pServerSocket->port < 22) || (pServerSocket->port > 65535))
    {
        pServerSocket->port = SOCKPORT_DEFAULT;
        AddLogMsg("Invalid socket port, setting to default!");
    }
    sprintf(msg, "Host name: %s, port: %d", pServerSocket->host->h_name, pServerSocket->port);
    AddLogMsg(msg);

    // set address family and options
    pServerSocket->address.sin_family = AF_INET;
    pServerSocket->address.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
    pServerSocket->address.sin_family = pServerSocket->host->h_addrtype;
    pServerSocket->address.sin_port = htons(pServerSocket->port);

    // bind socket
    if (SOCKET_ERROR == bind(
                            pServerSocket->s_handle,
                            (struct sockaddr *)&pServerSocket->address,
                            sizeof(pServerSocket->address)))
    {
        pServerSocket->printError();
        WSACleanup();
        return -1;
    }

    // listen socket
    if (SOCKET_ERROR == listen(pServerSocket->s_handle, 1))
    {
        pServerSocket->printError();
        WSACleanup();
        return -1;
    }

    pServerSocket->clients_connected = 0;

    while (1)
    {
        addrLen = sizeof(pServerSocket->address);
        // accept socket
        pServerSocket->cl_handle = accept(
            pServerSocket->s_handle,
            (struct sockaddr *)&pServerSocket->address,
            &addrLen);
        if (SOCKET_ERROR == pServerSocket->cl_handle)
        {
            pServerSocket->printError();
            WSACleanup();
            return -1;
        }

        AddLogMsg("Client connected");
        do
        {
            rc = pServerSocket->socketHandler(pServerSocket->cl_handle);
        } while (-1 != rc);

        AddLogMsg("Client disconnected, waiting for new connection");
        pServerSocket->disconnectClient(); // close socket to client
    }
    return 0;
}
