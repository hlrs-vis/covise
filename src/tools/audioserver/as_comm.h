/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_comm.h
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

#ifndef AS_COMM_H_
#define AS_COMM_H_

#define MAX_CLIENTS 16
#define MAX_PARAMS 32

class as_Comm
{
private:
    char *params[MAX_PARAMS];
    char dataFilename[_MAX_PATH];
    unsigned long dataReceived;
    unsigned long dataExpected;

public:
    bool sendSync(void);
    HWND hWnd;
    as_Comm(HWND hWnd);
    ~as_Comm();
    long tokenizeMessage(char *buf, long bufLen);
    long processData(char *buf, long bufLen);
    long decodeCommand(char *cmd, int numParams);
    long decodeCommand2(char *cmd, int numParams);
    //	void eventHandler( WPARAM wParam, LPARAM lParam );

    class as_Commands *Commands;
    int sendHandle(long handle);
};

extern as_Comm *AS_Communication;
#endif
