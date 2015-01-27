/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_PORT_H
#define _INV_PORT_H

/* $Id: InvPort.h /main/vir_main/1 19-Nov-2001.11:48:19 sasha_te $ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    :  the port connection  for the renderer
//
//
// * Class(es)      : InvPort
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 29.03.94 V 1.0
//
//
//
//**************************************************************************
//
//
//

#include <covise/covise.h>

//
// C stuff
//
#include <unistd.h>

//
// ec stuff
//
#include <covise/covise_appproc.h>

//
// X11 stuff
//
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/keysym.h>

//
// CLASSES
//
class InvPort;

//
// other classes
//
#include "InvDefs.h"
#include "InvError.h"

//
// defines
//
#define MAXDATALEN 500
#define MAXTOKENS 25
#define MAXHOSTLEN 20
#define MAX_REPLACE 32768

extern ApplicationProcess *appmod;
extern int port;
extern char *host;
extern int proc_id;
extern int socket_id;
#ifdef ONCE
XtInputId X_id;
void remove_socket(int)
{
    XtRemoveInput(X_id);
}

#else
extern XtInputId X_id;
extern void remove_socket(int);
#endif

//================================================================
// InvPort
//================================================================

class InvPort
{
private:
    static void socketCommunicationCB(XtPointer client_data, int *source, XtInputId *id);

public:
    InvPort(){};

    void setConnection(XtAppContext app_context, XtPointer client_data)
    {
        socket_id = appmod->get_socket_id(remove_socket);
        X_id = XtAppAddInput(app_context, socket_id,
                             XtPointer(XtInputReadMask),
                             (XtInputCallbackProc)InvPort::socketCommunicationCB, client_data);
    };

    void removeConnection()
    {
        XtRemoveInput(X_id);
    };

    ~InvPort(){};
};
#endif
