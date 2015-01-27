/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLOT_PORT_H
#define _PLOT_PORT_H

/* $Id: PlotPort.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: PlotPort.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    :  the port connection
//
//
// * Class(es)      : PlotPort
//
//
// * inherited from : none
//
//
// * Author  : Uwe Woessner
//
//
// * History : 11.11.94 V 1.0
//
//
//
//**************************************************************************
//
//
//
//

#include <covise/covise.h>
#include <unistd.h>

//
// ec stuff
//
#include <covise/covise_process.h>

//
// X11 stuff
//
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/keysym.h>

//
// CLASSES
//
class PlotPort;

//
// other classes
//
#include "PlotError.h"

//
// defines
//
#define MAXDATALEN 255
#define MAXTOKENS 25
#define MAXHOSTLEN 20

extern int port;
extern char *host;
extern int proc_id;
extern int socket_id;

//================================================================
// PlotPort
//================================================================

class PlotPort
{
private:
    XtInputId X_id;

    static void socketCommunicationCB(XtPointer client_data, int *source, XtInputId *id);

public:
    PlotPort(){};

    void setConnection(XtAppContext app_context, XtPointer client_data)
    {
        X_id = XtAppAddInput(app_context, socket_id,
                             XtPointer(XtInputReadMask),
                             (XtInputCallbackProc)PlotPort::socketCommunicationCB, client_data);
        //  (XtTimerCallbackProc)PlotPort::socketCommunicationCB, client_data);
    };

    void removeConnection()
    {
        XtRemoveInput(X_id);
    };

    ~PlotPort(){};
};
#endif
