/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log: PlotPort.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//static char rcsid[] = "$Id: PlotPort.C,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $";

//**************************************************************************
//
// * Description    : This is the port connection for the renderer
//
//
// * Class(es)      :  PlotPort
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau, Uwe Woessner
//
//
// * History : 29.03.94 V 1.0
//
//
//
//**************************************************************************
//
// debug stuff (local use)
//
#include <covise/covise.h>
#ifdef DEBUG
#define DBG
#endif

//
// ec stuff
//
#include <net/covise_connect.h>
#include <net/message.h>
#include <covise/covise_process.h>
#include <covise/covise_appproc.h>
#include <do/coDoData.h>

#include <unistd.h>

#include "PlotPort.h"
#include "PlotCommunication.h"

using namespace covise;
//
// globals
//
PlotCommunication *cm;
void drawgraph(void);
extern class ApplicationProcess *appmod;

extern void bailout(); /* from xmgr.C , the exit routine */

//#########################################################################
// PlotPort
//#########################################################################

//=========================================================================
// socket communication callback
//=========================================================================
void PlotPort::socketCommunicationCB(XtPointer, int *, XtInputId *)
{
    // char message[255];
    // char DataBuffer[MAXDATALEN];
    char *token[MAXTOKENS];
    char *sep = (char *)"\n";
    Message *msg;

    //
    // look for message and decide action
    //

    while ((msg = appmod->check_for_ctl_msg()) != 0)
    {
        cm = new PlotCommunication();

        if (msg->data.length() > 0)
        {
            //DBG  cerr << "RENDERER: message data length > 0 " << endl;
            cm->parseMessage(msg->data.accessData(), &token[0], MAXTOKENS, sep);
        }

        if (msg->type != COVISE_MESSAGE_QUIT && token[0] == NULL)
        {
            print_comment(__LINE__, __FILE__, "received invalid message");
        }
        else
        {
            switch (msg->type)
            {

            //
            // QUIT
            //
            case COVISE_MESSAGE_QUIT:
                print_comment(__LINE__, __FILE__, "Module correctly finished");
                cm->sendQuitMessage();
                delete appmod;
                delete msg;
                bailout();
                break;

            //
            // RENDER
            //
            case COVISE_MESSAGE_RENDER:
                if (strcmp(token[0], "MASTER") == 0)
                {
                    print_comment(__LINE__, __FILE__, "got master message  \n");
                    cm->receiveMasterMessage();
                }
                if (strcmp(token[0], "SLAVE") == 0)
                {
                    print_comment(__LINE__, __FILE__, "got slave message  \n");
                    cm->receiveSlaveMessage();
                }
                if (strcmp(token[0], "COMMAND") == 0)
                {
                    cm->receiveCommandMessage(token[1]);
                }
                if (strcmp(token[0], "COMMAND_V") == 0)
                {
                    cm->receiveCommand_ValuesMessage(token[1]);
                }
                if (strcmp(token[0], "COMMAND_S") == 0)
                {
                    cm->receiveCommand_StringMessage(token[1]);
                }
                if (strcmp(token[0], "COMMAND_F") == 0)
                {
                    cm->receiveCommand_FloatMessage(token[1]);
                }
                break;

            //
            // ADD_OBJECT
            //
            case COVISE_MESSAGE_ADD_OBJECT:
                print_comment(__LINE__, __FILE__, "got add object message");

                if ((strcmp(token[0], "") != 0) && (strcmp(token[0], " ") != 0))
                {
                    cm->receiveAddObjectMessage(token[0], 0);

                    cm->sendFinishMessage();
                }
                else
                {
                    print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
                    cm->sendFinishMessage();
                }
                break;

            //
            //  DELETE_OBJECT
            //
            case COVISE_MESSAGE_DELETE_OBJECT:
                print_comment(__LINE__, __FILE__, "got delete object message");

                if (token[0] && (strcmp(token[0], "") != 0) && (strcmp(token[0], " ") != 0))
                {

                    cm->receiveDeleteObjectMessage(token[0]);

                    cm->sendFinishMessage();
                }
                else
                {
                    print_comment(__LINE__, __FILE__, "got empty object name");
                    cm->sendFinishMessage();
                }
                drawgraph();
                break;

            //
            // REPLACE_OBJECT
            //
            case COVISE_MESSAGE_REPLACE_OBJECT:
                if (token[0] == NULL || token[1] == NULL)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
                }
                else if (strcmp(token[0], token[1]) != 0)
                {
                    print_comment(__LINE__, __FILE__, "got replace object message with different names");

                    if ((strcmp(token[0], "") != 0) && (strcmp(token[0], " ") != 0))
                    {

                        cm->receiveDeleteObjectMessage(token[0]);
                    }
                    else
                    {
                        print_comment(__LINE__, __FILE__, "got empty object name");
                    }
                    if ((strcmp(token[0], "") != 0) && (strcmp(token[0], " ") != 0))
                    {

                        cm->receiveAddObjectMessage(token[1], 0);
                    }
                    else
                    {
                        print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
                    }
                }
                else
                {
                    print_comment(__LINE__, __FILE__, "got replace object message");

                    if ((strcmp(token[0], "") != 0) && (strcmp(token[0], " ") != 0))
                    {
                        cm->receiveAddObjectMessage(token[0], 1);
                    }
                    else
                    {
                        print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
                    }
                }
                cm->sendFinishMessage();
                break;
            default:
                break;

            } // end switch
        }
        delete cm;
        delete msg;
    } // end while
}
