/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

// **************************************************************************
//
// * Description    : This is the port connection for the renderer
//
//
// * Class(es)      :  InvPort
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
// **************************************************************************
//
// debug stuff (local use)
//
#include <covise/covise.h>
#define ONCE
#ifdef DEBUG
#define DBG
#endif

//
// ec stuff
//
#include <net/covise_connect.h>
#include <covise/covise_msg.h>
#include <covise/covise_process.h>

#include "InvAnnotationManager.h"

//
// C stuff
//
#include <unistd.h>

#include "InvPort.h"
#include "InvCommunication.h"

//
// Time measurement stuff
//
#ifdef TIMING
#include "InvTimer.h"
#endif

//
// globals
//
extern ApplicationProcess *appmod;
extern char username[100];
extern Widget MasterRequest;

//#########################################################################
// InvPort
//#########################################################################

//=========================================================================
// socket communication callback
//=========================================================================
void InvPort::socketCommunicationCB(XtPointer, int *, XtInputId *)
{
    //char message[255];
    //char DataBuffer[MAXDATALEN];
    char *token[MAXTOKENS];
    char *sep = (char *)"\n";
    char *msgdata; // for buffer data delete
    Message *msg;

#ifdef TIMING
    ap = appmod;
#endif

    //
    // look for message and decide action
    //

    while ((msg = appmod->check_for_ctl_msg()) != nullptr)
    {
        InvCommunication *cm = new InvCommunication();

        if (msg->data.length() > 0)
        {
            //DBG  cerr << "RENDERER: message data length > 0 " << endl;
            token[0] = nullptr;
            token[1] = nullptr;
            token[2] = nullptr;
            token[3] = nullptr;
            msgdata = msg->data.accessData(); 
            cm->parseMessage(msgdata, &token[0], MAXTOKENS, sep);
        }
        else
        {
            msgdata = nullptr;
        }

        switch (msg->type)
        {

        case COVISE_MESSAGE_UI:
            if (strcmp(token[0], "PARREP-S") == 0)
            {
                // 	      cerr << "InvPort::socketCommunicationCB(..) got PAREP-S msg " << token[0] << endl;
                // 	      if (token[1]) cerr << "InvPort::socketCommunicationCB(..) token-1- " <<  token[1] << endl;
                // 	      if (token[2]) cerr << "InvPort::socketCommunicationCB(..) token-2- " <<  token[2] << endl;
                // 	      if (token[3]) cerr << "InvPort::socketCommunicationCB(..) token-3- " <<  token[3] << endl;
                // 	      if (token[4]) cerr << "InvPort::socketCommunicationCB(..) token-4- " <<  token[4] << endl;
                // 	      if (token[5]) cerr << "InvPort::socketCommunicationCB(..) token-5- " <<  token[5] << endl;
                // 	      if (token[6]) cerr << "InvPort::socketCommunicationCB(..) token-6- " <<  token[6] << endl;
                // 	      if (token[7]) cerr << "InvPort::socketCommunicationCB(..) token-7- " <<  token[7] << endl;

                Annotations->update(token[7]);
            }
            else if (strcmp(token[0], "INEXEC") == 0)
            {
                XtSetSensitive(MasterRequest, false);
            }
            else if (strcmp(token[0], "FINISHED") == 0)
            {
                if (!rm_isMaster())
                {
                    XtSetSensitive(MasterRequest, true);
                }
            }
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_QUIT:

            print_comment(__LINE__, __FILE__, "Module correctly finished");

            cm->receiveDeleteAll();
            //cm->sendQuitMessage();  //writing to a closed socket
            delete appmod;
            delete msg;
            exit(1);
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_RENDER:

            if (strcmp(token[0], "MASTER") == 0)
            {
                cm->receiveMasterMessage(token[1], token[2], token[3]);
            }

            else if (strcmp(token[0], "SLAVE") == 0)
            {
                cm->receiveSlaveMessage(token[1], token[2], token[3]);
            }

            else if (strcmp(token[0], "UPDATE") == 0)
            {
                cm->receiveUpdateMessage(token[1], token[2], token[3]);
            }

            else if (strcmp(token[0], "CAMERA") == 0)
            {
                cm->receiveCameraMessage(token[1]);
            }

            else if (strcmp(token[0], "TRANSFORM") == 0)
            {
                cm->receiveTransformMessage(token[1]);
            }

            else if (strcmp(token[0], "TELEPOINTER") == 0)
            {
                cm->receiveTelePointerMessage(token[1]);
            }

            else if (strcmp(token[0], "DRAWSTYLE") == 0)
            {
                cm->receiveDrawstyleMessage(token[1]);
            }

            else if (strcmp(token[0], "LIGHTMODE") == 0)
            {
                cm->receiveLightModeMessage(token[1]);
            }

            else if (strcmp(token[0], "SELECTION") == 0)
            {
                cm->receiveSelectionMessage(token[1]);
            }

            else if (strcmp(token[0], "DESELECTION") == 0)
            {
                cm->receiveDeselectionMessage(token[1]);
            }

            //// part switching  has arrived
            else if (strcmp(token[0], "PART") == 0)
            {
                cm->receivePartMessage(token[1]);
            }

            //// reference part has arrived
            else if (strcmp(token[0], "REFPART") == 0)
            {
                cm->receiveReferencePartMessage(token[1]);
            }

            //// reset scene has arrived
            else if (strcmp(token[0], "RESET") == 0)
            {
                cm->receiveResetSceneMessage();
            }

            //// transparency level has arrived
            else if (strcmp(token[0], "TRANSPARENCY") == 0)
            {
                cm->receiveTransparencyMessage(token[1]);
            }

            //// sync change has arrived
            else if (strcmp(token[0], "SYNC") == 0)
            {
                cm->receiveSyncModeMessage(token[1]);
            }

            //// fog change has arrived
            else if (strcmp(token[0], "FOG") == 0)
            {
                cm->receiveFogMessage(token[1]);
            }

            //// antialiasing change has arrived
            else if (strcmp(token[0], "ANTIALIASING") == 0)
            {
                cm->receiveAntialiasingMessage(token[1]);
            }

            //// backcolor change has arrived
            else if (strcmp(token[0], "BACKCOLOR") == 0)
            {
                cm->receiveBackcolorMessage(token[1]);
            }

            //// axis change has arrived
            else if (strcmp(token[0], "AXIS") == 0)
            {
                cm->receiveAxisMessage(token[1]);
            }

            //// clipping plane change has arrived
            else if (strcmp(token[0], "CLIP") == 0)
            {
                cm->receiveClippingPlaneMessage(token[1]);
            }

            //// viewing change has arrived
            else if (strcmp(token[0], "VIEWING") == 0)
            {
                cm->receiveViewingMessage(token[1]);
            }

            //// projection change has arrived
            else if (strcmp(token[0], "PROJECTION") == 0)
            {
                cm->receiveProjectionMessage(token[1]);
            }

            //// decoration change has arrived
            else if (strcmp(token[0], "DECORATION") == 0)
            {
                cm->receiveDecorationMessage(token[1]);
            }

            //// headlight change has arrived
            else if (strcmp(token[0], "HEADLIGHT") == 0)
            {
                cm->receiveHeadlightMessage(token[1]);
            }

            //// colormap change has arrived
            else if (strcmp(token[0], "COLORMAP") == 0)
            {
                cm->receiveColormapMessage(token[1]);
            }

            //// sequencer change has arrived
            else if (strcmp(token[0], "SEQUENCER") == 0)
            {
                cm->receiveSequencerMessage(token[1]);
            }
            else if (strcmp(token[0], "ANNOTATION") == 0)
            {
                Annotations->update(token[7]);
            }

            //// user name has arrived
            else if (strcmp(token[0], "USERNAME") == 0)
            {
                int mod_id;
                if (sscanf(token[1], "%d", &mod_id) != 1)
                {
                    fprintf(stderr, "InvPort::socketCommunicationCB: sscanf failed\n");
                }
                if (mod_id == appmod->get_id())
                {

                    strcpy(username, token[2]);

                    char name[32]; // add the machine name
                    gethostname(name, 24);

                    if (username[0])
                    {
                        strcat(username, "@");
                        strcat(username, name);
                    }
                    else
                    {
                        strcpy(username, name);
                    }
                }
            }
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_ADD_OBJECT:

            print_comment(__LINE__, __FILE__, "got add object message");
            if (strcmp(token[0], "") != 0)
            {
                // AW: try to create the object, so we know whether it is empty
                const coDistributedObject *data_obj = nullptr;
                data_obj = coDistributedObject::createFromShm(token[0]);
                if (data_obj)
                {
                    cm->receiveAddObjectMessage(data_obj, token[0], 0);
                    Annotations->showAll();
                    Annotations->hasDataObj(true);
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
            }
            cm->sendFinishMessage();
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_DELETE_OBJECT:

            print_comment(__LINE__, __FILE__, "got delete object message");

            if (token[0] && strcmp(token[0], "") != 0)
                cm->receiveDeleteObjectMessage(token[0]);
            else
                print_comment(__LINE__, __FILE__, "got empty object name");

            cm->sendFinishMessage();
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_REPLACE_OBJECT:

            if (token[0] == nullptr || token[1] == nullptr)
                print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
            else
            {
                // AW: try to create the object, so we know whether it is empty
                const coDistributedObject *data_obj = nullptr;
                data_obj = coDistributedObject::createFromShm(token[1]);

                // if both object's names are different
                if (strcmp(token[0], token[1]) != 0)
                {
                    print_comment(__LINE__, __FILE__,
                                  "got replace object message with different names");

                    // AW: only remove Object if new obj exists
                    if (strcmp(token[0], "") != 0)
                        if (data_obj)
                            cm->receiveDeleteObjectMessage(token[0]);
                        else
                            cm->setReplace(token[0], token[1]);
                    else
                        print_comment(__LINE__, __FILE__, "got empty object name");

                    if (data_obj)
                        cm->receiveAddObjectMessage(data_obj, token[1], 0);
                    else
                        print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
                }

                else // token[0] == token[1]
                {
                    print_comment(__LINE__, __FILE__, "got replace object message");
                    if (data_obj)
                        // 1: replace object
                        cm->receiveAddObjectMessage(data_obj, token[0], 1);
                    else
                        print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
                }
            }
            cm->sendFinishMessage();
            break;
        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_REMOVED_HOST:
            char *tp_text;
            char *tmp;
            // removing .xxx part of the hostname
            // for compatibility with USERNAME message handling
            tmp = strchr(token[1], '.');
            if (tmp)
            {
                tmp[0] = '\0';
            }
            tp_text = new char[strlen(token[0]) + strlen(token[1]) + 30];
            sprintf(tp_text, "%s@%s %d 0 0 0 0", token[0], token[1], CO_RMV);

            // removing Telepointer
            cm->receiveTelePointerMessage(tp_text);
            delete[] tp_text;
            break;
        default:
            break;

        } // end switch
        delete cm;
        delete msg;

    } // end while
}
