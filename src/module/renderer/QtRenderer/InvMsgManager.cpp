/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <qsocketnotifier.h>
#include <qmessagebox.h>
#include <qtextedit.h>
#include <qlineedit.h>
#include <qstringlist.h>

//
// ec stuff
//
#include <covise/covise.h>
#include <covise/covise_process.h>
#include <covise/covise_msg.h>
#include <covise/covise_appproc.h>

//
// C stuff
//
#ifndef _WIN32
#include <unistd.h>
#endif

#ifndef YAC
#include "InvMain.h"
#else
#include "InvMain_yac.h"
#endif
#include "InvMsgManager.h"
#include "InvCommunicator.h"
#include "InvDefs.h"

void remove_socket(int)
{
    fprintf(stderr, "X Socket Connection was removed from QtRenderer\n");
}

//======================================================================

InvMsgManager::InvMsgManager()
    : QObject()
{
    QSocketNotifier *sn;

    renderer->socket_id = renderer->appmod->get_socket_id(remove_socket);
    sn = new QSocketNotifier(renderer->socket_id, QSocketNotifier::Read);

    QObject::connect(sn, SIGNAL(activated(int)),
                     this, SLOT(dataReceived(int)));
}

InvMsgManager::~InvMsgManager()
{
}

//------------------------------------------------------------------------
// receive data on socket
//------------------------------------------------------------------------
void InvMsgManager::dataReceived(int)
{
    Message *msg;
    QStringList list;

    //
    // look for message and decide action
    //

    while ((msg = renderer->appmod->check_for_ctl_msg()) != NULL)
    {
        if (msg->data.length() > 0)
        {
            //cerr << "_____________________________________________" << endl;
            //cerr << msg->data.data() << endl;
            //cerr << "_____________________________________________" << endl;
            list = QString(msg->data.data()).split("\n");
        }

        /* otherwise quit messages are ignored else
      {
         print_comment(__LINE__,  __FILE__, "empty message");
         return;
      }*/

        switch (msg->type)
        {
        /*
                  case UI:
                     if(list[0] == "PARREP-S")
                        {
                        // 	      cerr << "InvPort::socketCommunicationCB(..) got PAREP-S msg " << list[0] << endl;
                        // 	      if (list[1]) cerr << "InvPort::socketCommunicationCB(..) token-1- " <<  list[1] << endl;
                        // 	      if (list[2]) cerr << "InvPort::socketCommunicationCB(..) token-2- " <<  list[2] << endl;
                        // 	      if (list[3]) cerr << "InvPort::socketCommunicationCB(..) token-3- " <<  list[3] << endl;
                        // 	      if (list[4]) cerr << "InvPort::socketCommunicationCB(..) token-4- " <<  list[4] << endl;
                        // 	      if (list[5]) cerr << "InvPort::socketCommunicationCB(..) token-5- " <<  list[5] << endl;
                        // 	      if (list[6]) cerr << "InvPort::socketCommunicationCB(..) token-6- " <<  list[6] << endl;
         // 	      if (list[7]) cerr << "InvPort::socketCommunicationCB(..) token-7- " <<  list[7] << endl;

         //Annotations->update(list[7]);
         }

         else if(list[0] == "INEXEC")
         {
         XtSetSensitive( MasterRequest, false );
         }
         else if(list[0] == "FINISHED")
         {
         if( renderer->isMaster() )
         {
         XtSetSensitive( MasterRequest, true );
         }
         }
         break;*/

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_QUIT:

            print_comment(__LINE__, __FILE__, "Module correctly finished");
            renderer->cm->receiveDeleteAll();
            delete renderer->appmod;
            delete msg;
            exit(1);
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_RENDER:

            if (list[0] == "MASTER")
            {
                //renderer->cm->receiveMasterMessage(list[1],list[2],list[3]);
                renderer->cm->receiveMasterMessage(list[1]);
            }

            else if (list[0] == "SLAVE")
            {
                renderer->cm->receiveSlaveMessage(list[1]);
            }

            else if (list[0] == "UPDATE")
            {
                renderer->cm->receiveUpdateMessage(list[1], list[2], list[3]);
            }

            else if (list[0] == "CAMERA")
            {
                renderer->cm->receiveCameraMessage(list[1]);
            }

            else if (list[0] == "TRANSFORM")
            {
                renderer->cm->receiveTransformMessage(list[1]);
            }

            else if (list[0] == "TELEPOINTER")
            {
                renderer->cm->receiveTelePointerMessage(list[1]);
            }

            else if (list[0] == "DRAWSTYLE")
            {
                renderer->cm->receiveDrawstyleMessage(list[1]);
            }

            else if (list[0] == "LIGHTMODE")
            {
                renderer->cm->receiveLightModeMessage(list[1]);
            }

            else if (list[0] == "SELECTION")
            {
                renderer->cm->receiveSelectionMessage(list[1]);
            }

            else if (list[0] == "DESELECTION")
            {
                renderer->cm->receiveDeselectionMessage(list[1]);
            }

            //// part switching  has arrived
            else if (list[0] == "PART")
            {
                renderer->cm->receivePartMessage(list[1]);
            }

            //// reference part has arrived
            else if (list[0] == "REFPART")
            {
                renderer->cm->receiveReferencePartMessage(list[1]);
            }

            //// reset scene has arrived
            else if (list[0] == "RESET")
            {
                renderer->cm->receiveResetSceneMessage();
            }

            //// transparency level has arrived
            else if (list[0] == "TRANSPARENCY")
            {
                renderer->cm->receiveTransparencyMessage(list[1]);
            }

            //// sync change has arrived
            else if (list[0] == "SYNC")
            {
                renderer->cm->receiveSyncModeMessage(list[1]);
            }

            //// fog change has arrived
            else if (list[0] == "FOG")
            {
                renderer->cm->receiveFogMessage(list[1]);
            }

            //// antialiasing change has arrived
            else if (list[0] == "ANTIALIASING")
            {
                renderer->cm->receiveAntialiasingMessage(list[1]);
            }

            //// backcolor change has arrived
            else if (list[0] == "BACKCOLOR")
            {
                renderer->cm->receiveBackcolorMessage(list[1].toLatin1());
            }

            //// axis change has arrived
            else if (list[0] == "AXIS")
            {
                renderer->cm->receiveAxisMessage(list[1].toLatin1());
            }

            //// clipping plane change has arrived
            else if (list[0] == "CLIP")
            {
                renderer->cm->receiveClippingPlaneMessage(list[1].toLatin1());
            }

            //// viewing change has arrived
            else if (list[0] == "VIEWING")
            {
                renderer->cm->receiveViewingMessage(list[1]);
            }

            //// projection change has arrived
            else if (list[0] == "PROJECTION")
            {
                renderer->cm->receiveProjectionMessage(list[1]);
            }

            //// decoration change has arrived
            else if (list[0] == "DECORATION")
            {
                renderer->cm->receiveDecorationMessage(list[1]);
            }

            //// headlight change has arrived
            else if (list[0] == "HEADLIGHT")
            {
                renderer->cm->receiveHeadlightMessage(list[1]);
            }

            //// colormap change has arrived
            else if (list[0] == "COLORMAP")
            {
                renderer->cm->receiveColormapMessage(list[1]);
            }

            //// sequencer change has arrived
            else if (list[0] == "SEQUENCER")
            {
                renderer->cm->receiveSequencerMessage(list[1]);
            }

            else if (list[0] == "ANNOTATION")
            {
                //Annotations->update(list[7]);
            }

            //// user name has arrived
            else if (list[0] == "USERNAME")
            {
                int mod_id = list[1].toInt();
                if (mod_id == renderer->appmod->get_id())
                {
                    renderer->m_username = list[2];

                    //const char nname[24];                  // add the machine name
                    //gethostname(nname,24);
                    //QString name = nname;

                    if (!renderer->m_username.isEmpty())
                    {
                        renderer->m_username.append("@");
                        renderer->m_username.append(renderer->hostname);
                    }
                    else
                    {
                        renderer->m_username = renderer->hostname;
                    }
                }
            }
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_ADD_OBJECT:

            print_comment(__LINE__, __FILE__, "got add object message");
            if (strcmp(list[0].toLatin1(), "") != 0)
            {
                // AW: try to create the object, so we know whether it is empty
                const coDistributedObject *data_obj = coDistributedObject::createFromShm(coObjInfo(list[0].toLatin1()));
                if (data_obj)
                {
                    renderer->cm->receiveAddObjectMessage(data_obj, (char *)(const char *)list[0].toLatin1(), 0);
                    //Annotations->showAll();
                    //Annotations->hasDataObj(true);
                }
            }

            else
            {
                print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
            }
            renderer->cm->sendFinishMessage();
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_DELETE_OBJECT:

            print_comment(__LINE__, __FILE__, "got delete object message");

            if (!list.isEmpty() && !list[0].isNull())
            {
                renderer->cm->receiveDeleteObjectMessage(list[0]);
            }
            else
                print_comment(__LINE__, __FILE__, "got empty object name");

            renderer->cm->sendFinishMessage();
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_REPLACE_OBJECT:

            if (!(list.size() > 1) || list[0].isNull() || list[1].isNull())
                print_comment(__LINE__, __FILE__, "ERROR: got empty object name");

            else
            {
                // AW: try to create the object, so we know whether it is empty
                const coDistributedObject *data_obj = coDistributedObject::createFromShm(coObjInfo(list[1].toLatin1()));
                // if both object's names are different
                if (list[0] != list[1])
                {
                    print_comment(__LINE__, __FILE__,
                                  "got replace object message with different names");

                    // AW: only remove Object if new obj exists
                    if (!list[0].isNull())
                        if (data_obj)
                            renderer->cm->receiveDeleteObjectMessage(list[0]);
                        else
                            renderer->cm->setReplace(list[0], list[1]);
                    else
                        print_comment(__LINE__, __FILE__, "got empty object name");

                    if (data_obj)
                        renderer->cm->receiveAddObjectMessage(data_obj, list[1].toLatin1(), 0);
                    else
                        print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
                }

                else // list[0] == list[1]
                {
                    print_comment(__LINE__, __FILE__, "got replace object message");
                    if (data_obj)
                        // 1: replace object
                        renderer->cm->receiveAddObjectMessage(data_obj, list[0].toLatin1(), 1);
                    else
                        print_comment(__LINE__, __FILE__, "ERROR: got empty object name");
                }
            }
            renderer->cm->sendFinishMessage();
            break;

        ////////////////////////////////////////////////////////////////////////////////
        case COVISE_MESSAGE_REMOVED_HOST:
        /*char* tp_text;
            char* tmp;
            // removing .xxx part of the hostname
            // for compatibility with USERNAME message handling
            tmp = strchr(list[1],'.');
            if(tmp)
            {
               tmp[0] = '\0';
            }
            tp_text = new char[strlen(list[0])+strlen(list[1])+30];
            sprintf(tp_text,"%s@%s %d 0 0 0 0",list[0],list[1],CO_RMV);

            // removing Telepointer
            renderer->cm->receiveTelePointerMessage(tp_text);
            delete []tp_text;
            break;*/

        default:
            break;
        }
        delete msg;

    } // end while
}
