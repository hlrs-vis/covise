/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QDebug>
#include <QMessageBox>
#include <QSocketNotifier>
#include <QTimer>
#include <QApplication>
#include <QClipboard>
#include <QMimeData>

#include <util/coLog.h>
#include <covise/covise_msg.h>
#include <covise/covise_appproc.h>
#include <net/concrete_messages.h>

#include "MEMessageHandler.h"
#include "MEFileBrowser.h"
#include "handler/MEMainHandler.h"
#include "handler/MENodeListHandler.h"
#include "handler/MEHostListHandler.h"
#include "widgets/MEUserInterface.h"
#include "widgets/MEGraphicsView.h"
#include "nodes/MENode.h"
#include "ports/MEFileBrowserPort.h"
#include "ports/MEDataPort.h"

MEMessageHandler *MEMessageHandler::singleton = NULL;

using covise::Message;

void remove_socket(int)
{
    qWarning() << "X Socket Connection was removed from QtMEUserInterface::instance()\n";
}

/*!
    \class MEMessageHandler
    \brief Handler for sending and receiving messages from controller, crb, modules & other UI
*/

MEMessageHandler::MEMessageHandler(int argc, char **argv)
    : QObject()
    , m_clonedNode(NULL)
    , m_periodictimer(NULL)
{

    singleton = this;
    // check modus
    m_standalone = false;
    m_userInterface = new covise::UserInterface((char*)"AppModule", argc, argv);

    // make socket
    int TCP_Socket = m_userInterface->get_socket_id(remove_socket);
    QSocketNotifier* sn = new QSocketNotifier(TCP_Socket, QSocketNotifier::Read);

    // weil unter windows manchmal Messages verloren gehen
    // der SocketNotifier wird nicht oft genug aufgerufen)
#if defined(_WIN32) || defined(__APPLE__)
    m_periodictimer = new QTimer;
    QObject::connect(m_periodictimer, SIGNAL(timeout()), this, SLOT(handleWork()));
    m_periodictimer->start(1000);
#endif

    QObject::connect(sn, SIGNAL(activated(int)), this, SLOT(dataReceived(int)));
}

MEMessageHandler *MEMessageHandler::instance()
{
    return singleton;
}

MEMessageHandler::~MEMessageHandler()
{
    if (m_periodictimer)
    {
        m_periodictimer->stop();
        delete m_periodictimer;
    }
}

void MEMessageHandler::handleWork()
{
    dataReceived(1);
}

//!
//! send an UI message
//!
void MEMessageHandler::sendMessage(int type, const QString &text)
{
    QByteArray ba = text.toUtf8();
    Message msg{ (covise::covise_msg_type)type , covise::DataHandle{ba.data(),strlen(ba.data()) + 1, false } };

#if 0
   qDebug() << "Message send _____________________________ ";
   qDebug() << msg->data;
   qDebug() << "__________________________________________ ";
#endif

    if (m_userInterface)
        m_userInterface->send_ctl_msg(&msg);
}

//!
//! receive data on TCP/IP socket
//!
void MEMessageHandler::dataReceived(int)
{
    MENode *node;
    MEParameterPort *pport;
    QString buffer, ptype;

    if (MEMainHandler::instance())
    {
        while (Message *msg = m_userInterface->check_for_msg())
        {
            // empty message
            if (msg->data.length() == 0)
            {
                covise::print_comment(__LINE__, __FILE__, "empty message");
            }
            else
            {
                QStringList list = QString(msg->data.data()).split("\n", QString::SkipEmptyParts);

#if 0
            qDebug() << "Message received _________________________";
            qDebug() << msg->type;
            qDebug() << msg->data.data();
            qDebug() << "__________________________________________";
#endif

                int nitem = list.count();

                switch (msg->type)
                {
                case covise::COVISE_MESSAGE_CRB_EXEC:
                {
                    covise::CRB_EXEC exec{*msg};
                    if (!strcmp(exec.name ,"ViNCE") || !strcmp(exec.name,"Renderer"))
                        MEUserInterface::instance()->startRenderer(exec);
                }
                break;

                //
                // QUIT
                //
                case covise::COVISE_MESSAGE_QUIT:
                    MEMainHandler::instance()->quit();
                    break;

                //
                // PARINFO (module send updates for parameters)
                //
                case covise::COVISE_MESSAGE_PARINFO:
                    node = MENodeListHandler::instance()->getNode(list[0], list[1], list[2]);
                    if (node != NULL)
                    {
                        // portname
                        pport = node->getParameterPort(list[3]);
                        ptype = list[4]; // porttype not used
                        if (pport != NULL) // portvalue
                            pport->modifyParameter(list[5]);
                    }
                    break;

                //
                // START(module)
                //
                case covise::COVISE_MESSAGE_START:
                    MEMainHandler::instance()->startNode(list[0], list[1], list[2]);
                    break;

                //
                // FINISHED (module)
                //
                case covise::COVISE_MESSAGE_FINISHED:
                    MEMainHandler::instance()->finishNode(list);
                    break;

                //
                // ERROR
                //
                case covise::COVISE_MESSAGE_COVISE_ERROR:

                    if (MEMainHandler::instance()->cfg_ErrorHandling)
                    {
                        // from module
                        if (nitem > 1)
                        {
                            buffer = list[0];
                            buffer.append("_");
                            buffer.append(list[1]);
                            buffer.append("@");
                            buffer.append(list[2]);
                            QMessageBox::critical(0, buffer, list[3]);
                        }

                        // from controller
                        else
                        {
                            QMessageBox::critical(0, "Controller:", list[0]);
                        }
                    }

                    else
                    {

                        // from module
                        if (nitem > 1)
                        {
                            buffer = list[0];
                            buffer.append("_");
                            buffer.append(list[1]);
                            buffer.append("@");
                            buffer.append(list[2]);
                            buffer.append(": ");
                            buffer.append(list[3]);
                        }

                        // from controller
                        else
                        {
                            buffer = "Controller: ";
                            buffer.append(list[0]);
                        }
                        MEUserInterface::instance()->writeInfoMessage(buffer, Qt::red);
                    }
                    break;

                //
                // WARNING  (message from module)
                //
                case covise::COVISE_MESSAGE_WARNING:

                    // from module
                    if (nitem > 1)
                    {
                        buffer = list[0];
                        buffer.append("_");
                        buffer.append(list[1]);
                        buffer.append("@");
                        buffer.append(list[2]);
                        buffer.append(": ");
                        buffer.append(list[3]);
                    }

                    // from controller
                    else
                    {
                        buffer = "Controller: ";
                        buffer.append(list[0]);
                    }
                    MEUserInterface::instance()->writeInfoMessage(buffer, Qt::blue);
                    break;

                //
                // INFO (message from module)
                //
                case covise::COVISE_MESSAGE_INFO:
                    // chat message from other host
                    if (list[0] == "CHAT")
                    {
                        MEUserInterface::instance()->writeChatContent(list);
                    }

                    // from module
                    else if (nitem > 1)
                    {
                        buffer = list[0];
                        buffer.append("_");
                        buffer.append(list[1]);
                        buffer.append("@");
                        buffer.append(list[2]);
                        buffer.append(": ");
                        buffer.append(list[3]);
                        MEUserInterface::instance()->writeInfoMessage(buffer, Qt::darkGreen);
                    }

                    // from controller
                    else
                    {
                        buffer = "Controller: ";
                        buffer.append(list[0]);
                        QMessageBox::critical(0, "Controller", list[0]);
                    }

                    break;

                case covise::COVISE_MESSAGE_UI:
                    receiveUIMessage(msg);
                    break;

                case covise::COVISE_MESSAGE_LAST_DUMMY_MESSAGE:
                {
                    Message msg2{ covise::COVISE_MESSAGE_LAST_DUMMY_MESSAGE , covise::DataHandle{(char*)" ", 2, false} };
                    m_userInterface->send_ctl_msg(&msg2);
                    break;
                }

                case covise::COVISE_MESSAGE_PARAMDESC:
                    // used in python interface
                    break;

                default:
                    qCritical() << "======> unknown message type" << msg->type;
                    qCritical() << "======> ... data = " << msg->data.data();
                    break;
                }
            }
            m_userInterface->delete_msg(msg);
        }
    }
}

//!
//! parse an UI message
//!
void MEMessageHandler::receiveUIMessage(Message *msg)
{

    const char *data = msg->data.data();

    QString mname, hname, number;
    QString ptype, pname, pvalue;
    int nr;
    MENode *node;
    MEDataPort *port;
    MEParameterPort *pport;

    QStringList list = QString::fromUtf8(data).split("\n");

    //
    // request of a slave user interface for master
    //
    if (list[0] == "MASTERREQ" && MEMainHandler::instance()->isMaster())
    {
        QString text = "Grant master status to ";
        text.append(list[1]);
        text.append("@");
        text.append(list[2]);
        text.append("?");

        QString data;
        switch (QMessageBox::question(MEUserInterface::instance(), "COVISE Map Editor - Grant Master", text, "Grant", "Deny", "", 0, 1))
        {

        case 0:
            data = "STATUS\nSLAVE\n";
            MEMainHandler::instance()->setMaster(false);
            break;

        case 1:
            data = "STATUS\nMASTER\n";
            MEMainHandler::instance()->setMaster(true);
            break;
        }

        sendMessage(covise::COVISE_MESSAGE_UI, data);
    }

    //
    else if (list[0].contains("MASTER"))
    {
        MEMainHandler::instance()->setMaster(true);
        if (list[0] == "MASTER_RESTART")
            MEMainHandler::instance()->setInMapLoading(true);

        if (list.size() > 1 && list[1] == "MINI_GUI")
            MEUserInterface::instance()->setMiniGUI(true);
        else
            MEUserInterface::instance()->setMiniGUI(false);
    }

    //
    else if (list[0].contains("SLAVE"))
    {
        MEMainHandler::instance()->setMaster(false);
        if (list[0] == "SLAVE_RESTART")
            MEMainHandler::instance()->setInMapLoading(true);

        if (list.size() > 1 && list[1] == "MINI_GUI")
            MEUserInterface::instance()->setMiniGUI(true);
        else
            MEUserInterface::instance()->setMiniGUI(false);
    }

    //
    else if (list[0] == "UNDO_BUFFER_TRUE")
    //
    {
        MEUserInterface::instance()->enableUndo(true);
    }

    //
    else if (list[0] == "UNDO_BUFFER_FALSE")
    //
    {
        MEUserInterface::instance()->enableUndo(false);
    }

    //
    else if (list[0] == "UPDATE_LOADED_MAPNAME")
    //
    {
        MEMainHandler::instance()->updateLoadedMapname(list[1]);
    }

    //
    else if (list[0] == "CONVERTED_NET_FILE")
    //
    {
        QMessageBox::critical(0, "COVISE Controller - Old Network File Format",
                              "The file " + list[1] + " was in an old format and has been converted to the current format.\n" + "The original network file has been renamed to " + list[1] + ".bak.");
    }

    else if (list[0] == "FAILED_NET_FILE_CONVERSION")
    //
    {
        QMessageBox::critical(0, "COVISE Controller - Unknown Network File Format",
                              "The file " + list[1] + " is in an unknown format and could not be converted to a network file.\n");
    }

    //
    else if (list[0] == "START_READING")
    //
    {
        MEMainHandler::instance()->setInMapLoading(true);
        MEMainHandler::instance()->enableExecution(false);
        if (list.count() > 1 && !list[1].isEmpty())
            MEMainHandler::instance()->storeMapName(list[1]);
    }

    //
    else if (list[0] == "END_READING")
    //
    {
        MEGraphicsView::instance()->setAutoSelectNewNodes(false);
        MEMainHandler::instance()->setInMapLoading(false);
        MEMainHandler::instance()->enableExecution(true);
        if (list[1] == "true")
        {
            MEGraphicsView::instance()->showMap();
            MEMainHandler::instance()->setMapModified(false);
        }
        if (MEMainHandler::instance()->isExecOnChange())
            MEMainHandler::instance()->execNet();
    }

    //
    else if (list[0] == "ICONIFY")
    //
    {
        MEUserInterface::instance()->showMinimized();
    }

    //
    else if (list[0] == "MAXIMIZE")
    //
    {
        MEUserInterface::instance()->showMaximized();
    }

    //
    else if (list[0] == "LIST")
    //
    {
        // create a new host and set categories and modules
        MEMainHandler::instance()->initHost(list);
    }

    //
    else if (list[0] == "COLLABORATIVE_STATE")
    //
    {
        MEMainHandler::instance()->showHostState(list);
    }

    //
    else if (list[0] == "INIT")
    //
    {
        MEMainHandler::instance()->initNode(list);
    }

    //
    else if (list[0] == "SYNC")
    //
    {
        /*
      // no exec on change during init
      save_exec = MEMainHandler::instance()->isExecOnChange();
      MEUserInterface::instance()->changeExecButton(false);

      // get node (original)
      m_clonedNode = MENodeListHandler::instance()->getNode(list[6], list[7], list[8]);
      if(m_clonedNode != NULL)
      {

         // init new node
         mname  = list[1];
         number = list[2];
         hname  = list[3];
         x      = list[4].toInt();
         y      = list[5].toInt();
         m_currentNode = MEMainHandler::instance()->nodeHandler->addNode();
         m_currentNode->syncNode(mname, number, hname, x, y, m_clonedNode);
         MEUserInterface::instance()->mirrorList.prepend(m_currentNode);

      }

      // reset mode
      MEUserInterface::instance()->changeExecButton(save_exec);  */
    }

    //
    // message is only used to clear lists if operation SYNC was done
    //
    else if (list[0] == "CLEAR_COPY_LIST")
    {
        //MEMainHandler::instance()->clearCopyList();
    }

    //
    // receive the description for all modules
    //
    else if (list[0] == "DESC")
    {
        MEMainHandler::instance()->setDescriptionOfNode(list);
    }

    //
    // store message content to the application clipboard
    //
    else if (list[0] == "SETCLIPBOARD")
    {
        QByteArray ba(data);
        QMimeData *mimeData = new QMimeData();
        mimeData->setData("covise/clipboard", ba);
        QApplication::clipboard()->setMimeData(mimeData);
    }

    //
    // select all nodes created from a clipboard
    //
    else if (list[0] == "SELECT_CLIPBOARD")
    {
        MEMainHandler::instance()->showClipboardNodes(list);
    }

    //
    // pipeline finished
    //
    else if (list[0] == "FINISHED")
    {
        MEMainHandler::instance()->updateTimer();
        //setCSCW_state(true);
    }

    //
    else if (list[0] == "DEL_SYNC")
    //
    {
        /*   node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
         if(node != NULL)
         {
            MENode *cloneFrom = node->getClone();
            if(cloneFrom)
            {
               cloneFrom->m_syncList.remove(cloneFrom->m_syncList.indexOf(node));
               delete node;
            }
         }*/
    }

    //
    // remove a node
    //
    else if (list[0] == "DEL_REQ")
    {
        int no = 1;
        int iel = 1;
        for (int i = 0; i < no; i++)
        {
            node = MENodeListHandler::instance()->getNode(list[iel], list[iel + 1], list[iel + 2]);
            if (node != NULL)
            {
                if (node->getCategory() == "Renderer")
                    MEUserInterface::instance()->stopRenderer(list[iel], list[iel + 1], list[iel + 2]);
                MEMainHandler::instance()->removeNode(node);
            }
            iel = iel + 3;
        }
    }
    else if (list[0] == "DEL")
    {
        int no = list[1].toInt();
        int iel = 2;
        for (int i = 0; i < no; i++)
        {
            node = MENodeListHandler::instance()->getNode(list[iel], list[iel + 1], list[iel + 2]);
            if (node != NULL)
            {
                if (node->getCategory() == "Renderer")
                    MEUserInterface::instance()->stopRenderer(list[iel], list[iel + 1], list[iel + 2]);
                MEMainHandler::instance()->removeNode(node);
            }
            iel = iel + 3;
        }
    }

    //

    // module has died, deactivate node in canvas
    //
    else if (list[0] == "DIED")
    {
        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);

        // auto delete
        if (node != NULL)
            node->setDead(true);
    }

    //
    // move nodes on canvasArea
    //
    else if (list[0] == "MOV")
    {
        int no = list[1].toInt();
        int iel = 2;
        for (int i = 0; i < no; i++)
        {
            node = MENodeListHandler::instance()->getNode(list[iel], list[iel + 1], list[iel + 2]);
            int x = list[iel + 3].toInt();
            int y = list[iel + 4].toInt();
            if (node != NULL)
                MEMainHandler::instance()->moveNode(node, x, y);
            iel = iel + 5;
        }
    }

    //
    // set sensitive status for parameter
    // enable/disable
    //
    else if (list[0] == "PARSTATE")
    {
        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node != NULL)
        {
            pport = node->getParameterPort(list[4]);
            if (pport != NULL)
            {
                if (list[5] == "FALSE")
                    pport->setSensitive(false);

                else if (list[5] == "TRUE")
                    pport->setSensitive(true);
            }
        }
    }

    //
    // request parameter from module
    //
    else if (list[0] == "PARREQ")
    {
        if (MEMainHandler::instance()->isMaster())
        {
            node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
            if (node != NULL)
            {
                pport = node->getParameterPort(list[4]);
                if (pport != NULL)
                {
                    //pport->moduleParameterRequest();
                }
            }
        }
    }

    //
    else if (list[0] == "PARAM_RESTORE")
    //
    {
        if (MEMainHandler::instance()->isMaster())
        {
            node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
            if (node != NULL)
            {
                pport = node->getParameterPort(list[4]);
                if (pport != NULL)
                {
                    nr = list[6].toInt();
                    pport->modifyParam(list, nr, 7);
                }
            }
        }
    }

    //
    // a parameter was modified new message type
    //
    else if (list[0].contains("PARAM"))
    {
        if (list.size() == 7)
        {
            node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
            if (node != NULL)
            {
                // portname
                pport = node->getParameterPort(list[4]);
                ptype = list[5]; // porttype not used
                if (pport != NULL) // portvalue
                    pport->modifyParameter(list[6]);
            }
        }

        else
        {
            QString text = "Old fashioned PARAM message";
            qCritical() << text << list;
            MEUserInterface::instance()->printMessage(text);

            node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
            if (node != NULL)
            {
                pport = node->getParameterPort(list[4]);
                if (pport != NULL)
                {
                    nr = list[6].toInt();
                    pport->modifyParam(list, nr, 7);
                }
            }
        }
    }

    //
    // change the appearance type of a parameter
    //
    else if (list[0] == "APP_CHANGE")
    {
        // no param type is provided, very suspicious
        // last parameter is new appearance type
        int no = list.count();
        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node != NULL)
        {
            pport = node->getParameterPort(list[4]);
            if (pport != NULL)
            {
                nr = list[no - 1].toInt();
                pport->setAppearance(nr);
                pport->showControlLine();
            }
        }
    }

    //
    // onnect ports
    //
    else if (list[0] == "OBJCONN")
    {
        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node != NULL)
        {
            port = node->getDataPort(list[4]);
            if (port != NULL)
            {
                MENode *node1 = MENodeListHandler::instance()->getNode(list[5], list[6], list[7]);
                if (node1 != NULL)
                {
                    MEDataPort *port1 = node1->getDataPort(list[8]);
                    if (port1 != NULL)
                        MEMainHandler::instance()->addLink(node, port, node1, port1);
                }
            }
        }
    }

    //
    // connect ports (message contains more than one connection)
    // message from controller when restart or coyping the current state
    //
    else if (list[0] == "OBJCONN2")
    {
        int iend = list[1].toInt();
        int it = 2;
        for (int k = 0; k < iend; k++)
        {
            node = MENodeListHandler::instance()->getNode(list[it], list[it + 1], list[it + 2]);
            if (node != NULL)
            {
                port = node->getDataPort(list[it + 3]);
                if (port != NULL)
                {
                    MENode *node1 = MENodeListHandler::instance()->getNode(list[it + 4], list[it + 5], list[it + 6]);
                    if (node1 != NULL)
                    {
                        MEDataPort *port1 = node1->getDataPort(list[it + 7]);
                        if (port1 != NULL)
                            MEMainHandler::instance()->addLink(node, port, node1, port1);
                    }
                }
            }
            it = it + 8;
        }
    }

    //
    // disconnect ports
    //
    else if (list[0] == "DELETE_LINK")
    {
        MENode *node1 = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node1 != NULL)
        {
            MEDataPort *port1 = node1->getDataPort(list[4]);
            if (port1 != NULL)
            {
                MENode *node2 = MENodeListHandler::instance()->getNode(list[5], list[6], list[7]);
                if (node2 != NULL)
                {
                    MEDataPort *port2 = node2->getDataPort(list[8]);
                    if (port2 != NULL)
                        MEMainHandler::instance()->removeLink(node1, port1, node2, port2);
                }
            }
        }
    }

    //
    // remove a parameter to the control panel
    // message comes from UI
    //
    else if (list[0] == "RM_PANEL" || list[0] == "RM_PANEL_F")
    {
        // no param type is provided, very suspicious
        // last parameter is new appearance type
        int no = list.count();
        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node != NULL)
        {
            pport = node->getParameterPort(list[4]);
            if (pport != NULL)
            {
                nr = list[no - 1].toInt();
                pport->setAppearance(nr);
                pport->setMapped(false);
            }
        }
    }

    //
    // add a parameter to the control panel
    // message comes from UI
    //
    else if (list[0] == "ADD_PANEL" || list[0] == "ADD_PANEL_F" || list[0] == "ADD_PANEL_DEFAULT")
    {
        // no param type is provided, very suspicious
        // last parameter is new appearance type
        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node != NULL)
        {
            pport = node->getParameterPort(list[4]);
            if (pport != NULL)
            {
                nr = list[5].toInt();
                if (nr >= 0)
                {
                    pport->setAppearance(nr);
                    pport->setMapped(true);
                }
            }
        }
    }

    //
    // add/remove a parameter to/from the control panel
    // message comes from module
    //
    else if (list[0] == "SHOW" || list[0] == "HIDE")
    {

        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node != NULL)
        {
            pport = node->getParameterPort(list[4]);
            if (pport != NULL)
            {
                if (list[0] == "SHOW")
                    pport->setMapped(true);
                else
                    pport->setMapped(false);
            }
        }
    }

    //
    else if (list[0] == "NEW_ALL" || list[0] == "NEW" || list[0] == "OPEN")
    {
        MEMainHandler::instance()->reset();
    }

    //
    // get default execution mode for add partners/hosts
    //
    else if (list[0] == "HOSTINFO")
    {
        MEMainHandler::instance()->showCSCWDefaults(list);
    }

    //
    // get the password
    // try a new  password again
    //
    else if (list[0] == "ADDHOST" || list[0] == "ADDHOST_FAILED")
    {
        MEMainHandler::instance()->showCSCWParameter(list, MEMainHandler::ADDHOST);
    }

    //
    // get the password
    // try a new  password again
    //
    else if (list[0] == "ADDPARTNER" || list[0] == "ADDPARTNER_FAILED")
    {
        MEMainHandler::instance()->showCSCWParameter(list, MEMainHandler::ADDPARTNER);
    }

    //
    // remove a host
    //
    else if (list[0] == "RMV_LIST")
    {
        MEHost *host = MEHostListHandler::instance()->getHost(list[1], list[2]);
        MEMainHandler::instance()->removeHost(host);
    }

    //
    // set a new module title
    //
    else if (list[0] == "MODULE_TITLE")
    {
        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node != NULL)
        {
            node->setLabel(list[4]);
        }
    }

    //
    // module sends new port description
    //
    else if (list[0] == "PORT_DESC")
    {
        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node != NULL)
        {
            port = node->getDataPort(list[4]);
            if (port != NULL)
                port->setHelpText();
        }
    }

    //
    // module sends new module description
    //
    else if (list[0] == "MODULE_DESC")
    {
        node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
        if (node != NULL)
            node->setDesc(list[4]);
    }

    //
    // open/close slave module parameter window
    //
    else if (list[0] == "OPEN_INFO" || list[0] == "CLOSE_INFO")
    {
        if (!MEMainHandler::instance()->isMaster())
        {
            node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
            if (node != NULL)
                node->bookClick();
        }
    }

    //
    // result from crb for a file search
    //
    else if (list[0] == "FILE_SEARCH_RESULT")
    {
        node = MENodeListHandler::instance()->getNode(list[3], list[4], list[1]);

        // message for main browser ?
        if (node == NULL)
        {
            pport = NULL;
            MEUserInterface::instance()->updateMainBrowser(list);
        }

        // message for a parameter port
        else
        {
            pport = node->getParameterPort(list[5]);
            if (static_cast<MEFileBrowserPort *>(pport)->getBrowser())
                static_cast<MEFileBrowserPort *>(pport)->getBrowser()->updateTree(list);
        }
    }

    //
    // result from crb for a file lookup
    //
    else if (list[0] == "FILE_LOOKUP_RESULT")
    {
        node = MENodeListHandler::instance()->getNode(list[3], list[4], list[1]);

        // message for main browser ?
        if (node == NULL)
        {
            pport = NULL;
            MEUserInterface::instance()->lookupResult(list[6], list[7], list[8]);
        }

        // message for a parameter port
        else
        {
            pport = node->getParameterPort(list[5]);
            if (static_cast<MEFileBrowserPort *>(pport)->getBrowser())
                static_cast<MEFileBrowserPort *>(pport)->getBrowser()->lookupResult(list[6], list[7], list[8]);
        }
    }

    //
    // UI - MIRROR_STATE  was set
    //
    else if (list[0] == "MIRROR_STATE")
    {
        //MEMainHandler::instance()->mirrorStateChanged(list[1].toInt());
    }

    //
    // UI - INEXEC (XXX: ignore for the time being)
    // UI - DC (old data connection message )
    //
    else if (list[0] == "INEXEC" || list[0] == "DC")
    {
    }

    //
    // OpenCover wants to open a tablet userinterface
    //
    else if (list[0] == "WANT_TABLETUI")
    {
        MEUserInterface::instance()->activateTabletUI();
    }

    //
    // show help
    //
    else if (list[0] == "SHOW_HELP")
    {
        QString help = "/index.html";
        if (!list[1].isEmpty())
            help = list[1];
        if (!help.startsWith("/"))
            help.prepend("/");
        MEMainHandler::instance()->onlineCB(help);
    }

    //
    // ignore for now
    //
    else if (list[0] == "GRMSG")
    {
    }

    //
    // message not yet supported
    //
    else
    {
        if (MEMainHandler::instance()->isMaster())
        {
            QString text = "THIS MESSAGE IS NOT SUPPORTED >>> " + list[0] + "   !!!!!!";
            MEUserInterface::instance()->printMessage(text);
        }
    }

    // insert end
}
