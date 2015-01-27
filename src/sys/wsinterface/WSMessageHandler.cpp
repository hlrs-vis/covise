/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WIN32
#include <pwd.h>
#endif

#include <QSocketNotifier>
#include <QTimer>
#include <QDebug>
#include <QStringList>
#include <QCoreApplication>
#include <QRegExp>

#include <covise/covise_msg.h>
#include <covise/covise_appproc.h>
using namespace covise;

#include "WSMessageHandler.h"
#include "WSMainHandler.h"
#include "WSModule.h"

#include "WSIntVectorParameter.h"
#include "WSIntSliderParameter.h"
#include "WSIntScalarParameter.h"
#include "WSFloatVectorParameter.h"
#include "WSFloatSliderParameter.h"
#include "WSFloatScalarParameter.h"
#include "WSFileBrowserParameter.h"
#include "WSChoiceParameter.h"
#include "WSColormapChoiceParameter.h"
#include "WSBooleanParameter.h"
#include "WSStringParameter.h"
#include "WSEventManager.h"
#include "WSLink.h"
#include "WSColormap.h"
#include "WSPort.h"

#define WSMESSAGEHANDLER_POSTEVENT(EVENT)         \
    {                                             \
        covise::covise__##EVENT##Event e;         \
        WSMainHandler::instance()->postEvent(&e); \
    }
#define WSMESSAGEHANDLER_POSTEVENT_1(EVENT, param1) \
    {                                               \
        covise::covise__##EVENT##Event e(param1);   \
        WSMainHandler::instance()->postEvent(&e);   \
    }

covise::WSMessageHandler *covise::WSMessageHandler::singleton = 0;

void remove_socket(int)
{
    qWarning() << "X Socket Connection was removed from WSUI\n";
}

/*!
    \class WSMessageHandler
    \brief Handler for sending and receiving messages from controller, crb, modules & other UI
*/

covise::WSMessageHandler::WSMessageHandler(int argc, char **argv)
    : QObject()
    , periodicTimer(0)
{

    WSMessageHandler::singleton = this;

    // check modus
    if ((argc < 7) || (argc > 8))
    {
        qCritical() << "WS Interface with inappropriate arguments called";
        qCritical() << "No. of arguments is " << argc << endl;
        for (int i = 0; i < argc; i++)
            qCritical() << argv[i];

        standalone = true;
        userInterface = 0;
    }
    else
    {
        this->standalone = false;
        this->userInterface = new UserInterface((char *)"AppModule", argc, argv);

        // make socket
        int TCP_Socket = this->userInterface->get_socket_id(remove_socket);
        QSocketNotifier *sn = new QSocketNotifier(TCP_Socket, QSocketNotifier::Read);

// weil unter windows manchmal Messages verloren gehen
// der SocketNotifier wird nicht oft genug aufgerufen)
#if defined(_WIN32)
        this->periodicTimer = new QTimer;
        QObject::connect(this->periodicTimer, SIGNAL(timeout()), this, SLOT(handleWork()));
        this->periodicTimer->start(1000);
#endif

        QObject::connect(sn, SIGNAL(activated(int)), this, SLOT(dataReceived(int)));
    }

#ifndef YAC
    if (!isStandalone())
    {
        // send dummy message to tell the controller that it is safe now to send messages
        dataReceived(1);

        // tell crb if we are ready for an embedded ViNCE renderer
        sendMessage(COVISE_MESSAGE_MSG_OK, "");
    }
#endif
}

covise::WSMessageHandler *covise::WSMessageHandler::instance()
{
    return WSMessageHandler::singleton;
}

covise::WSMessageHandler::~WSMessageHandler()
{
    if (this->periodicTimer)
    {
        this->periodicTimer->stop();
        delete this->periodicTimer;
    }
}

void covise::WSMessageHandler::executeNet()
{
    sendMessage(COVISE_MESSAGE_UI, "EXEC\n");
}

void covise::WSMessageHandler::openNet(const QString &filename)
{
    sendMessage(COVISE_MESSAGE_UI, "NEW\n");
    sendMessage(COVISE_MESSAGE_UI, QString("OPEN\n%1").arg(filename));
    sendMessage(COVISE_MESSAGE_UI, QString("UPDATE_LOADED_MAPNAME\n%1").arg(filename));
}

void covise::WSMessageHandler::moduleDeletedCB(const QString &moduleID)
{
    covise::covise__ModuleDelEvent e(moduleID.toStdString());
    WSMainHandler::instance()->postEvent(&e);
}

void covise::WSMessageHandler::moduleChangeCB()
{
    covise::covise__ModuleChangeEvent e(qobject_cast<covise::WSModule *>(sender())->getSerialisable());
    WSMainHandler::instance()->postEvent(&e);
}

void covise::WSMessageHandler::linkDeletedCB(const QString &linkID)
{
    covise::covise__LinkDelEvent e(linkID.toStdString());
    WSMainHandler::instance()->postEvent(&e);
}

void covise::WSMessageHandler::deleteModule(const covise::WSModule *module)
{
    QStringList buffer;
    buffer << "DEL"
           << "1" << module->getName() << module->getInstance() << module->getHost();
    sendMessage(covise::COVISE_MESSAGE_UI, buffer.join("\n"));
}

void covise::WSMessageHandler::deleteLink(const covise::WSLink *link)
{
    const covise::WSModule *from = link->from()->getModule();
    const covise::WSModule *to = link->to()->getModule();
    QStringList buffer;
    buffer << "DELETE_LINK"
           << from->getName() << from->getInstance() << from->getHost() << link->from()->getName()
           << to->getName() << to->getInstance() << to->getHost() << link->to()->getName();

    std::cerr << "WSMessageHandler::deleteLink info: " << qPrintable(buffer.join("\n")) << std::endl;
    sendMessage(covise::COVISE_MESSAGE_UI, buffer.join("\n"));
}

void covise::WSMessageHandler::quit()
{
    sendMessage(COVISE_MESSAGE_QUIT, "");
}

//!
//! send an UI message
//!
void covise::WSMessageHandler::sendMessage(int type, const QString &text)
{
    Message *msg = new Message;
    msg->type = (covise_msg_type)type;
    QByteArray ba = text.toLatin1();
    msg->data = ba.data();
    msg->length = (int)strlen(msg->data) + 1;

#ifdef DEBUG
//   qDebug() << "Message send _____________________________ ";
//   qDebug() << msg->data;
//   qDebug() << "__________________________________________ ";
#endif

    if (this->userInterface)
        this->userInterface->send_ctl_msg(msg);
    delete msg;
}

//!
//! receive data on TCP/IP socket
//!
void covise::WSMessageHandler::dataReceived(int)
{
    QString buffer, ptype;

    if (WSMainHandler::instance())
    {
        while (Message *msg = this->userInterface->check_for_msg())
        {
            if (msg->type == COVISE_MESSAGE_SOCKET_CLOSED)
            {
                qDebug() << "WSMessageHandler::dataReceived err: socket closed, exiting";
                exit(1);
            }
            // empty message
            else if (msg->length == 0)
                qDebug() << "WSMessageHandler::dataReceived err: empty message";

            else
            {
                QStringList list = QString(msg->data).split("\n", QString::SkipEmptyParts);

#if 0
            qDebug() << "Message received _________________________";
            qDebug() << msg->type;
            qDebug() << msg->data;
            qDebug() << "__________________________________________";
#endif

                //int nitem = list.count();

                switch (msg->type)
                {
                //               case COVISE_MESSAGE_CRB_EXEC:
                //               {
                //               }
                //               break;

                //
                // QUIT
                //
                case COVISE_MESSAGE_QUIT:
                    WSMESSAGEHANDLER_POSTEVENT(Quit);
                    std::cerr << "WSMessageHandler::dataReceived fixme: have to wait some time before exit" << std::endl;
                    exit(0);
                    break;

                //
                // PARINFO (module send updates for parameters)
                //
                case COVISE_MESSAGE_PARINFO:
                {
                    WSModule *module = WSMainHandler::instance()->getMap()->getModule(list[0], list[1], list[2]);
                    if (module != 0)
                    {
                        WSParameter *parameter = module->getParameter(list[3]);
                        if (parameter != 0)
                        {
                            WSTools::setParameterFromString(parameter, list[5]);
                            covise::covise__ParameterChangeEvent e(module->getID().toStdString(), parameter->getSerialisable());
                            WSMainHandler::instance()->postEvent(&e);
                        }
                    }
                }
                break;

                //
                // START(module)
                //
                case COVISE_MESSAGE_START:
                    WSMESSAGEHANDLER_POSTEVENT_1(ModuleExecuteStart, (list[0] + "_" + list[1] + "_" + list[2]).toStdString());
                    break;

                //
                // FINISHED (module)
                //
                case COVISE_MESSAGE_FINISHED:
                    WSMESSAGEHANDLER_POSTEVENT_1(ModuleExecuteFinish, (list[0] + "_" + list[1] + "_" + list[2]).toStdString());
                    break;

                //
                // ERROR
                //
                //               case COVISE_MESSAGE_COVISE_ERROR:
                //                   if(WSMainHandler::instance()->cfg_ErrorHandling)
                //                   {
                //                      // from module
                //                      if(nitem > 1)
                //                      {
                //                         buffer = list[0];
                //                         buffer.append("_");
                //                         buffer.append(list[1]);
                //                         buffer.append("@");
                //                         buffer.append(list[2]);
                //                         QMessageBox::critical(0, buffer, list[3]);
                //                      }

                //                      // from controller
                //                      else
                //                      {
                //                         buffer = "Controller: ";
                //                         buffer.append(list[0]);
                //                         QMessageBox::critical(0, "Controller" , list[0]);
                //                      }
                //                   }

                //                   else
                //                   {

                //                      // from module
                //                      if(nitem > 1)
                //                      {
                //                         buffer = list[0];
                //                         buffer.append("_");
                //                         buffer.append(list[1]);
                //                         buffer.append("@");
                //                         buffer.append(list[2]);
                //                         buffer.append(": ");
                //                         buffer.append(list[3]);
                //                      }

                //                      // from controller
                //                      else
                //                      {
                //                         buffer = "Controller: ";
                //                         buffer.append(list[0]);
                //                      }
                //                      MEUserInterface::instance()->writeInfoMessage(buffer, Qt::red);
                //                   }
                //                  break;

                //
                // WARNING  (message from module)
                //
                //               case COVISE_MESSAGE_WARNING:
                // from module
                //                   if(nitem > 1)
                //                   {
                //                      buffer = list[0];
                //                      buffer.append("_");
                //                      buffer.append(list[1]);
                //                      buffer.append("@");
                //                      buffer.append(list[2]);
                //                      buffer.append(": ");
                //                      buffer.append(list[3]);
                //                   }

                //                   // from controller
                //                   else
                //                   {
                //                      buffer = "Controller: ";
                //                      buffer.append(list[0]);
                //                   }
                //                  MEUserInterface::instance()->writeInfoMessage(buffer, Qt::blue);
                //                  break;

                //
                // INFO (message from module)
                //
                //               case COVISE_MESSAGE_INFO:
                // chat message from other host
                //                   if(list[0] ==   "CHAT")
                //                   {
                //                      QString tmp = list[2].section('>', -1, -1);
                //                      buffer = list[1];
                //                      buffer.append(" said: ");
                //                      buffer.append(tmp);
                //                      MEUserInterface::instance()->textForChatLine(buffer);
                //                   }

                //                   // from module
                //                   else if(nitem > 1)
                //                   {
                //                      buffer = list[0];
                //                      buffer.append("_");
                //                      buffer.append(list[1]);
                //                      buffer.append("@");
                //                      buffer.append(list[2]);
                //                      buffer.append(": ");
                //                      buffer.append(list[3]);
                //                      MEUserInterface::instance()->writeInfoMessage(buffer, Qt::darkGreen);
                //                   }

                //                   // from controller
                //                   else
                //                   {
                //                      buffer = "Controller: ";
                //                      buffer.append(list[0]);
                //                      QMessageBox::critical(0, "Controller" , list[0]);
                //                   }

                //                  break;

                case COVISE_MESSAGE_UI:
                    receiveUIMessage(msg);
                    break;

                case COVISE_MESSAGE_LAST_DUMMY_MESSAGE:
                {
                    Message *msg2 = new Message();
                    msg2->type = COVISE_MESSAGE_LAST_DUMMY_MESSAGE;
                    msg2->data = (char *)" ";
                    msg2->length = 2;
                    this->userInterface->send_ctl_msg(msg2);
                    delete[] msg2 -> data;
                    delete msg2;
                    break;
                }

                default:
                    ;

#ifdef DEBUG
                    qDebug() << "Unsupported message received _____________";
                    qDebug() << msg->type;
                    qDebug() << msg->data;
                    qDebug() << "__________________________________________";
#endif
                    break;
                }
            }
            msg->delete_data();
            this->userInterface->delete_msg(msg);
        }
    }
}

//!
//! parse an UI message
//!
void covise::WSMessageHandler::receiveUIMessage(Message *msg)
{

    char *data = msg->data;

    QStringList list = QString(data).split("\n");
    //qDebug() << " *** QStringList in receiveUIMessage: " << list;
    //qDebug(qPrintable(list.join(" ")));

    if (list[0] == "MASTER")
    {
        WSMainHandler::instance()->setMaster(true);
    }
    else if (list[0] == "SLAVE")
    {
        WSMainHandler::instance()->setMaster(false);
    }
    else if (list[0] == "UPDATE_LOADED_MAPNAME")
    {
        WSMainHandler::instance()->getMap()->setMapName(list[1]);
    }
    else if (list[0] == "START_READING")
    {
        WSMainHandler::instance()->getMap()->setMapName(list[1]);
        WSMESSAGEHANDLER_POSTEVENT_1(OpenNet, list[1].toStdString());
    }
    else if (list[0] == "END_READING")
    {
        WSMESSAGEHANDLER_POSTEVENT_1(OpenNetDone, WSMainHandler::instance()->getMap()->getMapName().toStdString());
    }

    //
    else if (list[0] == "ICONIFY")
    { // Not needed
    }
    else if (list[0] == "MAXIMIZE")
    { // Not needed
    }
    else if (list[0] == "LIST")
    {
        QStringList::iterator listIterator = list.begin();
        QString host = *(++listIterator);
        QString user = *(++listIterator);
        int numberOfModules = (++listIterator)->toInt();

        //std::cerr << "WSMessageHandler::receiveUIMessage info: got list of " << numberOfModules << " modules for host " << qPrintable(host) << std::endl;

        for (int ctr = 0; ctr < numberOfModules; ++ctr)
        {
            QString name = *(++listIterator);
            QString category = *(++listIterator);
            WSMainHandler::instance()->addModule(name, category, host);
        }
    }
    else if (list[0] == "INIT")
    {
        QStringList::iterator listIterator = list.begin();
        QString name = *(++listIterator);
        QString instance = *(++listIterator);
        QString host = *(++listIterator);
        int positionX = (++listIterator)->toInt();
        int positionY = (++listIterator)->toInt();

        WSModule *inModule = WSMainHandler::instance()->getModule(name, host);
        if (inModule == 0)
        {
            std::cerr << "WSMessageHandler::receiveUIMessage err: cannot instantiate " << qPrintable(name) << " on " << qPrintable(host) << std::endl;
            return;
        }

        WSModule *module = WSMainHandler::instance()->getMap()->addModule(inModule, instance, host);
        module->setPosition(positionX, positionY);
        this->runningModulesWithoutDescription.append(module);
    }
    //else if(list[0] == "SYNC")
    //{
    // no exec on change during init
    //save_exec = WSMainHandler::instance()->isExecOnChange();
    //MEUserInterface::instance()->changeExecButton(false);

    //WSModule * module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
    //   WSModule *module2 = WSMainHandler::instance()->getMap()->getModule(list[6], list[7], list[8]);
    //   if(module2 != NULL)
    //   {
    //      qDebug() << " Message: SYNC (not implemented yet).";
    //m_currentNode = WSMainHandler::instance()->nodeHandler->addNode();
    //m_currentNode->syncNode(list[1], list[2], list[3], list[4].toInt(), list[5].toInt(), module2);
    //MEUserInterface::instance()->mirrorList.prepend(m_currentNode);
    //   }
    // reset mode
    //MEUserInterface::instance()->changeExecButton(save_exec);
    //}

    //
    // message is only used to clear lists if operation SYNC was done
    //
    //else if(list[0] == "CLEAR_COPY_LIST")
    //{
    //WSMainHandler::instance()->clearCopyList();
    //}

    //
    // receive the description for all modules
    //
    else if (list[0] == "DESC")
    {

        QStringList::iterator listIterator = list.begin();
        QString name = *(++listIterator);
        QString category = *(++listIterator);
        QString host = *(++listIterator);

        for (QList<WSModule *>::iterator moduleIterator = this->runningModulesWithoutDescription.begin();
             moduleIterator != this->runningModulesWithoutDescription.end();)
        {
            WSModule *module = *moduleIterator;
            if (module->getName() == name && module->getHost() == host)
            {
                //std::cerr << "WSMessageHandler::receiveUIMessage info: adding description for " << qPrintable(name) << std::endl;

                module->setCategory(category);
                module->setDescription(*(++listIterator));
                int numberOfInputPorts = (++listIterator)->toInt();
                int numberOfOutputPorts = (++listIterator)->toInt();
                int numberOfParameters = (++listIterator)->toInt();
                int numberOfSomethingYetUnknown = (++listIterator)->toInt();
                (void)numberOfSomethingYetUnknown;

                for (int ctr = 0; ctr < numberOfInputPorts; ++ctr)
                {
                    QString name = *(++listIterator);
                    QStringList typeList = (++listIterator)->split('|');
                    QString unknown = *(++listIterator); // TODO unknown
                    QString type = *(++listIterator);
                    WSPort::PortType pType = WSPort::Default;
                    if (type == "opt")
                        pType = WSPort::Optional;
                    if (type == "dep")
                        pType = WSPort::Dependent;
                    module->addInputPort(name, typeList, pType);
                }

                for (int ctr = 0; ctr < numberOfOutputPorts; ++ctr)
                {
                    QString name = *(++listIterator);
                    QStringList typeList = (++listIterator)->split('|');
                    QString unknown = *(++listIterator); // TODO unknown
                    QString type = *(++listIterator);
                    WSPort::PortType pType = WSPort::Default;
                    if (type == "opt")
                        pType = WSPort::Optional;
                    if (type == "dep")
                        pType = WSPort::Dependent;
                    module->addOutputPort(name, typeList, pType);
                }

                // File browser filters are a separate parameter, thus save the file browsers for later use
                QMap<QString, WSFileBrowserParameter *> fileBrowsers;

                for (int ctr = 0; ctr < numberOfParameters; ++ctr)
                {
                    QString name = *(++listIterator);
                    QString type = *(++listIterator);
                    QString description = *(++listIterator);
                    QString value = *(++listIterator);
                    QString immediate = *(++listIterator);
                    (void)immediate; // All parameters are immediate

                    if (type == "Browser")
                        type = "FileBrowser";
                    if (type == "BrowserFilter")
                    {
                        QRegExp expression("(.*)___filter");
                        if (!expression.exactMatch(name))
                        {
                            std::cerr << "WSMessageHandler::receiveUIMessage err: invalid BrowserFilter name " << qPrintable(name) << std::endl;
                            continue;
                        }
                        WSFileBrowserParameter *browser = fileBrowsers[expression.cap(1)];
                        if (browser == 0)
                        {
                            std::cerr << "WSMessageHandler::receiveUIMessage err: Browser not found for BrowserFilter " << qPrintable(name) << std::endl;
                            continue;
                        }
#ifdef DEBUG
                        std::cerr << "WSMessageHandler::receiveUIMessage fixme: BrowserFilter not implemented" << std::endl;
#endif
                        continue;
                    }

                    WSParameter *parameter = module->addParameter(name, type, description);

                    if (parameter != 0)
                    {
                        if (parameter->getType() == "FileBrowser")
                        {
                            fileBrowsers.insert(name, dynamic_cast<WSFileBrowserParameter *>(parameter));
                        }

                        WSTools::setParameterFromString(parameter, value);
                    }
                }

                // Inform the handler when the module is destroyed
                connect(module, SIGNAL(deleted(const QString &)), this, SLOT(moduleDeletedCB(const QString &)));
                connect(module, SIGNAL(changed()), this, SLOT(moduleChangeCB()));

                covise::covise__ModuleAddEvent e(module->getSerialisable());
                WSMainHandler::instance()->postEvent(&e);

                moduleIterator = this->runningModulesWithoutDescription.erase(moduleIterator);
            }
            else
            {
                ++moduleIterator;
            }
        }
    }

    //
    // store message content to the application clipboard
    //
    //else if(list[0] == "SETCLIPBOARD")
    //{
    //       QByteArray ba(data);
    //       QMimeData *mimeData = new QMimeData();
    //       mimeData->setData("covise/clipboard", ba);
    //       QApplication::clipboard()->setMimeData(mimeData);
    //}

    //
    // select all nodes created from a clipboard
    //
    //else if(list[0] == "SELECT_CLIPBOARD")
    //{
    //      WSMainHandler::instance()->showClipboardNodes(list);
    //}

    //
    // request of a slave user interface for master
    //
    //else if(list[0] == "MASTERREQ" /* && WSMainHandler::instance()->isMaster()*/)
    //{
    //   qDebug() << " Message: MASTERREQ (not implemented yet).";
    //       QString text = "Grant m_masterUI status to ";
    //       text.append(list[1]);
    //       text.append("@");
    //       text.append(list[2]);
    //       text.append("?");

    //       QString data;
    //       switch( QMessageBox::question(MEUserInterface::instance(), "COVISE: QtMEUserInterface::instance()",text,  "Grant", "Deny", "",  0, 1 ))
    //       {

    //          case 0:
    //             data = "STATUS\nSLAVE\n";
    //             WSMainHandler::instance()->setMaster(false);
    //             break;

    //          case 1:
    //             data = "STATUS\nMASTER\n";
    //             break;
    //       }

    //       sendMessage(COVISE_MESSAGE_UI, data);
    //}

    //
    // pipeline finished
    //
    else if (list[0] == "FINISHED")
    {
        WSMESSAGEHANDLER_POSTEVENT(ExecuteFinish);
    }
    //else if(list[0] == "DEL_SYNC")
    //{
    //   WSModule * module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
    //   if(module != NULL)
    //   {
    //      qDebug() << " Message: DEL_SYNC (not implemented yet).";
    /*         WSModule *moduleClone = module->getClone();
         if(moduleClone)
         {
            moduleClone->m_syncList.remove(moduleClone->m_syncList.indexOf(module));
            delete module;
         }*/
    //   }
    //}

    //
    // remove a node
    //

    else if (list[0] == "DEL" || list[0] == "DEL_REQ")
    {
        QString moduleID = WSMainHandler::instance()->getMap()->makeKeyName(list[2], list[3], list[4]);
        if (WSMainHandler::instance()->getMap()->getModule(moduleID) != 0)
            WSMainHandler::instance()->getMap()->removeModule(moduleID);
    }

    // module has died, deactivate node in canvas
    //
    else if (list[0] == "DIED")
    {
        WSModule *module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
        if (module != 0)
            module->setDead(true);
    }

    //
    // move node on canvasArea
    //
    //else if(list[0] == "MOV")
    //{
    //       if(WSMainHandler::instance()->isMaster())
    //       {
    //          node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
    //          if(node != NULL)
    //          {
    //             int x = list[4].toInt();
    //             int y = list[5].toInt();
    //             WSMainHandler::instance()->moveNode(node, x, y);
    //          }
    //       }
    //}

    //
    // set sensitive status for parameter
    // enable/disable
    //
    else if (list[0] == "PARSTATE")
    {
        WSModule *module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
        if (module != 0)
        {
            WSParameter *parameter = module->getParameter(list[4]);
            if (parameter != 0)
            {
                if (list[5].toLower() == "false")
                    parameter->setEnabled(false);
                else if (list[5].toLower() == "true")
                    parameter->setEnabled(true);

                covise::covise__ParameterChangeEvent e(module->getID().toStdString(), parameter->getSerialisable());
                WSMainHandler::instance()->postEvent(&e);
            }
        }
    }

    //
    // request parameter from module
    //
    //else if(list[0] == "PARREQ")
    //{
    //   qDebug() << " Message: PARREQ (not implemented yet).";
    //       if(WSMainHandler::instance()->isMaster())
    //       {
    //          node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
    //          if(node != NULL)
    //          {
    //             pport = node->getParameterPort(list[4]);
    //             if(pport != NULL)
    //             {
    //                //pport->moduleParameterRequest();
    //             }
    //          }
    //       }
    //   }
    //else if(list[0] == "PARAM_RESTORE" )
    //{
    //   qDebug() << " Message: PARAM_RESTORE (not implemented yet).";
    //       if(WSMainHandler::instance()->isMaster())
    //       {
    //          node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
    //          if(node != NULL)
    //          {
    //             pport = node->getParameterPort(list[4]);
    //             if(pport != NULL)
    //             {
    //                nr    = list[6].toInt();
    //                pport->modifyParam(list, nr, 7);
    //             }
    //          }
    //       }
    //}

    //
    // a parameter was modified
    //
    else if (list[0].contains("PARAM"))
    {
        if (list.size() == 7)
        {
            WSModule *module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
            if (module != 0)
            {
                WSParameter *parameter = module->getParameter(list[4]);
                if (parameter != 0)
                {
                    bool changed = WSTools::setParameterFromString(parameter, list[6]);
                    if (changed)
                    {
                        covise::covise__ParameterChangeEvent e(module->getID().toStdString(), parameter->getSerialisable());
                        WSMainHandler::instance()->postEvent(&e);
                    }
                }
            }
        }
    }

    //
    // change the appearance type of a parameter
    //
    //else if(list[0] == "APP_CHANGE")
    //{
    //   qDebug() << " Message: APP_CHANGE (not implemented yet).";
    //       // no param type is provided, very suspicious
    //       // last parameter is new appearance type
    //       int no = list.count();
    //       node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
    //       if(node != NULL)
    //       {
    //          pport = node->getParameterPort(list[4]);
    //          if(pport != NULL )
    //          {
    //             nr = list[no-1].toInt();
    //             pport->setAppearance(nr);
    //             pport->showControlLine();
    //          }
    //       }
    //}

    //
    // connect ports
    //
    else if (list[0] == "OBJCONN")
    {
        QString fromModule = WSMainHandler::instance()->getMap()->makeKeyName(list[1], list[2], list[3]);
        QString toModule = WSMainHandler::instance()->getMap()->makeKeyName(list[5], list[6], list[7]);
        covise::WSLink *link = WSMainHandler::instance()->getMap()->link(fromModule, list[4], toModule, list[8]);
        if (link != 0)
        {
            connect(link, SIGNAL(deleted(QString)), this, SLOT(linkDeletedCB(QString)));
            covise::covise__LinkAddEvent e(link->getSerialisable());
            WSMainHandler::instance()->postEvent(&e);
        }
    }

    //
    // connect ports (message contains more than one connection)
    // message from controller when restart or coyping the current state
    //
    //else if(list[0] == "OBJCONN2")
    //{
    //   qDebug() << " Message: OBJCONN2 (not implemented yet).";
    //       int iend = list[1].toInt();
    //       int it = 2;
    //       for(int k=0; k<iend; k++)
    //       {
    //          node = MENodeListHandler::instance()->getNode(list[it], list[it+1], list[it+2]);
    //          if(node != NULL)
    //          {
    //             port = node->getDataPort(list[it+3]);
    //             if(port != NULL)
    //             {
    //                MENode *node1 = MENodeListHandler::instance()->getNode(list[it+4], list[it+5], list[it+6]);
    //                if(node1 != NULL)
    //                {
    //                   MEDataPort *port1 = node1->getDataPort(list[it+7]);
    //                   if(port1 != NULL)
    //                      WSMainHandler::instance()->addLink(node, port, node1, port1);
    //                }
    //             }
    //          }
    //          it=it+8;
    //       }
    //}

    //
    // disconnect ports
    //
    else if (list[0] == "DELETE_LINK")
    {
        QString fromModule = WSMainHandler::instance()->getMap()->makeKeyName(list[1], list[2], list[3]);
        QString toModule = WSMainHandler::instance()->getMap()->makeKeyName(list[5], list[6], list[7]);

        covise::WSLink *link = WSMainHandler::instance()->getMap()->getLink(covise::WSLink::makeID(fromModule, list[4], toModule, list[8]));
        if (link != 0)
        {
            covise::covise__LinkDelEvent e(link->getLinkID().toStdString());
            WSMainHandler::instance()->postEvent(&e);
            WSMainHandler::instance()->getMap()->unlink(link);
        }
    }

    //
    // remove a parameter to the control panel
    // message comes from UI

    else if (list[0] == "RM_PANEL" || list[0] == "RM_PANEL_F" ||
             //
             // add/remove a parameter to/from the control panel
             // message comes from module
             list[0] == "HIDE")
    {
        WSModule *module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
        if (module != 0)
        {
            WSParameter *parameter = module->getParameter(list[4]);
            if (parameter != 0)
            {
                parameter->setMapped(false);
                covise::covise__ParameterChangeEvent e(module->getID().toStdString(), parameter->getSerialisable());
                WSMainHandler::instance()->postEvent(&e);
            }
        }
    }

    //
    // add a parameter to the control panel
    // message comes from UI
    //
    else if (list[0] == "ADD_PANEL" || list[0] == "ADD_PANEL_F" || list[0] == "ADD_PANEL_DEFAULT" ||
             //
             // add/remove a parameter to/from the control panel
             // message comes from module
             list[0] == "SHOW")
    {
        WSModule *module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
        if (module != 0)
        {
            WSParameter *parameter = module->getParameter(list[4]);
            if (parameter != 0)
            {
                if (list[0] == "SHOW" || list[5].toInt() >= 0)
                {
                    parameter->setMapped(true);
                    covise::covise__ParameterChangeEvent e(module->getID().toStdString(), parameter->getSerialisable());
                    WSMainHandler::instance()->postEvent(&e);
                }
            }
        }
    }

    //
    else if (list[0] == "NEW_ALL" || list[0] == "NEW")
    {
        WSMainHandler::instance()->newMap();
    }

    //
    // get default execution mode for add partners/hosts
    //
    //else if(list[0] == "HOSTINFO")
    //{
    //      WSMainHandler::instance()->showCSCWDefaults(list);
    //}

    //
    // get the password
    // try a new  password again
    //
    //else if(list[0] == "ADDHOST" ||
    //   list[0] == "ADDHOST_FAILED")
    //{
    //      WSMainHandler::instance()->showCSCWParameter(list, WSMainHandler::ADDHOST);
    //}

    //
    // get the password
    // try a new  password again
    //
    //else if(list[0] == "ADDPARTNER" ||
    //   list[0] == "ADDPARTNER_FAILED")
    //{
    //      WSMainHandler::instance()->showCSCWParameter(list, WSMainHandler::ADDPARTNER);
    //}

    //
    // remove a host
    //
    else if (list[0] == "RMV_LIST")
    {
        WSMainHandler::instance()->removeHost(list[1]);
    }

    //
    // set a new module title
    //
    else if (list[0] == "MODULE_TITLE")
    {
        WSModule *module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
        if (module != 0)
        {
            module->setTitle(list[4]);
        }
    }

    //
    // module sends new port description
    //
    // else if(list[0] == "PORT_DESC")
    // {
    //    WSModule * module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
    //    if(module != NULL)
    //    {
    //       qDebug() << " Message: PORT_DESC (not implemented yet).";
    //    }
    // }

    //
    // module sends new module description
    //
    else if (list[0] == "MODULE_DESC")
    {
        WSModule *module = WSMainHandler::instance()->getMap()->getModule(list[1], list[2], list[3]);
        if (module != 0)
            module->setDescription(list[4]);
    }

    //
    // open/close slave module parameter window
    //
    //else if(list[0] == "OPEN_INFO" || list[0] == "CLOSE_INFO")
    //{
    //       if(!WSMainHandler::instance()->isMaster())
    //       {
    //          node = MENodeListHandler::instance()->getNode(list[1], list[2], list[3]);
    //          if(node != NULL)
    //             node->bookClick();
    //       }
    //}

    //
    // result from crb for a file search
    //
    //else if(list[0] == "FILE_SEARCH_RESULT")
    //{
    //       node = MENodeListHandler::instance()->getNode(list[3], list[4], list[1]);

    //       // message for main browser ?
    //       if(node == NULL)
    //       {
    //          pport = NULL;
    //          MEUserInterface::instance()->updateMainBrowser(list);
    //       }

    //       // message for a parameter port
    //       else
    //       {
    //          pport = node->getParameterPort(list[5]);
    //          if(static_cast<MEFileBrowserPort *>(pport)->getBrowser() )
    //             static_cast<MEFileBrowserPort *>(pport)->getBrowser()->updateTree(list);
    //       }
    //}

    //
    // result from crb for a file lookup
    //
    //else if(list[0] == "FILE_LOOKUP_RESULT")
    //{
    //       node = MENodeListHandler::instance()->getNode(list[3], list[4], list[1]);

    //       // message for main browser ?
    //       if(node == NULL)
    //       {
    //          pport = NULL;
    //          MEUserInterface::instance()->lookupResult(list[6], list[7], list[8]);
    //       }

    //       // message for a parameter port
    //       else
    //       {
    //          pport = node->getParameterPort(list[5]);
    //          if(static_cast<MEFileBrowserPort *>(pport)->getBrowser() )
    //             static_cast<MEFileBrowserPort *>(pport)->getBrowser()->lookupResult(list[6], list[7], list[8]);
    //       }
    //}

    //
    // UI - MIRROR_STATE  was set
    //
    //else if(list[0] == "MIRROR_STATE")
    //{
    //WSMainHandler::instance()->mirrorStateChanged(list[1].toInt());
    //}

    //
    // UI - INEXEC
    //
    else if (list[0] == "INEXEC")
    {
        WSMESSAGEHANDLER_POSTEVENT(ExecuteStart);
    }

    //
    // UI - DC (old data connection message )
    //
    //if (list[0] == "DC")
    //{
    //}

    //
    // OpenCover wants to open a tablet userinterface
    //
    //else if(list[0] == "WANT_TABLETUI")
    //{
    //      MEUserInterface::instance()->activateTabletUI();
    //}

    //
    // message not yet supported
    //
    else
    {
#ifdef DEBUG
        if (1 /*WSMainHandler::instance()->isMaster()*/)
        {

            qDebug() << "Unsupported message received _____________";
            qDebug() << msg->type;
            qDebug() << msg->data;
            qDebug() << "__________________________________________";
        }
#endif
    }
    // insert end
}

void covise::WSMessageHandler::handleWork()
{
    dataReceived(1);
}

/// EOF
