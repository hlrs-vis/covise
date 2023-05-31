/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 ** Description: ViNCE collaboration plugin.                                 **
 **                                                                          **
 ** Author: Andreas Kopecki	                                             **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coProgressBarMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coSubMenuItem.h>

#include "VCCollaborationPlugin.h"

#include <iostream>

VCCollaborationPlugin::VCCollaborationPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, connected(false)
, currentDataId(0)
, currentFile(0)
, currentFileSize(0)
, port("port", "COVER.Plugin.ViNCECollaboration.Broker")
, server("server", "COVER.Plugin.ViNCECollaboration.Broker")
, app(0)
, vccMenu(0)
, vccMenuEntry(0)
, dataCheckboxGroup(0)
{

    if (!port.hasValidValue())
        port = 6666;
    if (!server.hasValidValue())
        server = "127.0.0.1";

    if (QApplication::instance() == 0)
    {
        int argc = 0;
        this->app = new QApplication(argc, 0);
        this->app->setAttribute(Qt::AA_MacDontSwapCtrlAndMeta);
    }

    std::cerr << "VCCollaborationPlugin::<init> info: connecting to broker at " << qPrintable((QString)server) << ":" << port << std::endl;

    this->client = new MsgClient();

    connect(this->client, SIGNAL(errorOccured(const QString &)), this, SLOT(errorHandler(const QString &)));
    connect(this->client, SIGNAL(connected()), this, SLOT(onConnect()));
    connect(this->client, SIGNAL(disconnected()), this, SLOT(onDisconnect()));
    connect(this->client, SIGNAL(clientJoined(int, const QString &, bool)), this, SLOT(onJoin(int, const QString &, bool)));
    connect(this->client, SIGNAL(clientLeft(int)), this, SLOT(onLeave(int)));
    connect(this->client, SIGNAL(sessionNotification(const QString &, bool)), this, SLOT(onSessionNotification(const QString &, bool)));
    connect(this->client, SIGNAL(dataInfoReceived(int, const QString &, unsigned long, int)), this, SLOT(onDataInfoReceived(int, const QString &, unsigned long, int)));
    connect(this->client, SIGNAL(dataAvailable(int, unsigned long, MsgDataContainer)), this, SLOT(onDataAvailable(int, unsigned long, MsgDataContainer)));
    connect(this->client, SIGNAL(dataTransferAccepted(int)), this, SLOT(onDataTransferAccepted(int)));
    connect(this->client, SIGNAL(dataTransferClosed(int)), this, SLOT(onDataTransferClosed(int)));
    connect(this->client, SIGNAL(dataTransferRejected(int, int, const QString &)), this, SLOT(onDataTransferRejected(int, int, const QString &)));
    connect(this->client, SIGNAL(dataTransferWaiting(int, int, int, const QString &, unsigned long, int)), this, SLOT(onDataTransferWaiting(int, int, int, const QString &, unsigned long, int)));
    connect(this->client, SIGNAL(readyForNextPart(int)), this, SLOT(onReadyForNextPart(int)));
    connect(this->client, SIGNAL(ownStatusChanged(const QString &, int)), this, SLOT(onOwnStatusChange(const QString &, int)));

    this->client->setUserName("OpenCOVER");
    this->client->connectToBroker(server, port);
}

// this is called if the plugin is removed at runtime
VCCollaborationPlugin::~VCCollaborationPlugin()
{
    if (this->app)
        this->app->quit();
    delete this->app;
    this->app = 0;
}

bool VCCollaborationPlugin::init()
{
    this->vccMenu = new coRowMenu("ViNCE Collaboration");
    this->vccMenuEntry = new coSubMenuItem("ViNCE Collaboration");

    this->vccMenuEntry->setMenu(this->vccMenu);

    cover->getMenu()->add(vccMenuEntry);

    this->dataCheckboxGroup = new coCheckboxGroup(true);

    return true;
}

void VCCollaborationPlugin::preFrame()
{

    if (!this->transfersPending.empty() && this->currentDataName == "")
    {
        this->currentDataName = this->transfersPending.takeFirst();
        this->client->requestSessionData(this->currentDataName);
    }

    if (this->app)
        this->app->processEvents();
}

void VCCollaborationPlugin::errorHandler(const QString &message)
{
    std::cerr << "VCCollaborationPlugin::errorHandler err: " << qPrintable(message) << std::endl;
}

void VCCollaborationPlugin::onConnect()
{
    std::cerr << "VCCollaborationPlugin::onConnect info: connection to " << qPrintable((QString)server) << ":" << port << " established" << std::endl;
    this->connected = true;
}

void VCCollaborationPlugin::onDisconnect()
{
    std::cerr << "VCCollaborationPlugin::onConnect info: connection to " << qPrintable((QString)server) << ":" << port << " closed" << std::endl;
    this->connected = false;
}

void VCCollaborationPlugin::onJoin(int senderId, const QString &userName, bool isMaster)
{
    std::cerr << "VCCollaborationPlugin::onJoin info: user " << qPrintable(userName) << " joined the session" << (isMaster ? " as master" : "") << std::endl;
    this->users[senderId] = userName;
}

void VCCollaborationPlugin::onLeave(int senderId)
{
    std::cerr << "VCCollaborationPlugin::onLeave info: user " << qPrintable(users[senderId]) << " left the session" << std::endl;
    this->users.remove(senderId);
}

void VCCollaborationPlugin::onSessionNotification(const QString &sessionName, bool isOnline)
{
    if (isOnline)
    {
        std::cerr << "VCCollaborationPlugin::onSessionNotification info: new session '" << qPrintable(sessionName) << (client->requiresPassword(sessionName) ? "' (password protected)" : "'") << std::endl;
        joinSession(sessionName);
    }
    else
    {
        std::cerr << "VCCollaborationPlugin::onSessionNotification info: session '" << qPrintable(sessionName) << "' closed" << std::endl;
        if (sessionName == currentSession)
            leaveSession();
    }
}

void VCCollaborationPlugin::onDataInfoReceived(int uploaderId, const QString &dataName, unsigned long dataSize, int userTag)
{
    std::cerr << "VCCollaborationPlugin::onDataInfoReceived info: data info " << uploaderId
              << ":" << qPrintable(dataName) << ":" << dataSize << ":" << userTag << std::endl;

    coCheckboxMenuItem *cbItem = new coCheckboxMenuItem(dataName.toStdString(), false, this->dataCheckboxGroup);
    coProgressBarMenuItem *pbItem = new coProgressBarMenuItem(dataName.toStdString());

    cbItem->setMenuListener(this);
    this->vccMenu->add(cbItem);
    this->vccMenu->add(pbItem);
    this->dataCheckboxItems.insert(dataName, cbItem);
    this->dataProgressBarItems.insert(dataName, pbItem);

    this->transfersPending.append(dataName);
}

void VCCollaborationPlugin::onDataAvailable(int dataId, unsigned long startPos, MsgDataContainer data)
{
    //   std::cerr << "VCCollaborationPlugin::onDataAvailable info: new data available " << dataId
    //             << "@" << startPos << std::endl;

    if (this->currentFile == 0)
    {
        std::cerr << "VCCollaborationPlugin::onDataAvailable err: no file open " << std::endl;
        return;
    }

    if (this->currentDataId != dataId)
    {
        std::cerr << "VCCollaborationPlugin::onDataAvailable err: current id " << this->currentDataId << " does not match incoming " << dataId << std::endl;
        return;
    }

    if (startPos == this->currentFile->size())
    {
        this->currentFile->write((char *)data.data(), data.size());

        int completed = 0;

        if (this->currentFileSize == 0)
            std::cerr << "VCCollaborationPlugin::onDataAvailable info: retrieved 0%" << std::endl;
        else
        {

            completed = (int)(100.0f * (float)this->currentFile->size() / (float)this->currentFileSize);

            //         std::cerr << "VCCollaborationPlugin::onDataAvailable info: downloading " << qPrintable(this->currentDataName) << " ("
            //                   << completed << "%)" << std::endl;

            this->dataProgressBarItems[this->currentDataName]->setProgress(completed);
            this->dataTransferCompleted[this->currentDataName] = completed;
        }
    }
    else
    {

        std::cerr << "VCCollaborationPlugin::onDataAvailable err: corrupt data transfer, resume position for data transfer #"
                  << dataId << " invalid" << std::endl;

        client->closeDataTransfer(dataId);
    }
}

void VCCollaborationPlugin::onDataTransferAccepted(int dataId)
{
    std::cerr << "VCCollaborationPlugin::onDataTransferAccepted info: data transfer accepted " << dataId << std::endl;
}

void VCCollaborationPlugin::onDataTransferClosed(int dataId)
{
    std::cerr << "VCCollaborationPlugin::onDataTransferClosed info: data transfer closed " << dataId << std::endl;

    if (this->currentFile == 0)
        return;

    if (this->currentDataId != dataId)
        return;

    this->dataProgressBarItems[this->currentDataName]->setProgress(100);

    this->currentFile->close();
    delete this->currentFile;
    this->currentFile = 0;
    this->currentFileSize = 0;
    this->currentDataId = 0;

    this->currentDataName = "";
}

void VCCollaborationPlugin::onDataTransferRejected(int dataId, int reasonCode, const QString &reasonDesc)
{
    std::cerr << "VCCollaborationPlugin::onDataTransferRejected info: data transfer of "
              << dataId << " rejected because of '" << qPrintable(reasonDesc)
              << "' (" << reasonCode << ")" << std::endl;
}

void VCCollaborationPlugin::onDataTransferWaiting(int srcId, int dstId, int dataId, const QString &dataName, unsigned long dataSize, int userTag)
{
    (void)srcId;
    (void)dataId;
    (void)userTag;

    if ((dstId != this->client->getClientId()) && (dstId != 0))
        return;

    if (this->currentDataId != 0)
    {
        // Abort current transfer and put it back into download queue
        // BUG Doesn't really work, re-requests all files...
        this->client->closeDataTransfer(currentDataId);
        this->currentDataId = 0;
        this->transfersPending.prepend(this->currentDataName);
        this->currentDataName = "";
        if (this->currentFile != 0)
        {
            this->currentFile->close();
            delete this->currentFile;
            this->currentFile = 0;
            this->currentFileSize = 0;
        }
    }

    std::cerr << "VCCollaborationPlugin::onDataTransferWaiting info: data '"
              << qPrintable(dataName) << "' waiting for download (size = " << dataSize << ")"
              << std::endl;

    this->currentDataName = dataName;
    this->currentFile = new QFile("/tmp/" + dataName);
    this->currentFile->open(QIODevice::WriteOnly);
    this->currentFileSize = dataSize;
    this->currentDataId = dataId;

    this->client->acceptDataTransfer(dataId);
}

void VCCollaborationPlugin::onReadyForNextPart(int dataId)
{
    std::cerr << "VCCollaborationPlugin::onReadyForNextPart info: client ready to send next part of " << dataId << std::endl;
}

void VCCollaborationPlugin::joinSession(const QString &session)
{

    if (this->client->requiresPassword(session))
    {
        std::cerr << "VCCollaborationPlugin::joinSession warn: cannot connect to password protected session '" << qPrintable(session) << "'" << std::endl;
        return;
    }

    std::cerr << "VCCollaborationPlugin::joinSession info: joining session '" << qPrintable(session) << "'" << std::endl;
    this->client->joinSession(session, "");
    this->currentSession = session;

    if (this->client->getCurrentSession() != "")
        client->requestSessionData("", true);
}

void VCCollaborationPlugin::leaveSession()
{
    std::cerr << "VCCollaborationPlugin::leaveSession info: leaving session" << std::endl;
    this->client->leaveSession();
    this->currentSession = "";
}

void VCCollaborationPlugin::onOwnStatusChange(const QString &session, int status)
{

    (void)session;

    switch (status)
    {
    case MsgClient::SS_NOSESSION:
    case MsgClient::SS_MASTER:
    case MsgClient::SS_PARTICIPANT:
        break;
    default:
        ;
    };

    if (this->client->getCurrentSession() != "")
        client->requestSessionData("", true);
}

void VCCollaborationPlugin::menuEvent(coMenuItem *item)
{

    if (item->isOfClassName("coCheckboxMenuItem"))
    {

        coCheckboxMenuItem *cbItem = dynamic_cast<coCheckboxMenuItem *>(item);
        const char *cbItemText = cbItem->getLabel()->getString();

        if (cbItem->getState() && this->dataTransferCompleted.contains(cbItemText))
        {
            QString filename = QString("/tmp/%1").arg(cbItemText);
            if (this->dataTransferCompleted[cbItemText] > 99)
            {
                QString extension = coVRFileManager::instance()->findFileExt(filename.toLatin1().data());
                if (extension != "" && coVRFileManager::instance()->getFileHandler(extension.toLatin1().data()))
                {
                    std::cerr << "VCCollaborationPlugin::menuEvent info: trying to load " << qPrintable(filename) << std::endl;
                    coVRFileManager::instance()->loadFile(filename.toLatin1().data());
                }
                else if (extension != "" && coVRFileManager::instance()->findIOHandler(extension.toLatin1().data()))
                {
                    std::cerr << "VCCollaborationPlugin::menuEvent info: trying to load " << qPrintable(filename) << std::endl;
                    coVRFileManager::instance()->loadFile(filename.toLatin1().data());
                }

                {
                    std::cerr << "VCCollaborationPlugin::menuEvent warn: cannot load " << qPrintable(filename) << std::endl;
                }
            }
            else
            {
                std::cerr << "VCCollaborationPlugin::menuEvent warn: cannot load " << qPrintable(filename)
                          << ", still incomplete at " << this->dataTransferCompleted[cbItemText] << "%" << std::endl;
            }
        }
        else if (!cbItem->getState())
        {
            coVRFileManager::instance()->unloadFile();
        }
    }
}

COVERPLUGIN(VCCollaborationPlugin)
