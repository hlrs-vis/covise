/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VCCOLLABORATION_PLUGIN_H
#define _VCCOLLABORATION_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 ** Description: ViNCE collaboration plugin.                                 **
 **                                                                          **
 ** Author: Andreas Kopecki	                                             **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>

#include <MsgClient.h>
#include <config/coConfig.h>

#include <QApplication>
#include <QObject>
#include <QMap>
#include <QStringList>
#include <OpenVRUI/coRowMenu.h>

namespace covise
{
class coCheckboxMenuItem;
class coCheckboxGroup;
class coSubMenuItem;
class coProgressBarMenuItem;
}

using namespace covise;
using namespace opencover;

class PLUGINEXPORT VCCollaborationPlugin : public QObject, public coVRPlugin, public coMenuListener
{

    Q_OBJECT

public:
    VCCollaborationPlugin();
    ~VCCollaborationPlugin();

    // this will be called in PreFrame
    virtual void preFrame();
    virtual bool init();

private slots:

    //! Handles the error messages from the client class.
    void errorHandler(const QString &message);

    //! Called when the client connected to the broker.
    void onConnect();

    //! Called when the client disconnected from the broker.
    void onDisconnect();

    //! Called when some other client joined the active session.
    /*! @param senderId The id of the client which joined.
    *  @param userName The name of the client which joined.
    *  @param isMaster Tells if the new client is master or not.
    */
    void onJoin(int senderId, const QString &userName, bool isMaster);
    //! Called when a client leaves the active session.
    /*! @param senderId The id of the client which left. */

    void onLeave(int senderId);
    //! Called when a new session was created or an old one was removed.
    /*! @param sessionName The name of the session.
    *  @param isOnline <code>true</code> if it was created and <code>false</code>
    *  if it was removed.
    */

    void onSessionNotification(const QString &sessionName, bool isOnline);

    //! Called when the own status in the active session changed.
    /*! @param sessionName The name of the active session.
    *  @param status The status of the session.
    */
    void onOwnStatusChange(const QString &sessionName, int status);

    //! Called when data information was received from the broker.
    void onDataInfoReceived(int uploaderId, const QString &dataName, unsigned long dataSize, int userTag);

    //! Called when data is available for download.
    void onDataAvailable(int dataId, unsigned long startPos, MsgDataContainer data);

    //! Called when a initiated data dransfer was accepted.
    void onDataTransferAccepted(int dataId);

    //! Called when an ongoing data transfer was closed.
    void onDataTransferClosed(int dataId);

    //! Called when a initiated data transfer was rejected.
    void onDataTransferRejected(int dataId, int reasonCode, const QString &reasonDesc);

    //! Called when a data transfer has been initiated and waits for being accepted.
    void onDataTransferWaiting(int srcId, int dstId, int dataId, const QString &dataName, unsigned long dataSize, int userTag);

    //! Called when the client is ready to send the next data part.
    void onReadyForNextPart(int dataId);

private:
    void joinSession(const QString &session);
    void leaveSession();

    void menuEvent(coMenuItem *);

    MsgClient *client;

    bool connected;

    QString currentSession;
    int currentDataId;
    QString currentDataName;
    QFile *currentFile;
    int currentFileSize;

    QMap<int, QString> users;

    coConfigInt port;
    coConfigString server;

    QApplication *app;

    coMenu *vccMenu;
    coSubMenuItem *vccMenuEntry;

    coCheckboxGroup *dataCheckboxGroup;

    QMap<QString, coCheckboxMenuItem *> dataCheckboxItems;
    QMap<QString, coProgressBarMenuItem *> dataProgressBarItems;
    QMap<QString, int> dataTransferCompleted;

    QStringList transfersPending;
};
#endif
