/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WS_MESSAGEHANDLER_H
#define WS_MESSAGEHANDLER_H

#include <QObject>
#include <QHash>
#include <QString>

class QTimer;

namespace covise
{

class Message;
class UserInterface;
class covise__Event;
class WSEventManager;
class WSLink;
class WSModule;
class WSParameter;

class WSMessageHandler : public QObject
{
    Q_OBJECT

public:
    WSMessageHandler(int argc, char **argv);
    ~WSMessageHandler();

    static WSMessageHandler *instance();

    bool isStandalone()
    {
        return this->standalone;
    }
    void sendMessage(int, const QString &);
    covise::UserInterface *getUIF()
    {
        return this->userInterface;
    }

    void executeNet();
    void openNet(const QString &filename);

    void deleteModule(const WSModule *module);
    void deleteLink(const WSLink *module);

    void quit();

public slots:

    void dataReceived(int);
    void handleWork();

    void moduleDeletedCB(const QString &moduleID);
    void moduleChangeCB();
    void linkDeletedCB(const QString &linkID);

private:
    static WSMessageHandler *singleton;

    bool standalone;
    covise::UserInterface *userInterface;
    QTimer *periodicTimer;

    void receiveUIMessage(covise::Message *);

    QList<WSModule *> runningModulesWithoutDescription;
};
}
#endif
