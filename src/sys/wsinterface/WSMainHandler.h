/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSMAINHANDLER_H
#define WSMAINHANDLER_H

#include <QObject>
#include <QMap>
#include <QString>
#include <QStringList>
#include <QLinkedList>
#include <QUuid>

#include "WSModule.h"
#include "WSMap.h"

namespace covise
{

class WSEventManager;

class WSMainHandler : public QObject
{

    Q_OBJECT

public:
    static WSMainHandler *instance();
    ~WSMainHandler();

    /**
       * Removes a host from list of hosts
       * @param inHost The map of hosts
       */
    QString removeHost(const QString &inName);

    /**
       * Add the modules available on that host to list
       * @param inName The name of module to be added
       * @param inHost The host the module is available on
       */
    WSModule *addModule(const QString &inName, const QString &inCategory, const QString &inHost);

    void deleteModule(const QString &moduleID);

    /**
       * Get the available modules
       * @return QMap: availableModules
       */
    QMap<QString, QLinkedList<WSModule *> > getAvailableModules()
    {
        return this->availableModules;
    }

    WSModule *getModule(const QString &name, const QString &host) const;

    /**
       * Get the map of loaded modules
       * @return WSMap: map
       */
    WSMap *getMap() const
    {
        return this->map;
    }

    WSMap *newMap()
    {
        return setMap(new WSMap());
    }

    /**
       * Set value of master
       * @param inMaster Boolean: true if, false if not master
       */
    void setMaster(bool inMaster)
    {
        this->master = inMaster;
    }

    /**
       * Get the value of master
       * @return Boolean: true if, false if not master
       */
    bool isMaster() const
    {
        return this->master;
    }

    void setParameter(const QString &moduleID, covise::covise__Parameter *parameter);

    void setParameterFromString(const QString &moduleID, const QString &parameter, const QString &value);

    void executeModule(const QString &moduleID);

    void instantiateModule(const QString &module, const QString &host, int x = -1, int y = -1);

    void link(const QString &fromModule, const QString &fromPort, const QString &toModule, const QString &toPort);
    void unlink(const QString &linkID);

    QList<covise::WSLink *> getLinks() const;

    QUuid addEventListener();
    void removeEventListener(const QString &uuid);

    covise::covise__Event *consumeEvent(const QString &uuid, int timeout = 10000);
    void postEvent(const covise::covise__Event *event);

private:
    WSMainHandler();

    /**
       * Print content of a map
       * @param  inMap The name of the QMap to be printed
       */
    void printList(QMap<QString, QLinkedList<WSModule *> > inMap);

    /**
       * Set the map of loaded modules
       * @param inMap The new value of map
       */
    WSMap *setMap(WSMap *inMap);

    static WSMainHandler *singleton;
    // Contains a map of the host and the modules available on that host
    QMap<QString, QLinkedList<WSModule *> > availableModules;
    // The loaded modules
    WSMap *map;

    WSEventManager *eventManager;

    bool master;
};
}
#endif // WSMAINHANDLER_H
