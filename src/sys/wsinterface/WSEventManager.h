/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSEVENTMANAGER_H
#define WSEVENTMANAGER_H

#include <QUuid>
#include <QList>
#include <QMap>
#include <QMutex>
//#include <QObject>

namespace covise
{

class covise__Event;
class WSModule;
class WSParameter;

class WSEventManager //: public QObject
{

    //Q_OBJECT

public:
    WSEventManager();
    virtual ~WSEventManager();

    void addEventListener(const QUuid &uuid);
    void removeEventListener(const QUuid &uuid);

    covise::covise__Event *consumeEvent(const QUuid &uuid, int timeout = 10000);
    void postEvent(const covise::covise__Event *event);

    static WSEventManager *instance();

private:
    QMap<QUuid, QList<covise::covise__Event *> > events;
    QMutex lock;
};
}
#endif // WSEVENTMANAGER_H
