/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSEventManager.h"

#include <QTime>
#include <QThread>
#include <QStringList>

#include <WSCoviseStub.h>

#include <util/unixcompat.h>

covise::WSEventManager::WSEventManager()
{
}

covise::WSEventManager::~WSEventManager()
{
}

void covise::WSEventManager::addEventListener(const QUuid &uuid)
{
    QMutexLocker l(&lock);
    if (!this->events.contains(uuid))
    {
        std::cerr << "WSEventManager::addEventListener info: adding " << qPrintable(uuid.toString()) << std::endl;
        this->events.insert(uuid, QList<covise::covise__Event *>());
    }
    else
    {
        std::cerr << "WSEventManager::addEventListener info: failed to add " << qPrintable(uuid.toString()) << std::endl;
    }

    // std::cerr << "-------------------------------------------" << std::endl;
    // foreach(QUuid uuid_, this->events.keys())
    //    std::cerr << qPrintable(uuid_.toString()) << " " << std::endl;
    // std::cerr << "-------------------------------------------" << std::endl;
}

void covise::WSEventManager::removeEventListener(const QUuid &uuid)
{
    QMutexLocker l(&lock);
    if (this->events.contains(uuid))
    {
        std::cerr << "WSEventManager::removeEventListener info: removing "
                  << qPrintable(uuid.toString()) << std::endl;
        QList<covise::covise__Event *> eventQueue = this->events.take(uuid);
        foreach (covise::covise__Event *e, eventQueue)
            delete e;
    }
    else
    {
        std::cerr << "WSEventManager::removeEventListener info: failed to remove " << qPrintable(uuid.toString()) << std::endl;
    }
    // std::cerr << "-------------------------------------------" << std::endl;
    // foreach(QUuid uuid_, this->events.keys())
    //    std::cerr << qPrintable(uuid_.toString()) << " " << std::endl;
    // std::cerr << "-------------------------------------------" << std::endl;
}

/**
 * Poll for an event. If no event occured, this call blocks until an event is available
 * or <i>timeout</i> msecs have elapsed. The default timeout is 10 seconds. The timeout
 * is capped at 5 minutes. Clients that did not consume events for more than 15 minutes
 * are removed automatically.
 * @param uuid the UUID of the event listener
 * @param timeout an optional timeout value in msecs (default = 10s)
 * @return a pending event or 0 if the timeout elapsed and no event was available
 */

covise::covise__Event *covise::WSEventManager::consumeEvent(const QUuid &uuid, int timeout)
{

    static const int MaximumTimeout = 300000;

    QMutexLocker l(&lock);

    // Wait at most 5 minutes
    if (timeout > MaximumTimeout)
        timeout = MaximumTimeout;

    if (!this->events.contains(uuid))
    {
#ifdef DEBUG
        std::cerr << "WSEventManager::consumeEvent err: no such UUID '"
                  << qPrintable(uuid.toString()) << "'" << std::endl;
#endif
        return 0;
    }
    else if (this->events[uuid].size() > 0)
    {
        return this->events[uuid].takeFirst();
    }

    QTime time;
    for (time.start(); time.elapsed() < timeout;)
    {
        lock.unlock();
        usleep(10000);
        lock.lock();
        if (this->events.contains(uuid) && this->events[uuid].size() > 0)
            return this->events[uuid].takeFirst();
    }

    if (time.elapsed() >= MaximumTimeout)
    {
        std::cerr << "WSEventManager::consumeEvent info: auto-expiring message listener " << qPrintable(uuid.toString()) << std::endl;
        QList<covise::covise__Event *> eList = this->events.take(uuid);
        foreach (covise::covise__Event *e, eList)
            delete e;
    }

    return 0;
}

void covise::WSEventManager::postEvent(const covise::covise__Event *event)
{
    QMutexLocker l(&lock);
    foreach (QUuid uuid, this->events.keys())
    {
        this->events[uuid].push_back(event->clone());
    }
}
