/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSCOVISECLIENT_H
#define WSCOVISECLIENT_H

#include <QList>
#include <QMap>
#include <QMutex>
#include <QObject>
#include <QQueue>
#include <QString>
#include <QStringList>
#include <QThread>

#include "WSExport.h"

namespace covise
{

class WSMap;
class WSModule;
class WSParameter;
class covise__Event;

class COVISEProxy;

class WSLIBEXPORT WSCOVISEClient : public QThread
{

    Q_OBJECT
    Q_DISABLE_COPY(WSCOVISEClient)

    Q_PROPERTY(WSMap *map READ getMap)
    Q_PROPERTY(QString endpoint READ getEndpoint)
    Q_PROPERTY(bool inExecute READ isInExecute)

public:
    WSCOVISEClient();
    virtual ~WSCOVISEClient();

public slots:
    /// Attach to a running COVISE session
    bool attach(const QString &endpoint = "http://localhost:31111/");

    /// Detach from COVISE
    bool detach();

    const QString &getEndpoint() const
    {
        return this->endpoint;
    }

    bool isInExecute() const;

    /// Deliver events using signals
    void setEventsAsSignal(bool on, bool alsoQueueRaw = false);

    /// Don't send changes to COVISE
    void setReadOnly(bool ro);
    bool isReadOnly() const;

    /// Gets the current map of instantiated modules
    WSMap *getMap() const;

    /// Get an available module by name and host
    WSModule *getModule(const QString &name, const QString &host) const;

    /// Get all available modules for a host
    QList<WSModule *> getModules(const QString &host) const;

    /// Get all hosts currently in a session
    QStringList getHosts() const;

    /// Execute the whole pipeline
    void executeNet();

    /// Execute a single module
    void executeModule(const QString &moduleID);

    /// Set a parameter using a string representation
    void setParameterFromString(const QString &moduleID, const QString &parameter, const QString &value);

    /// Get a parameter as string representation
    QString getParameterAsString(const QString &moduleID, const QString &parameter);

    /// Instantiate an available module on a certain host
    void instantiateModule(const QString &module, const QString &host);

    /// Deletes an instantiated module
    void deleteModule(const QString &moduleID);

    void link(const QString &fromModuleID, const QString &fromPort, const QString &toModuleID, const QString &toPort);
    void unlink(const QString &linkID);

    /// Open a map
    void openNet(const QString &filename);

    /// Quit COVISE
    void quit();

    /// Check for events
    covise::covise__Event *takeEvent();

signals:
    void eventLink(const QString &fromModuleID, const QString &toModuleID);
    void eventUnlink(const QString &linkID);
    void eventModuleAdd(const QString &moduleID);
    void eventModuleDel(const QString &moduleID);
    void eventModuleDied(const QString &moduleID);
    void eventModuleChanged(const QString &moduleID);
    void eventModuleExecuteStart(const QString &moduleID);
    void eventModuleExecuteFinish(const QString &moduleID);
    void eventExecuteStart();
    void eventExecuteFinish();

    void eventParameterChanged(const QString &moduleID, const QString &name, const QString &value);
    void eventOpenNet(const QString &mapname);
    void eventOpenNetDone(const QString &mapname);
    void eventQuit();

private slots:
    void parameterChangeCB(covise::WSParameter *);

private:
    void clearAvailableModules();

    virtual void run();

    // Contains a map of the host and the modules available on that host
    QMap<QString, QList<covise::WSModule *> > availableModules;
    // The loaded modules
    covise::WSMap *map;

    QString endpoint;
    bool attached;

    bool keepRunning;

    QString eventUUID;

    QQueue<covise::covise__Event *> eventQueue;
    QMutex eventQueueLock;

    bool eventsAsSignal;
    bool alsoQueueRaw;

    bool readOnly;
    bool inExecute;
};
}

#endif // WSCOVISECLIENT_H
