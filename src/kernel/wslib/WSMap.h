/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSMAP_H
#define WSMAP_H

#include "WSModule.h"

#include <QObject>

namespace covise
{

class WSLink;

class WSLIBEXPORT WSMap : public QObject
{

    Q_OBJECT

    Q_PROPERTY(QString mapname READ getMapName WRITE setMapName)

    Q_DISABLE_COPY(WSMap)

public:
    WSMap();
    virtual ~WSMap();

    /**
       * Adds a module to the map of running modules. Takes its parameter as template and clones it. This method
       * is intended to be used when instantiating non-instantiated modules
       * @param module the module to be added
       * @param instance the module instance id
       * @param host the host the module is running on
       * @returns the module created
       */
    WSModule *addModule(const WSModule *module, const QString &instance, const QString &host);

    /**
       * Adds a module to the map of running modules. Takes its parameter as template and clones it. This method
       * is intended to be used when adding already instantiated modules
       * @param module the module to be added
       * @returns the module created
       */
    WSModule *addModule(const WSModule *module);

    /**
       * Adds a module to the map of running modules. Takes its parameter as template and clones it. This method
       * is intended to be used when adding already instantiated modules
       * @param module the module to be added
       * @returns the module created
       */
    WSModule *addModule(const covise::covise__Module &module);

    /**
       * Remove a module from the map
       * @returns the name of the the removed module
       * @param moduleID the moduleID of the module to be removed
       */
    const QString &removeModule(const QString &moduleID);

    /**
       * Removes a module from the map without deleting it
       * @returns the removed module or 0 if no module with moduleID found
       * @param  moduleID the moduleID of the module to be removed
       */
    WSModule *takeModule(const QString &moduleID);

public slots:

    /**
       * Add a link between an output port and an input port of a module
       * @param fromModule Module ID of the output port
       * @param fromPort Port name of the output port
       * @param toModule Module ID of the input port
       * @param toPort Port name of the input port
       * @return the new link created or 0 on error
       */
    WSLink *link(const QString &fromModule, const QString &fromPort, const QString &toModule, const QString &toPort);
    bool unlink(const QString &fromModule, const QString &fromPort, const QString &toModule, const QString &toPort);
    bool unlink(WSLink *link);
    bool unlink(const QString &linkID);

    WSLink *getLink(const QString &linkID) const;
    QList<WSLink *> getLinks() const;

    /**
       * @return the module or 0 if no module with that name/instance/host combination exists
       * @param  name The name of the module
       * @param  instance The number of the module
       * @param  host The IP of the host the module is running on
       */
    WSModule *getModule(const QString &name, const QString &instance, const QString &host) const;

    /**
       * @param  moduleID The id of the module
       * @return the module or 0 if no module with that moduleID exists
       */
    WSModule *getModule(const QString &moduleID) const;

    /**
       * Gets a module by title. If no module has the given title, 0 is returned.
       * If more than one module has that title, the result is undefined.
       * @param  title The title of the module
       * @return The module with the given title, 0 if no module has that title.
       */
    WSModule *getModuleByTitle(const QString &title) const;

    /**
       * Get a list of all running modules
       */
    QList<WSModule *> getModules() const;

    /**
       * Set the name of the map
       * @param name the name of the map
       */
    void setMapName(const QString &name);

    /**
       * Get the name of the currently loaded map
       * @return the name of the currently loaded map
       */
    const QString &getMapName() const
    {
        return this->mapName;
    }

    /**
       * Make unique name for module
       * @param name The name of the running module
       * @param instance The instance of the running module
       * @param host The host the module is running on
       */
    static QString makeKeyName(const QString &name, const QString &instance, const QString &host);

signals:
    void moduleAdded(WSModule *module);
    void moduleRemoved(const QString &moduleID);

private:
    // The running modules
    QMap<QString, WSModule *> runningModules;

    // The name of the map
    QString mapName;

    QMap<QString, covise::WSLink *> links;

    /**
       * Set the running Modules
       * @param inModules The new value of runningModules
       */
    void setRunningModules(const QMap<QString, WSModule *> &modules)
    {
        runningModules = modules;
    }

    /**
       * Print content of a map
       * @param  map The name of the QMap to be printed
       */
    void printList(const QMap<QString, WSModule *> &map);

private slots:
    void linkDestroyed(const QString &linkID);
};
}
#endif // WSMAP_H
