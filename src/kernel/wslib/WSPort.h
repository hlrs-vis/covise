/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSPORT_H
#define WSPORT_H

#include <QObject>
#include <QString>
#include <QStringList>

#include "WSCoviseStub.h"
#include "WSExport.h"

namespace covise
{

class WSLink;
class WSModule;

class WSLIBEXPORT WSPort : public QObject
{

    Q_OBJECT

    Q_ENUMS(PortType)
    Q_PROPERTY(QString name READ getName)
    Q_PROPERTY(QString id READ getID)
    Q_PROPERTY(QStringList types READ getTypes)
    Q_PROPERTY(PortType portType READ getPortType)

public:
    enum PortType
    {
        Default,
        Optional,
        Dependent
    };

    WSPort(const covise::WSModule *module, const QString &name, const QStringList &acceptedTypes, PortType portType);
    WSPort(const covise::WSModule *module, const covise::covise__Port &port);
    virtual ~WSPort();

public slots:

    /**
       * Get the name of the port
       */
    const QString &getName() const
    {
        return this->portName;
    }

    /**
       * Get the data types accepted or created by the port
       */
    const QStringList &getTypes() const
    {
        return this->dataTypes;
    }

    /**
       * Get the type of the port (default, optional, dependent)
       */
    PortType getPortType() const
    {
        return this->portType;
    }

    /**
       * Get the module this port belongs to
       */
    const covise::WSModule *getModule() const
    {
        return this->module;
    }

    //       void addLink(covise::WSLink * link);
    //       void removeLink(covise::WSLink * link);

    /**
       * Get the port ID
       */
    const QString &getID() const
    {
        return this->id;
    }

    //    signals:
    //       void linkAdded(covise::WSLink*);
    //       void linkRemoved(covise::WSLink*);

public:
    /**
       * Set the name of the port
       * @param inName The new value of portName
       */
    void setName(const QString &inName)
    {
        portName = inName;
    }

    /**
       * Set the data types accepted or created by the port
       * @param inType The new value of dataTypes
       */
    void setTypes(const QStringList &types)
    {
        this->dataTypes = types;
    }

    /**
       * Set the type of the port (default, optional, dependent)
       * @param inPortType The new value of portType
       */
    void setPortType(PortType inPortType)
    {
        this->portType = inPortType;
    }

    virtual covise::covise__Port getSerialisable() const;

    void setFromSerialisable(const covise::WSModule *module, const covise::covise__Port &serialisable);

private:
    // Name of the port
    QString portName;
    // Data types accepted or created by the port
    QStringList dataTypes;
    // Type of the port (default, optional, dependent)
    PortType portType;

    const covise::WSModule *module;

    QString id;

    //QList<covise::WSLink*> links;
};
}

#endif // WSPORT_H
