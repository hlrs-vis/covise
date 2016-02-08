/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSMODULE_H
#define WSMODULE_H

#include "WSExport.h"
#include "WSParameter.h"
#include "WSPort.h"

#include "WSCoviseStub.h"

#include <QPoint>
#include <QObject>

namespace covise
{

class WSLIBEXPORT WSModule : public QObject
{

    Q_OBJECT

    Q_PROPERTY(QString name READ getName)
    Q_PROPERTY(QString id READ getID)
    Q_PROPERTY(QString host READ getHost)
    Q_PROPERTY(QString description READ getDescription)
    Q_PROPERTY(QString category READ getCategory)
    Q_PROPERTY(QString title READ getTitle WRITE setTitle)
    Q_PROPERTY(QString instance READ getInstance)
    Q_PROPERTY(bool dead READ isDead)

public:
    WSModule(const QString &name, const QString &category, const QString &host);
    explicit WSModule(const covise::covise__Module &module);
    ~WSModule();

    /**
       * @return WSPort *
       * @param  name The name of input port
       * @param  types The data types that can be connected to this port
       * @param  portType Type of the port
       */
    WSPort *addInputPort(const QString &inName, const QStringList &inTypes, WSPort::PortType inPortType = WSPort::Default);

    /**
       * @return WSPort *
       * @param  name The name of output port
       * @param  types The data types that can be connected to this port
       * @param  portType Type of the port
       */
    WSPort *addOutputPort(const QString &name, const QStringList &types, WSPort::PortType portType = WSPort::Default);

    /**
       * @return WSParameter *
       * @param  name The name of parameter
       * @param  type The type of parameter
       * @param  description Description of the parameter
       */
    WSParameter *addParameter(const QString &name, const QString &type, const QString &description);

    WSPort *getOutputPort(const QString &name) const;
    WSPort *getInputPort(const QString &name) const;

    /**
       * Get the output ports
       * @return QMap: inputPorts
       */
    const QMap<QString, WSPort *> &getInputPorts() const
    {
        return this->inputPorts;
    }

    /**
       * Get the output ports
       * @return QMap: outputPorts
       */
    const QMap<QString, WSPort *> &getOutputPorts() const
    {
        return this->outputPorts;
    }

    /**
       * Get the parameters of the module
       * @return QMap: parameters
       */
    const QMap<QString, WSParameter *> &getParameters() const
    {
        return this->parameters;
    }

signals:
    void changed();
    void parameterChanged(covise::WSParameter *);
    void died();
    void deleted(const QString &moduleID);

public slots:

    // All return values have to be by value for the scripting interface to function

    /**
       * Get the host the module is running on, may be empty
       * @return QString: host
       */
    QString getHost() const
    {
        return this->host;
    }

    /**
       * Get the name of the module
       * @return QString: name
       */
    QString getName() const
    {
        return this->name;
    }

    /**
       * Get the ID of the module
       * @return QString: name
       */
    QString getID() const
    {
        return this->id;
    }

    /**
       * Get the description of module
       * @return QString: description
       */
    QString getDescription() const
    {
        return this->description;
    }

    /**
       * Get the category of the module
       * @return QString: category
       */
    QString getCategory() const
    {
        return this->category;
    }

    /**
       * Set the title of the module
       * @param title The new value of category
       */
    void setTitle(const QString &title);

    /**
       * Get the title of the module
       * @return QString: title
       */
    QString getTitle() const
    {
        return this->title;
    }

    /**
       * Get a named parameter of the module
       * @return WSParameter: the parameter with given name or 0 if nothing found
       */
    WSParameter *getParameter(const QString &name) const;

    /**
       * Get the location on the canvas of the module
       * @return QPoint: the X and Y position
       */
    QPoint getPosition() const
    {
        return this->position;
    }

    /**
       * Set the location on the canvas of the module
       * @param x the X position
       * @param y the Y position
       */
    void setPosition(int x, int y)
    {
        this->position.setX(x);
        this->position.setY(y);
        emit changed();
    }

    QString getInstance() const
    {
        return this->instance;
    }

    QStringList getParameterNames() const;

    bool isDead() const
    {
        return this->dead;
    }

public:
    virtual covise::covise__Module getSerialisable() const;
    void setFromSerialisable(const covise::covise__Module &serialisable);

    /**
       * This method is called when a non-running module is instantiated.
       */
    void instantiate(const QString &host, const QString &instance);

    /**
       * Set the module ID. Not propagated to COVISE.
       * @param id the new module ID
       */
    void setID(const QString &id)
    {
        this->id = id;
        emit changed();
    }

    /**
       * Set the category of the module. Not propagated to COVISE.
       * @param category the new category
       */
    void setCategory(const QString &category)
    {
        this->category = category;
        emit changed();
    }

    /**
       * Set the description of the module. Not propagated to COVISE.
       * @param description the new description
       */
    void setDescription(const QString &description)
    {
        this->description = description;
        emit changed();
    }

    /**
       * Marks the module as crashed. Not propagated to COVISE.
       * @param dead if true marks the module as dead.
       */
    void setDead(bool dead);

private slots:
    void parameterChangedCB(covise::WSParameter *);

private:
    // Name of module
    QString name;
    // ID of a (running) module
    QString id;
    // Host the module is running on, may be empty
    QString host;
    // Description of module
    QString description;
    // Category of module
    QString category;
    // Module title
    QString title;
    // Module instance
    QString instance;
    // Parameters of the module with name of each parameter
    QMap<QString, WSParameter *> parameters;
    // Input ports of the module with name of each port
    QMap<QString, WSPort *> inputPorts;
    // Output ports of the module with name of each port
    QMap<QString, WSPort *> outputPorts;

    // Position of the module on the canvas
    QPoint position;

    bool dead;
};
}
#endif // WSMODULE_H
