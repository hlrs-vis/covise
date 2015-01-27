/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSPARAMETER_H
#define WSPARAMETER_H

#include <QString>
#include <QStringList>
#include <QVector>
#include <QMap>
#include <QVariant>

#include <map>

#include "WSCoviseStub.h"

#include <QObject>

namespace covise
{

class WSLIBEXPORT WSParameter : public QObject
{

    Q_OBJECT

    Q_PROPERTY(QString name READ getName)
    Q_PROPERTY(QString type READ getType)
    Q_PROPERTY(QString description READ getDescription)
    Q_PROPERTY(bool enabled READ isEnabled WRITE setEnabled)
    Q_PROPERTY(bool mapped READ isMapped WRITE setMapped)

public:
    WSParameter(const QString &name, const QString &description, const QString &type);
    virtual ~WSParameter();

signals:
    void parameterChanged(covise::WSParameter *parameter);

public slots:
    /**
       * Get the name of the parameter
       * @return QString: name
       */
    const QString &getName() const
    {
        return this->name;
    }

    /**
       * Get type of the parameter
       * @return QString: type
       */
    const QString &getType() const
    {
        return this->type;
    }

    /**
       * Get the description of the paramter
       * @return QString: description
       */
    const QString &getDescription() const
    {
        return this->description;
    }

    /**
       * Enable or disable a parameter
       * @param state true when the parameter should be enabled.
       */
    void setEnabled(bool state)
    {
        this->enabled = state;
    }

    /**
       * Get the parameter state (enabled or disabled)
       * @return bool: true, when enabled
       */
    bool isEnabled() const
    {
        return this->enabled;
    }

    /**
       * Map the parameter to the control panel;
       */
    void setMapped(bool mapped)
    {
        this->mapped = mapped;
    }

    /**
       * Returns true, if the parameter is mapped in the control panel
       * @return bool: true, when mapped
       */
    bool isMapped() const
    {
        return this->mapped;
    }

    /**
       * Gets the number of components in a parameter. For non-vector parameter
       * returns 1.
       * @return int: The number of vector components
       */
    virtual int getComponentCount() const;

    /**
       * Get the value as string
       * @return QString: string representation of the value
       */
    virtual QString toString() const = 0;

    virtual WSParameter *clone() const = 0;

    virtual const covise::covise__Parameter *getSerialisable() = 0;

    /**
       * Sets the value from a native WS parameter
       * @return true, if the parameter was changed, false if the parameter did not change.
       */
    virtual bool setValueFromSerialisable(const covise::covise__Parameter *serialisable) = 0;

    virtual QString toCoviseString() const
    {
        QString value = toString();
        if (value == "")
            return QChar('\177');
        else
            return value;
    }

    static WSParameter *create(covise::covise__Parameter *parameter);

protected:
    const covise::covise__Parameter *getSerialisable(covise::covise__Parameter *parameter) const;

    static WSParameter *addPrototype(QString className, WSParameter *prototype)
    {
        static std::map<QString, WSParameter *> *p = new std::map<QString, WSParameter *>();
        //std::cerr << "WSParameter::addPrototype info: adding " << qPrintable(className) << std::endl;
        (*p)[className] = prototype;
        prototypes = p;
        return prototype;
    }

    bool equals(const covise::covise__Parameter *, const covise::covise__Parameter *);

protected:
    WSParameter(const WSParameter &);

private:
    // Name of the parameter
    QString name;
    // Type of the parameter
    QString type;
    // Description of the parameter
    QString description;

    bool enabled;
    bool mapped;

    static std::map<QString, WSParameter *> *prototypes;

    /**
       * Set the name of the parameter
       * @param inName The new value of name
       */
    void setName(const QString &inName)
    {
        this->name = inName;
    }

    /**
       * Set type of the parameter
       * @param inType The new value of type
       */
    void setType(const QString &inType)
    {
        this->type = inType;
    }

    /**
       * Set description of the parameter
       * @param inDescription The new value of description
       */
    void setDescription(const QString &inDescription)
    {
        this->description = inDescription;
    }
};
}

#endif // WSPARAMETER_H
