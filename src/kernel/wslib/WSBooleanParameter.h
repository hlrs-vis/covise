/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSBOOLEANPARAMETER_H
#define WSBOOLEANPARAMETER_H

#include "WSExport.h"
#include "WSParameter.h"

namespace covise
{

class WSLIBEXPORT WSBooleanParameter : public WSParameter
{
    Q_OBJECT

    Q_PROPERTY(bool value READ getValue WRITE setValue)

public:
    WSBooleanParameter(const QString &name, const QString &description, bool value = false);

    virtual ~WSBooleanParameter();

public slots:
    /**
       * Set the value of the parameter
       * @param inValue The new value of value
       */
    bool setValue(bool inValue);

    /**
       * Get the value of the parameter
       * @return value
       */
    bool getValue() const;

    virtual QString toString() const;

public:
    virtual WSParameter *clone() const;

    virtual const covise::covise__Parameter *getSerialisable();

protected:
    virtual bool setValueFromSerialisable(const covise::covise__Parameter *serialisable);

private:
    covise::covise__BooleanParameter parameter;

    WSBooleanParameter();
    static WSBooleanParameter *prototype;
};
}
#endif // WSBOOLEANPARAMETER_H
