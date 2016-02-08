/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSSTRINGPARAMETER_H
#define WSSTRINGPARAMETER_H

#include "WSExport.h"
#include "WSParameter.h"
#include "WSTools.h"

namespace covise
{

class WSLIBEXPORT WSStringParameter : public WSParameter
{

    Q_OBJECT

    Q_PROPERTY(QString value READ getValue WRITE setValue)

public:
    WSStringParameter(const QString &name, const QString &description, const QString &value = QString::null);

    virtual ~WSStringParameter();

public slots:
    /**
       * Set the value of the parameter
       * @param inValue The new value of value
       * @return true if the parameter was changed
       */
    bool setValue(const QString &inValue);

    /**
       * Get the value of the parameter
       * @return value
       */
    const QString getValue() const;

    virtual QString toString() const;

public:
    virtual WSParameter *clone() const;

    virtual const covise::covise__Parameter *getSerialisable();

protected:
    virtual bool setValueFromSerialisable(const covise::covise__Parameter *serialisable);

    virtual QString toCoviseString() const;

private:
    covise::covise__StringParameter parameter;
    WSStringParameter();
    static WSStringParameter *prototype;
};
}
#endif // WSSTRINGPARAMETER_H
