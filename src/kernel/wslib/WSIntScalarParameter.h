/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSINTSCALARPARAMETER_H
#define WSINTSCALARPARAMETER_H

#include "WSExport.h"
#include "WSParameter.h"

namespace covise
{

class WSLIBEXPORT WSIntScalarParameter : public WSParameter
{
    Q_OBJECT

    Q_PROPERTY(int value READ getValue WRITE setValue)

public:
    WSIntScalarParameter(const QString &name, const QString &description, int value = 0);

    virtual ~WSIntScalarParameter();

public slots:
    /**
       * Set the value of value
       * @param inValue the new value of value
       * @return true if the parameter was changed
       */
    bool setValue(int inValue);

    /**
       * Get the value of value
       * @return the value of value
       */
    int getValue() const;

    virtual QString toString() const;

public:
    virtual WSParameter *clone() const;

    virtual const covise::covise__Parameter *getSerialisable();

protected:
    virtual bool setValueFromSerialisable(const covise::covise__Parameter *serialisable);

private:
    covise::covise__IntScalarParameter parameter;
    WSIntScalarParameter();
    static WSIntScalarParameter *prototype;
};
}
#endif // WSINTSCALARPARAMETER_H
