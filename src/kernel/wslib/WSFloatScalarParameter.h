/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSFLOATSCALARPARAMETER_H
#define WSFLOATSCALARPARAMETER_H

#include "WSParameter.h"

namespace covise
{

class WSLIBEXPORT WSFloatScalarParameter : public WSParameter
{

    Q_OBJECT

    Q_PROPERTY(float value READ getValue WRITE setValue)

public:
    WSFloatScalarParameter(const QString &name, const QString &description, float value = 0.0f);

    virtual ~WSFloatScalarParameter();

public slots:
    /**
       * Set the value of the scalar parameter
       * @param inValue The new value of value
       * @return true if the parameter was changed
       */
    bool setValue(float inValue);

    /**
       * Get the value of the scalar parameter
       * @return value
       */
    float getValue() const;

    virtual QString toString() const;

public:
    virtual WSParameter *clone() const;

    virtual const covise::covise__Parameter *getSerialisable();

protected:
    virtual bool setValueFromSerialisable(const covise::covise__Parameter *serialisable);

private:
    covise::covise__FloatScalarParameter parameter;
    WSFloatScalarParameter();
    static WSFloatScalarParameter *prototype;
};
}

#endif // WSFLOATSCALARPARAMETER_H
