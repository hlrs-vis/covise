/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSFLOATSLIDERPARAMETER_H
#define WSFLOATSLIDERPARAMETER_H

#include "WSParameter.h"

namespace covise
{

class WSLIBEXPORT WSFloatSliderParameter : public WSParameter
{

    Q_OBJECT

    Q_PROPERTY(float value READ getValue WRITE setValue)
    Q_PROPERTY(float min READ getMin WRITE setMin)
    Q_PROPERTY(float max READ getMax WRITE setMax)

public:
    WSFloatSliderParameter(const QString &name, const QString &description);

    WSFloatSliderParameter(const QString &name, const QString &description, float value, float min, float max);

    virtual ~WSFloatSliderParameter();

public slots:
    /**
       * Set the minimum value of the slider
       * @param inMin The new value of min
       */
    void setMin(float inMin);

    /**
       * Get the minimum value of the slider
       * @return min
       */
    float getMin() const;

    /**
       * Set the maximum value of the slider
       * @param inMax The new value of max
       */
    void setMax(float inMax);

    /**
       * Get the maximum value of the slider
       * @return max
       */
    float getMax() const;

    /**
       * Set the value of the slider
       * @param inValue The new value of value
       * @return true if the parameter was changed
       */
    bool setValue(float inValue);

    /**
       * Set the value of the slider
       * @param value The new value
       * @param minimum The new minimum of the slider
       * @param maximum The new maximum of the slider
       * @return true if the parameter was changed
       */
    bool setValue(float value, float minimum, float maximum);

    /**
       * Get the value of the slider
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
    covise::covise__FloatSliderParameter parameter;
    WSFloatSliderParameter();
    static WSFloatSliderParameter *prototype;
};
}

#endif // WSFLOATSLIDERPARAMETER_H
