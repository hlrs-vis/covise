/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSINTSLIDERPARAMETER_H
#define WSINTSLIDERPARAMETER_H

#include "WSParameter.h"

namespace covise
{

class WSLIBEXPORT WSIntSliderParameter : public WSParameter
{
    Q_OBJECT

    Q_PROPERTY(int value READ getValue WRITE setValue)
    Q_PROPERTY(int min READ getMin WRITE setMin)
    Q_PROPERTY(int max READ getMax WRITE setMax)

public:
    WSIntSliderParameter(const QString &name, const QString &description);

    WSIntSliderParameter(const QString &name, const QString &description, int value, int min, int max);

    virtual ~WSIntSliderParameter();

public slots:
    /**
       * Set the minimum value of the slider
       * @param inMin The new value of min
       */
    void setMin(int inMin);

    /**
       * Get the minimum value of the slider
       * @return min
       */
    int getMin() const;

    /**
       * Set the maximum value of the slider
       * @param inMax The new value of max
       */
    void setMax(int inMax);

    /**
       * Get the maximum value of the slider
       * @return max
       */
    int getMax() const;

    /**
       * Set the value of the slider
       * @param inValue the new value of value
       * @return true if the parameter was changed
       */
    bool setValue(int inValue);

    /**
       * Set the value of the slider
       * @param value The new value
       * @param minimum The new minimum of the slider
       * @param maximum The new maximum of the slider
       * @return true if the parameter was changed
       */
    bool setValue(int value, int minimum, int maximum);

    /**
       * Get the value of the slider
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
    covise::covise__IntSliderParameter parameter;
    WSIntSliderParameter();
    static WSIntSliderParameter *prototype;
};
}
#endif // WSINTSLIDERPARAMETER_H
