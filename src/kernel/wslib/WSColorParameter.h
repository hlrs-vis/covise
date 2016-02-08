/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSCOLORPARAMETER_H
#define WSCOLORPARAMETER_H

#include "WSExport.h"
#include "WSParameter.h"

class WSColorParameter : public WSParameter
{
public:
    WSColorParameter();
    ~WSColorParameter();

    /**
       * Set the value of the parameter
       * @param inValue The new value of red
       */
    void setValue(QString inValue)
    {
        red = inValue;
    }

    /**
       * Get the value of the parameter
       * @return red
       */
    QString getRed()
    {
        return red;
    }

    /**
       * Set the value of the parameter
       * @param inValue The new value of green
       */
    void setGreen(QString inValue)
    {
        green = inValue;
    }

    /**
       * Get the value of the parameter
       * @return green
       */
    QString getGreen()
    {
        return green;
    }

    /**
       * Set the value of the parameter
       * @param inValue The new value of blue
       */
    void setBlue(QString inValue)
    {
        blue = inValue;
    }

    /**
       * Get the value of the parameter
       * @return blue
       */
    QString getBlue()
    {
        return blue;
    }

    /**
       * Set the value of the parameter
       * @param inValue The new value of alpha
       */
    void setAlpha(QString inValue)
    {
        alpha = inValue;
    }

    /**
       * Get the value of the parameter
       * @return alpha
       */
    QString getAlpha()
    {
        return alpha;
    }

private:
    QString red;
    QString green;
    QString blue;
    QString alpha;
};

#endif // WSCOLORPARAMETER_H
