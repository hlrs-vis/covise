/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSCHOICEPARAMETER_H
#define WSCHOICEPARAMETER_H

#include "WSExport.h"
#include "WSParameter.h"
#include "WSTools.h"

namespace covise
{

class WSLIBEXPORT WSChoiceParameter : public WSParameter
{

    Q_OBJECT

    Q_PROPERTY(QStringList value READ getValue WRITE setValue)
    Q_PROPERTY(int selected READ getSelected WRITE setSelected)
    Q_PROPERTY(QString selectedValue READ getSelectedValue)

public:
    WSChoiceParameter(const QString &name, const QString &description,
                      const QStringList &inValue = QStringList(), int selected = 1);

    virtual ~WSChoiceParameter();

public slots:
    /**
       * Set the values and selected value of the parameter
       * @param inValue The new value of value
       */
    bool setValue(const QStringList &inValue, int selected = 1);

    /**
       * Set the selected value of the parameter
       * @param inValue The new value of value
       */
    bool setValue(int selected);

    /**
       * Get the value of the parameter
       * @return value
       */
    QStringList getValue() const;

    const QString getSelectedValue();

    bool setSelected(int index);

    int getSelected() const;

    virtual int getComponentCount() const;

public:
    virtual QString toString() const;

    virtual WSParameter *clone() const;

    virtual const covise::covise__Parameter *getSerialisable();

protected:
    virtual bool setValueFromSerialisable(const covise::covise__Parameter *serialisable);

    virtual QString toCoviseString() const;

private:
    bool setValue(int selected, bool changed);
    covise::covise__ChoiceParameter parameter;
    WSChoiceParameter();
    static WSChoiceParameter *prototype;
};
}

#endif // WSCHOICEPARAMETER_H
