/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSINTVECTORPARAMETER_H
#define WSINTVECTORPARAMETER_H

#include "WSParameter.h"

#include <QList>
#include <QVariant>

namespace covise
{

class WSLIBEXPORT WSIntVectorParameter : public WSParameter
{
    Q_OBJECT

    Q_PROPERTY(QList<QVariant> value READ getVariantValue WRITE setVariantValue)

public:
    WSIntVectorParameter(const QString &name, const QString &description, const QVector<int> value = QVector<int>());

    virtual ~WSIntVectorParameter();

public slots:
    /**
       * Set the value of the vector
       * @param inValue The new value of value
       * @return true if the parameter was changed
       */
    bool setValue(const QVector<int> &inValue);

    /**
       * Get the value of the vector
       * @return value
       */
    QVector<int> getValue() const;

    virtual QString toString() const;

    virtual int getComponentCount() const;

public:
    virtual WSParameter *clone() const;

    virtual const covise::covise__Parameter *getSerialisable();

protected:
    virtual bool setValueFromSerialisable(const covise::covise__Parameter *serialisable);

private slots:
    void setVariantValue(const QList<QVariant> &inValue);
    QList<QVariant> getVariantValue() const;

private:
    covise::covise__IntVectorParameter parameter;
    WSIntVectorParameter();
    static WSIntVectorParameter *prototype;
};
}
#endif // WSINTVECTORPARAMETER_H
