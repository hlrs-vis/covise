/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSIntVectorParameter.h>

#include <typeinfo>

covise::WSIntVectorParameter *covise::WSIntVectorParameter::prototype = new covise::WSIntVectorParameter();

covise::WSIntVectorParameter::WSIntVectorParameter()
    : covise::WSParameter("", "", "IntVector")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__IntVectorParameter).name(), this);
}

covise::WSIntVectorParameter::WSIntVectorParameter(const QString &name, const QString &description, const QVector<int> value)
    : covise::WSParameter(name, description, "IntVector")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = value.toStdVector();
}

covise::WSIntVectorParameter::~WSIntVectorParameter()
{
}

bool covise::WSIntVectorParameter::setValue(const QVector<int> &inValue)
{
    std::vector<int> value = inValue.toStdVector();
    if (this->parameter.value.size() != inValue.size() || !std::equal(value.begin(), value.end(), this->parameter.value.begin()))
    {
        this->parameter.value = value;
        emit parameterChanged(this);
        return true;
    }
    else
    {
        return false;
    }
}

QVector<int> covise::WSIntVectorParameter::getValue() const
{
    return QVector<int>::fromStdVector(this->parameter.value);
}

void covise::WSIntVectorParameter::setVariantValue(const QList<QVariant> &inValue)
{
    this->parameter.value.clear();
    foreach (QVariant value, inValue)
        this->parameter.value.push_back(value.toInt());
    emit parameterChanged(this);
}

QList<QVariant> covise::WSIntVectorParameter::getVariantValue() const
{
    QList<QVariant> value;
    for (std::vector<int>::const_iterator v = this->parameter.value.begin(); v != this->parameter.value.end(); ++v)
        value << *v;
    return value;
}

QString covise::WSIntVectorParameter::toString() const
{
    QString rv = "";
    for (std::vector<int>::const_iterator v = this->parameter.value.begin(); v != this->parameter.value.end(); ++v)
    {
        rv += QString::number(*v) + " ";
    }
    rv = rv.trimmed();
    return rv;
}

covise::WSParameter *covise::WSIntVectorParameter::clone() const
{
    return new covise::WSIntVectorParameter(*this);
}

const covise::covise__Parameter *covise::WSIntVectorParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSIntVectorParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__IntVectorParameter *p = dynamic_cast<const covise::covise__IntVectorParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSIntVectorParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
                  << " of type " << serialisable->type << std::endl;
        return false;
    }

    bool changed = !equals(&(this->parameter), p) || this->parameter.value != p->value;

    if (changed)
    {
        this->parameter = *p;
        emit parameterChanged(this);
        return true;
    }
    else
    {
        return false;
    }
}

int covise::WSIntVectorParameter::getComponentCount() const
{
    return this->parameter.value.size();
}
