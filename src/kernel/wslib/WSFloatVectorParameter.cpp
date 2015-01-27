/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSFloatVectorParameter.h>

#include <typeinfo>

covise::WSFloatVectorParameter *covise::WSFloatVectorParameter::prototype = new covise::WSFloatVectorParameter();

covise::WSFloatVectorParameter::WSFloatVectorParameter()
    : covise::WSParameter("", "", "FloatVector")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__FloatVectorParameter).name(), this);
}

covise::WSFloatVectorParameter::WSFloatVectorParameter(const QString &name, const QString &description, const QVector<float> value)
    : covise::WSParameter(name, description, "FloatVector")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = value.toStdVector();
}
covise::WSFloatVectorParameter::~WSFloatVectorParameter()
{
}

bool covise::WSFloatVectorParameter::setValue(const QVector<float> &inValue)
{
    std::vector<float> value = inValue.toStdVector();
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

QVector<float> covise::WSFloatVectorParameter::getValue() const
{
    return QVector<float>::fromStdVector(this->parameter.value);
}

void covise::WSFloatVectorParameter::setVariantValue(const QList<QVariant> &inValue)
{
    this->parameter.value.clear();
    foreach (QVariant value, inValue)
        this->parameter.value.push_back((float)value.toDouble());
    emit parameterChanged(this);
}

QList<QVariant> covise::WSFloatVectorParameter::getVariantValue() const
{
    QList<QVariant> value;
    for (std::vector<float>::const_iterator v = this->parameter.value.begin(); v != this->parameter.value.end(); ++v)
        value << *v;
    return value;
}

QString covise::WSFloatVectorParameter::toString() const
{
    QString rv = "";
    for (std::vector<float>::const_iterator v = this->parameter.value.begin(); v != this->parameter.value.end(); ++v)
    {
        rv += QString::number(*v) + " ";
    }
    rv = rv.trimmed();
    return rv;
}

covise::WSParameter *covise::WSFloatVectorParameter::clone() const
{
    return new covise::WSFloatVectorParameter(*this);
}

const covise::covise__Parameter *covise::WSFloatVectorParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSFloatVectorParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__FloatVectorParameter *p = dynamic_cast<const covise::covise__FloatVectorParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSFloatVectorParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
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

int covise::WSFloatVectorParameter::getComponentCount() const
{
    return this->parameter.value.size();
}
