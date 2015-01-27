/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSFloatScalarParameter.h>

#include <typeinfo>

covise::WSFloatScalarParameter *covise::WSFloatScalarParameter::prototype = new covise::WSFloatScalarParameter();

covise::WSFloatScalarParameter::WSFloatScalarParameter()
    : covise::WSParameter("", "", "FloatScalar")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__FloatScalarParameter).name(), this);
}

covise::WSFloatScalarParameter::WSFloatScalarParameter(const QString &name, const QString &description, float value)
    : covise::WSParameter(name, description, "FloatScalar")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = value;
}

covise::WSFloatScalarParameter::~WSFloatScalarParameter()
{
}

bool covise::WSFloatScalarParameter::setValue(float inValue)
{
    bool changed = this->parameter.value != inValue;
    this->parameter.value = inValue;
    if (changed)
        emit parameterChanged(this);
    //dumpObjectInfo();
    return changed;
}

float covise::WSFloatScalarParameter::getValue() const
{
    return this->parameter.value;
}

QString covise::WSFloatScalarParameter::toString() const
{
    return QString::number(this->parameter.value);
}

covise::WSParameter *covise::WSFloatScalarParameter::clone() const
{
    return new WSFloatScalarParameter(*this);
}

const covise::covise__Parameter *covise::WSFloatScalarParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSFloatScalarParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__FloatScalarParameter *p = dynamic_cast<const covise::covise__FloatScalarParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSFloatScalarParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
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
