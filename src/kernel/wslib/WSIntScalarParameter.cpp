/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSIntScalarParameter.h>

#include <typeinfo>

covise::WSIntScalarParameter *covise::WSIntScalarParameter::prototype = new covise::WSIntScalarParameter();

covise::WSIntScalarParameter::WSIntScalarParameter()
    : covise::WSParameter("", "", "IntScalar")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__IntScalarParameter).name(), this);
}

covise::WSIntScalarParameter::WSIntScalarParameter(const QString &name, const QString &description, int value)
    : covise::WSParameter(name, description, "IntScalar")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = value;
}

covise::WSIntScalarParameter::~WSIntScalarParameter()
{
}

bool covise::WSIntScalarParameter::setValue(int inValue)
{
    bool changed = this->parameter.value != inValue;
    this->parameter.value = inValue;
    if (changed)
        emit parameterChanged(this);
    return changed;
}

int covise::WSIntScalarParameter::getValue() const
{
    return this->parameter.value;
}

QString covise::WSIntScalarParameter::toString() const
{
    return QString::number(this->parameter.value);
}

covise::WSParameter *covise::WSIntScalarParameter::clone() const
{
    return new covise::WSIntScalarParameter(*this);
}

const covise::covise__Parameter *covise::WSIntScalarParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSIntScalarParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__IntScalarParameter *p = dynamic_cast<const covise::covise__IntScalarParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSIntScalarParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
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
