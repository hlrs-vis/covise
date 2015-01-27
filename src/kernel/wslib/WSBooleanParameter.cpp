/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSBooleanParameter.h>

#include <typeinfo>

covise::WSBooleanParameter *covise::WSBooleanParameter::prototype = new covise::WSBooleanParameter();

covise::WSBooleanParameter::WSBooleanParameter()
    : covise::WSParameter("", "", "Boolean")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__BooleanParameter).name(), this);
}

covise::WSBooleanParameter::WSBooleanParameter(const QString &name, const QString &description, bool value)
    : covise::WSParameter(name, description, "Boolean")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = value;
}

covise::WSBooleanParameter::~WSBooleanParameter() {}

bool covise::WSBooleanParameter::setValue(bool inValue)
{
    bool changed = this->parameter.value != inValue;

    this->parameter.value = inValue;

    if (changed)
        emit parameterChanged(this);

    return changed;
}

bool covise::WSBooleanParameter::getValue() const
{
    return this->parameter.value;
}

QString covise::WSBooleanParameter::toString() const
{
    return this->parameter.value ? "TRUE" : "FALSE";
}

covise::WSParameter *covise::WSBooleanParameter::clone() const
{
    return new WSBooleanParameter(*this);
}

const covise::covise__Parameter *covise::WSBooleanParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSBooleanParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__BooleanParameter *p = dynamic_cast<const covise::covise__BooleanParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSBooleanParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
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
