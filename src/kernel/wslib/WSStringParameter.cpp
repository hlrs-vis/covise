/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSStringParameter.h>

#include <typeinfo>

covise::WSStringParameter *covise::WSStringParameter::prototype = new covise::WSStringParameter();

covise::WSStringParameter::WSStringParameter()
    : covise::WSParameter("", "", "String")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__StringParameter).name(), this);
}

covise::WSStringParameter::WSStringParameter(const QString &name, const QString &description, const QString &value)
    : covise::WSParameter(name, description, "String")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = value.toStdString();
}

covise::WSStringParameter::~WSStringParameter()
{
}

bool covise::WSStringParameter::setValue(const QString &inValue)
{
    bool changed = this->parameter.value != WSTools::fromCovise(inValue).toStdString();
    this->parameter.value = WSTools::fromCovise(inValue).toStdString();
    if (changed)
        emit parameterChanged(this);
    return changed;
}

const QString covise::WSStringParameter::getValue() const
{
    return toString();
}

QString covise::WSStringParameter::toString() const
{
    return QString::fromStdString(this->parameter.value);
}

covise::WSParameter *covise::WSStringParameter::clone() const
{
    return new covise::WSStringParameter(*this);
}

const covise::covise__Parameter *covise::WSStringParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSStringParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__StringParameter *p = dynamic_cast<const covise::covise__StringParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSStringParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
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

QString covise::WSStringParameter::toCoviseString() const
{
    return WSTools::toCovise(toString());
}
