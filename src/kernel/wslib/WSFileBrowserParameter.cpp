/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSFileBrowserParameter.h>

#include <typeinfo>

covise::WSFileBrowserParameter *covise::WSFileBrowserParameter::prototype = new covise::WSFileBrowserParameter();

covise::WSFileBrowserParameter::WSFileBrowserParameter()
    : covise::WSParameter("", "", "FileBrowser")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__FileBrowserParameter).name(), this);
}

covise::WSFileBrowserParameter::WSFileBrowserParameter(const QString &name, const QString &description, const QString value)
    : covise::WSParameter(name, description, "FileBrowser")
{
    WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = value.toStdString();
}

covise::WSFileBrowserParameter::~WSFileBrowserParameter()
{
}

bool covise::WSFileBrowserParameter::setValue(const QString &inValue)
{
    bool changed = this->parameter.value != WSTools::fromCovise(inValue).toStdString();
    this->parameter.value = WSTools::fromCovise(inValue).toStdString();
    if (changed)
        emit parameterChanged(this);
    return changed;
}

const QString covise::WSFileBrowserParameter::getValue() const
{
    return QString::fromStdString(this->parameter.value);
}

QString covise::WSFileBrowserParameter::toString() const
{
    return getValue();
}

covise::WSParameter *covise::WSFileBrowserParameter::clone() const
{
    return new covise::WSFileBrowserParameter(*this);
}

const covise::covise__Parameter *covise::WSFileBrowserParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSFileBrowserParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__FileBrowserParameter *p = dynamic_cast<const covise::covise__FileBrowserParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSFileBrowserParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
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

QString covise::WSFileBrowserParameter::toCoviseString() const
{
    return WSTools::toCovise(getValue());
}
