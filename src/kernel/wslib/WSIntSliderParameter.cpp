/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSIntSliderParameter.h>

#include <typeinfo>

covise::WSIntSliderParameter *covise::WSIntSliderParameter::prototype = new covise::WSIntSliderParameter();

covise::WSIntSliderParameter::WSIntSliderParameter()
    : covise::WSParameter("", "", "IntSlider")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__IntSliderParameter).name(), this);
}

covise::WSIntSliderParameter::WSIntSliderParameter(const QString &name, const QString &description)
    : covise::WSParameter(name, description, "IntSlider")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = 0;
    this->parameter.min = 0;
    this->parameter.max = 100;
}

covise::WSIntSliderParameter::WSIntSliderParameter(const QString &name, const QString &description, int value, int min, int max)
    : covise::WSParameter(name, description, "IntSlider")
{
    covise::WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = value;
    this->parameter.min = min;
    this->parameter.max = max;
}

covise::WSIntSliderParameter::~WSIntSliderParameter()
{
}

void covise::WSIntSliderParameter::setMin(int inMin)
{
    bool changed = this->parameter.min != inMin;
    this->parameter.min = inMin;
    if (changed)
        emit parameterChanged(this);
}

int covise::WSIntSliderParameter::getMin() const
{
    return this->parameter.min;
}

void covise::WSIntSliderParameter::setMax(int inMax)
{
    bool changed = this->parameter.max != inMax;
    this->parameter.max = inMax;
    if (changed)
        emit parameterChanged(this);
}

int covise::WSIntSliderParameter::getMax() const
{
    return this->parameter.max;
}

bool covise::WSIntSliderParameter::setValue(int inValue)
{
    bool changed = this->parameter.value != inValue;
    this->parameter.value = inValue;
    if (changed)
        emit parameterChanged(this);
    return changed;
}

bool covise::WSIntSliderParameter::setValue(int value, int minimum, int maximum)
{
    bool changed = this->parameter.value != value || this->parameter.min != minimum || this->parameter.max != maximum;

    this->parameter.value = value;
    this->parameter.min = minimum;
    this->parameter.max = maximum;

    if (changed)
        emit parameterChanged(this);
    return changed;
}

int covise::WSIntSliderParameter::getValue() const
{
    return this->parameter.value;
}

QString covise::WSIntSliderParameter::toString() const
{
    return QString::number(this->parameter.min) + " " + QString::number(this->parameter.max) + " " + QString::number(this->parameter.value);
}

covise::WSParameter *covise::WSIntSliderParameter::clone() const
{
    return new covise::WSIntSliderParameter(*this);
}

const covise::covise__Parameter *covise::WSIntSliderParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSIntSliderParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__IntSliderParameter *p = dynamic_cast<const covise::covise__IntSliderParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSIntSliderParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
                  << " of type " << serialisable->type << std::endl;
        return false;
    }

    bool changed = !equals(&(this->parameter), p) || this->parameter.value != p->value || this->parameter.min != p->min || this->parameter.max != p->max;

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
