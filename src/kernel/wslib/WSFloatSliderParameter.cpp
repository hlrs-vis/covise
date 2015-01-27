/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <WSFloatSliderParameter.h>

#include <typeinfo>

covise::WSFloatSliderParameter *covise::WSFloatSliderParameter::prototype = new covise::WSFloatSliderParameter();

covise::WSFloatSliderParameter::WSFloatSliderParameter()
    : covise::WSParameter("", "", "FloatSlider")
{
    covise::WSParameter::addPrototype(typeid(covise::covise__FloatSliderParameter).name(), this);
}

covise::WSFloatSliderParameter::WSFloatSliderParameter(const QString &name, const QString &description)
    : covise::WSParameter(name, description, "FloatSlider")
{
    WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = 0.0f;
    this->parameter.min = 0.0f;
    this->parameter.max = 1.0f;
}

covise::WSFloatSliderParameter::WSFloatSliderParameter(const QString &name, const QString &description, float value, float min, float max)
    : WSParameter(name, description, "FloatSlider")
{
    WSParameter::getSerialisable(&this->parameter);
    this->parameter.value = value;
    this->parameter.min = min;
    this->parameter.max = max;
}

covise::WSFloatSliderParameter::~WSFloatSliderParameter()
{
}

void covise::WSFloatSliderParameter::setMin(float inMin)
{
    bool changed = this->parameter.min != inMin;
    this->parameter.min = inMin;
    if (changed)
        emit parameterChanged(this);
}

float covise::WSFloatSliderParameter::getMin() const
{
    return this->parameter.min;
}

void covise::WSFloatSliderParameter::setMax(float inMax)
{
    bool changed = this->parameter.max != inMax;
    this->parameter.max = inMax;
    if (changed)
        emit parameterChanged(this);
}

float covise::WSFloatSliderParameter::getMax() const
{
    return this->parameter.max;
}

bool covise::WSFloatSliderParameter::setValue(float inValue)
{
    bool changed = this->parameter.value != inValue;
    this->parameter.value = inValue;
    if (changed)
        emit parameterChanged(this);
    return changed;
}

bool covise::WSFloatSliderParameter::setValue(float value, float minimum, float maximum)
{
    bool changed = this->parameter.value != value || this->parameter.min != minimum || this->parameter.max != maximum;

    this->parameter.value = value;
    this->parameter.min = minimum;
    this->parameter.max = maximum;

    if (changed)
        emit parameterChanged(this);

    return changed;
}

float covise::WSFloatSliderParameter::getValue() const
{
    return this->parameter.value;
}

QString covise::WSFloatSliderParameter::toString() const
{
    return QString::number(this->parameter.min) + " " + QString::number(this->parameter.max) + " " + QString::number(this->parameter.value);
}

covise::WSParameter *covise::WSFloatSliderParameter::clone() const
{
    return new covise::WSFloatSliderParameter(*this);
}

const covise::covise__Parameter *covise::WSFloatSliderParameter::getSerialisable()
{
    return WSParameter::getSerialisable(&this->parameter);
}

bool covise::WSFloatSliderParameter::setValueFromSerialisable(const covise::covise__Parameter *serialisable)
{
    const covise::covise__FloatSliderParameter *p = dynamic_cast<const covise::covise__FloatSliderParameter *>(serialisable);
    if (p == 0)
    {
        std::cerr << "WSFloatSliderParameter::setValueFromSerialisable err: wrong class called for parameter " << serialisable->name
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
