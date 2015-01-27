/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coFloatSliderParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
#include <covise/covise.h>
#include "coFloatSliderParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coFloatSliderParam::s_type = "FLOSLI";
coUifPara::Typeinfo coFloatSliderParam::s_paraType = coUifPara::numericType("FLOSLI");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coFloatSliderParam::coFloatSliderParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_min = 0.0;
    d_max = 1.0;
    d_value = 0.5;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coFloatSliderParam::~coFloatSliderParam()
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coFloatSliderParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coFloatSliderParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coFloatSliderParam::paramChange()
{
    return Covise::get_reply_float_slider(&d_min, &d_max, &d_value);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coFloatSliderParam::initialize()
{
    // we must allocate this due to Covise Appl-Lib impelentation bugs
    d_defString = new char[196];
    sprintf(d_defString, "%g %g %g", d_min, d_max, d_value);
    if (!strchr(d_defString, '.'))
        strcat(d_defString, ".");

    Covise::add_port(PARIN, d_name, "FloatSlider", d_desc);
    Covise::set_port_default(d_name, d_defString);
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coFloatSliderParam::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "FloatSlider : Min = " << d_min
         << " Max = " << d_max
         << " Value = " << d_value
         << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coFloatSliderParam::setValue(float min, float max, float value)
{
    // ensure correct ordering
    if (min == max)
        max = min + 1;

    if (min > max)
    {
        float x = min;
        min = max;
        max = x;
    }
    if (value > max)
        max = value;
    if (value < min)
        min = value;

    d_min = min;
    d_max = max;
    d_value = value;

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_slider_param(d_name, d_min, d_max, d_value);
    else
        return 1;
}

int coFloatSliderParam::setMin(float min)
{
    return setValue(min, d_max, d_value);
}

int coFloatSliderParam::setMax(float max)
{
    return setValue(d_min, max, d_value);
}

int coFloatSliderParam::setValue(float value)
{
    return setValue(d_min, d_max, value);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value

void coFloatSliderParam::getValue(float &min, float &max, float &value) const
{
    min = d_min;
    max = d_max;
    value = d_value;
}

float coFloatSliderParam::getMin() const { return d_min; }
float coFloatSliderParam::getMax() const { return d_max; }
float coFloatSliderParam::getValue() const { return d_value; }

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coFloatSliderParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coFloatSliderParam::getValString() const
{
    static char valString[192];
    sprintf(valString, "%f %f %f", d_min, d_max, d_value);
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coFloatSliderParam::setValString(const char *str)
{
    size_t retval;
    retval = sscanf(str, "%f %f %f", &d_min, &d_max, &d_value);
    if (retval != 3)
    {
        std::cerr << "coFloatSliderParam::setValString: sscanf failed" << std::endl;
        return;
    }
}
