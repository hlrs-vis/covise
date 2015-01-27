/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +                                                                         +
// +  coIntSliderParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coIntSliderParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coIntSliderParam::s_type = "INTSLI";
coUifPara::Typeinfo coIntSliderParam::s_paraType = coUifPara::numericType("INTSLI");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coIntSliderParam::coIntSliderParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_min = 0;
    d_max = 255;
    d_value = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coIntSliderParam::~coIntSliderParam()
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coIntSliderParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coIntSliderParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coIntSliderParam::paramChange()
{
    return Covise::get_reply_int_slider(&d_min, &d_max, &d_value);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coIntSliderParam::initialize()
{
    // we must allocate this due to Covise Appl-Lib impelentation bugs
    d_defString = new char[128];
    sprintf(d_defString, "%ld %ld %ld", d_min, d_max, d_value);

    Covise::add_port(PARIN, d_name, "IntSlider", d_desc);
    Covise::set_port_default(d_name, d_defString);
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coIntSliderParam::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "IntSlider : Min = " << d_min
         << " Max = " << d_max
         << " Value = " << d_value
         << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coIntSliderParam::setValue(long min, long max, long value)
{
    // ensure correct ordering
    if (min > max) // swap if set incorrectly
    {
        int x = min;
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

int coIntSliderParam::setMin(long min) { return setValue(min, d_max, d_value); }
int coIntSliderParam::setMax(long max) { return setValue(d_min, max, d_value); }
int coIntSliderParam::setValue(long val) { return setValue(d_min, d_max, val); }

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value

void coIntSliderParam::getValue(long &min, long &max, long &value) const
{
    min = d_min;
    max = d_max;
    value = d_value;
}

long coIntSliderParam::getMin() const { return d_min; }
long coIntSliderParam::getMax() const { return d_max; }
long coIntSliderParam::getValue() const { return d_value; }

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coIntSliderParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coIntSliderParam::getValString() const
{
    static char valString[192];
    sprintf(valString, "%ld %ld %ld", d_min, d_max, d_value);
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coIntSliderParam::setValString(const char *str)
{
    size_t retval;
    retval = sscanf(str, "%ld %ld %ld", &d_min, &d_max, &d_value);
    if (retval != 3)
    {
        std::cerr << "coIntSliderParam::setValString: sscanf failed" << std::endl;
        return;
    }
}
