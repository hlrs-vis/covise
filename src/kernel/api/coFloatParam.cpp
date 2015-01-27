/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coFloatParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coFloatParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coFloatParam::s_type = "FLOSCA";
coUifPara::Typeinfo coFloatParam::s_paraType
    = coUifPara::numericType("FLOSCA");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coFloatParam::coFloatParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_value = 0.0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coFloatParam::~coFloatParam()
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coFloatParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coFloatParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coFloatParam::paramChange()
{
    return Covise::get_reply_float_scalar(&d_value);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coFloatParam::initialize()
{
    // we must allocate this due to Covise Appl-Lib impelentation bugs
    d_defString = new char[64];
    sprintf(d_defString, "%f", d_value);

    Covise::add_port(PARIN, d_name, "FloatScalar", d_desc);
    Covise::set_port_default(d_name, d_defString);
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coFloatParam::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "FloatScalar: " << d_value << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coFloatParam::setValue(float val)
{
    d_value = val;

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_scalar_param(d_name, d_value);
    else
        return 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value: if called after init() : update on map

float coFloatParam::getValue() const
{
    return d_value;
}

/// ----- Prevent auto-generated functions by assert -------
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coFloatParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coFloatParam::getValString() const
{
    static char valString[64];
    sprintf(valString, "%f", d_value);
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coFloatParam::setValString(const char *str)
{
    size_t retval;
    retval = sscanf(str, "%f", &d_value);
    if (retval != 1)
    {
        std::cerr << "coFloatParam::setValString: sscanf failed" << std::endl;
        return;
    }
}
