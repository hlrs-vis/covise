/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coColorParam Parameter handling class                                  +
// +                                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coColorParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coColorParam::s_type = "COLOR";
coUifPara::Typeinfo coColorParam::s_paraType = coUifPara::numericType("COLOR");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coColorParam::coColorParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_value[0] = .5;
    d_value[1] = .5;
    d_value[2] = .5;
    d_value[3] = 1.;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coColorParam::~coColorParam()
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coColorParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coColorParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coColorParam::paramChange()
{
    return Covise::get_reply_color(&d_value[0], &d_value[1], &d_value[2], &d_value[3]);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coColorParam::initialize()
{
    // we must allocate this due to Covise Appl-Lib impelentation bugs
    d_defString = new char[196];
    sprintf(d_defString, "%g %g %g %g", d_value[0], d_value[1], d_value[2], d_value[3]);

    Covise::add_port(PARIN, d_name, "Color", d_desc);
    Covise::set_port_default(d_name, d_defString);
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coColorParam::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "Color : r = " << d_value[0]
         << " g = " << d_value[1]
         << " b = " << d_value[2]
         << " a = " << d_value[3]
         << endl;
}

/// set values
int coColorParam::setValue(float r, float g, float b, float a)
{
    d_value[0] = r;
    d_value[1] = g;
    d_value[2] = b;
    d_value[3] = a;

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_color_param(d_name, d_value[0], d_value[1], d_value[2], d_value[3]);
    else
        return 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value
//
float coColorParam::getValue(int c) const
{
    return (c >= 0 && c < 4) ? d_value[c] : 0.;
}

int coColorParam::getValue(float &r, float &g, float &b, float &a) const
{
    r = d_value[0];
    g = d_value[1];
    b = d_value[2];
    a = d_value[3];
    return 4;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coColorParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coColorParam::getValString() const
{
    static char valString[192];
    sprintf(valString, "%f %f %f %f", d_value[0], d_value[1], d_value[2], d_value[3]);
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coColorParam::setValString(const char *str)
{
    size_t retval;
    retval = sscanf(str, "%f %f %f %f ", &d_value[0], &d_value[1], &d_value[2], &d_value[2]);
    if (retval != 4)
    {
        std::cerr << "coColorParam::setValString: sscanf failed" << std::endl;
        return;
    }
}
