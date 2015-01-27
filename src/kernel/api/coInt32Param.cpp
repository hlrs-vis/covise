/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coInt32Param Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coInt32Param.h"
#include <appl/ApplInterface.h>

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coInt32Param::s_type = "INTSCA";
coUifPara::Typeinfo coInt32Param::s_paraType
    = coUifPara::numericType("INTSCA");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coInt32Param::coInt32Param(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_value = 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coInt32Param::~coInt32Param()
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coInt32Param::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coInt32Param::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coInt32Param::paramChange()
{
    return Covise::get_reply_int_scalar(&d_value);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coInt32Param::initialize()
{
    // we must allocate this due to Covise Appl-Lib impelentation bugs
    d_defString = new char[64];
    sprintf(d_defString, "%ld", d_value);

    Covise::add_port(PARIN, d_name, "IntScalar", d_desc);
    Covise::set_port_default(d_name, d_defString);
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coInt32Param::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "Integer scalar : " << d_value << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coInt32Param::setValue(long val)
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

long coInt32Param::getValue() const
{
    return d_value;
}

/// ----- Prevent auto-generated functions by assert -------
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coInt32Param::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coInt32Param::getValString() const
{
    static char valString[64];
    sprintf(valString, "%ld", d_value);
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coInt32Param::setValString(const char *str)
{
    size_t retval;
    retval = sscanf(str, "%ld", &d_value);
    if (retval != 1)
    {
        std::cerr << "coInt32Param::setValString: sscanf failed" << std::endl;
        return;
    }
}

/// Copy-Constructor: NOT IMPLEMENTED
coInt32Param::coInt32Param(const coInt32Param &)
    : coUifPara()
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
coInt32Param &coInt32Param::operator=(const coInt32Param &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT  IMPLEMENTED
coInt32Param::coInt32Param()
{
    assert(0);
}
