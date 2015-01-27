/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coStringParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coStringParam.h"
#include <appl/ApplInterface.h>
#include "coBlankConv.h"

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coStringParam::s_type = "STRING";
coUifPara::Typeinfo coStringParam::s_paraType = coUifPara::numericType("STRING");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coStringParam::coStringParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_value = NULL;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coStringParam::~coStringParam()
{
    delete[] d_value;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coStringParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coStringParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coStringParam::paramChange()
{
    const char *newVal;
    int retVal = Covise::get_reply_string(&newVal);
    if (retVal)
    {
        delete[] d_value;
        d_value = strcpy(new char[strlen(newVal) + 1], newVal);
    }
    return retVal;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coStringParam::initialize()
{
    // if no default is given, we give one
    if (!d_value)
        setValue("no default val");

    // we need a copy...
    d_defString = strcpy(new char[strlen(d_value) + 1], d_value);

    Covise::add_port(PARIN, d_name, "String", d_desc);
    Covise::set_port_default(d_name, d_defString);
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coStringParam::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "coStringParam : " << d_value << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coStringParam::setValue(const char *val)
{
    delete[] d_value;
    d_value = coBlankConv::all(val);

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_string_param(d_name, d_value);
    else
        return 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the converted value (\177 characters converted back to space)

const char *coStringParam::getValue() const
{
    char *c = d_value;
    while (*c)
    {
        if (*c == '\177')
            *c = ' ';
        c++;
    }
    c = d_value;
    if (c[0] == 1 && c[1] == 0)
        c[0] = 0;
    return d_value;
}

/// ----- Prevent auto-generated functions by assert -------
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coStringParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coStringParam::getValString() const
{
    return d_value;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coStringParam::setValString(const char *str)
{
    delete[] d_value;
    d_value = strcpy(new char[strlen(str) + 1], str);
}
