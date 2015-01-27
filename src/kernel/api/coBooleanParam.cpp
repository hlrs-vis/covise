/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coBooleanParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coBooleanParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coBooleanParam::s_type = "BOOL";
coUifPara::Typeinfo coBooleanParam::s_paraType
    = coUifPara::numericType("BOOL");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor: needs coModule pointer to link in list

coBooleanParam::coBooleanParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_value = 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coBooleanParam::~coBooleanParam()
{
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coBooleanParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coBooleanParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coBooleanParam::paramChange()
{
    return Covise::get_reply_boolean(&d_value);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coBooleanParam::initialize()
{
    Covise::add_port(PARIN, d_name, "Boolean", d_desc);

    // we must allocate this due to Covise Appl-Lib impelentation bugs
    d_defString = new char[6];
    if (d_value)
        strcpy(d_defString, "TRUE");
    else
        strcpy(d_defString, "FALSE");

    Covise::set_port_default(d_name, d_defString);

    d_init = 1;
    return;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coBooleanParam::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "Boolean : " << ((d_value) ? "TRUE" : "FALSE") << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coBooleanParam::setValue(bool value)
{
    d_value = value;

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_boolean_param(d_name, d_value);
    else
        return 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value: if called after init() : update on map

bool coBooleanParam::getValue() const
{
    return (d_value != 0);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coBooleanParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coBooleanParam::getValString() const
{
    static const char *trueStr = "TRUE", *falseStr = "FALSE";
    if (d_value)
        return trueStr;
    else
        return falseStr;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coBooleanParam::setValString(const char *str)
{
    d_value = (str && (*str == 't' || *str == 'T' || *str == '1'));
}
