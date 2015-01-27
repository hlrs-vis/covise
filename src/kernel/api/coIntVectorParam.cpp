/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coIntVectorParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coIntVectorParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coIntVectorParam::s_type = "INTVEC";
coUifPara::Typeinfo coIntVectorParam::s_paraType = coUifPara::numericType("INTVEC");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coIntVectorParam::coIntVectorParam(const char *name, const char *desc, int length)
    : coUifPara(name, desc)
{
    d_length = length;
    d_data = new long[d_length];
    int i;
    for (i = 0; i < length; i++)
        d_data[i] = 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coIntVectorParam::~coIntVectorParam()
{
    delete[] d_data;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coIntVectorParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coIntVectorParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coIntVectorParam::paramChange()
{
    int i;
    int res = 0;
    for (i = 0; i < d_length; i++)
        res |= Covise::get_reply_int_vector(i, d_data + i);
    return res;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coIntVectorParam::initialize()
{
    // we must allocate this due to Covise Appl-Lib impelentation bugs
    d_defString = new char[d_length * 32];
    d_defString[0] = '\0';

    char buffer[32];
    int i;
    for (i = 0; i < d_length; i++)
    {
        sprintf(buffer, "%ld ", d_data[i]);
        strcat(d_defString, buffer);
    }

    Covise::add_port(PARIN, d_name, "IntVector", d_desc);
    Covise::set_port_default(d_name, d_defString);
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coIntVectorParam::print(ostream &str) const
{
    coUifPara::print(str);
    int i;
    str << "IntVector:" << endl;
    for (i = 0; i < d_length; i++)
        str << " " << d_data[i];
    cerr << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coIntVectorParam::setValue(int pos, long data)
{
    if (pos >= 0 && pos < d_length)
    {
        d_data[pos] = data;
        /// If we have been initialized, update the map
        if (d_init)
            return Covise::update_vector_param(d_name, d_length, d_data);
        else
            return 1;
    }
    else
        Covise::sendWarning("coIntVectorParam index out of range");
    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coIntVectorParam::setValue(int size, const long *data)
{
    if (size != d_length)
    {
        delete[] d_data;
        d_data = new long[size];
        d_length = size;
    }
    memcpy(d_data, data, sizeof(*data) * size);

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_vector_param(d_name, d_length, d_data);
    else
        return 1;
}

/// set for 3d data
int coIntVectorParam::setValue(long data0, long data1, long data2)
{
    if (d_length != 3)
    {
        delete[] d_data;
        d_data = new long[3];
        d_length = 3;
    }
    d_data[0] = data0;
    d_data[1] = data1;
    d_data[2] = data2;

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_vector_param(d_name, d_length, d_data);
    else
        return 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value

long coIntVectorParam::getValue(int pos) const
{
    if (pos >= 0 && pos < d_length)
        return d_data[pos];

    Covise::sendWarning("coIntVectorParam index out of range");
    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value 3d special

int coIntVectorParam::getValue(long &data0, long &data1, long &data2) const
{
    if (d_length != 3)
    {
        Covise::sendWarning("Called getValue(int&,int&,int&) on non-3D param");
        return -1;
    }
    data0 = d_data[0];
    data1 = d_data[1];
    data2 = d_data[2];
    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coIntVectorParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coIntVectorParam::getValString() const
{
    static char *valString = NULL;
    delete[] valString;
    valString = new char[d_length * 64 + 8];
    sprintf(valString, "%d", d_length);
    int i;
    for (i = 0; i < d_length; i++)
        sprintf(valString + strlen(valString), " %ld", d_data[i]);
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coIntVectorParam::setValString(const char *str)
{
    if (!str || !(*str))
        return;

    // skip leading blanks
    while (*str && isspace(*str))
        str++;

    // get first number: number of elements
    int newlength;
    size_t retval;
    retval = sscanf(str, "%d", &newlength);
    if (retval != 1)
    {
        std::cerr << "coIntVectorParam::setValString: sscanf failed" << std::endl;
        return;
    }

    if (newlength != d_length)
    {
        delete[] d_data;
        d_data = new long[d_length];
        d_length = newlength;
    }

    // skip to next blank or end
    while (*str && !isspace(*str))
        str++;

    int i;
    for (i = 0; i < d_length; i++)
    {
        // skip leading blanks
        while (*str && isspace(*str))
            str++;

        // read one number
        retval = sscanf(str, "%ld", &d_data[i]);
        if (retval != 1)
        {
            std::cerr << "coIntVectorParam::setValString: sscanf failed" << std::endl;
            return;
        }

        // skip all non-blanks
        while (*str && !isspace(*str))
            str++;
    }
}
