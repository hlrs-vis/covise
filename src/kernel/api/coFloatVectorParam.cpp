/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coFloatVectorParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coFloatVectorParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coFloatVectorParam::s_type = "FLOVEC";
coUifPara::Typeinfo coFloatVectorParam::s_paraType = coUifPara::numericType("FLOVEC");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coFloatVectorParam::coFloatVectorParam(const char *name, const char *desc, int length)
    : coUifPara(name, desc)
{
    d_length = length;
    d_data = new float[d_length];
    int i;
    for (i = 0; i < length; i++)
        d_data[i] = 0.0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coFloatVectorParam::~coFloatVectorParam()
{
    delete[] d_data;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coFloatVectorParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coFloatVectorParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coFloatVectorParam::paramChange()
{
    int i;
    int res = 0;
    for (i = 0; i < d_length; i++)
        res |= Covise::get_reply_float_vector(i, d_data + i);
    return res;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coFloatVectorParam::initialize()
{
    // we must allocate this due to Covise Appl-Lib implementation bugs
    d_defString = new char[d_length * 32];
    d_defString[0] = '\0';

    char buffer[32];
    int i;
    for (i = 0; i < d_length; i++)
    {
        sprintf(buffer, "%g ", d_data[i]);
        strcat(d_defString, buffer);
    }

    Covise::add_port(PARIN, d_name, "FloatVector", d_desc);
    Covise::set_port_default(d_name, d_defString);
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coFloatVectorParam::print(ostream &str) const
{
    coUifPara::print(str);
    int i;
    str << "FloatVector:" << endl;
    for (i = 0; i < d_length; i++)
        str << " " << d_data[i];
    cerr << endl;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coFloatVectorParam::setValue(int pos, float data)
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
        Covise::sendWarning("coFloatVectorParam index out of range");
    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coFloatVectorParam::setValue(int size, const float *data)
{
    if (size != d_length)
    {
        delete[] d_data;
        d_data = new float[size];
        d_length = size;
    }
    memcpy(d_data, data, sizeof(float) * size);

    /// If we have been initialized, update the map
    if (d_init)
        return Covise::update_vector_param(d_name, d_length, d_data);
    else
        return 1;
}

/// set for 3d data
int coFloatVectorParam::setValue(float data0, float data1, float data2)
{
    if (d_length != 3)
    {
        delete[] d_data;
        d_data = new float[3];
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

float coFloatVectorParam::getValue(int pos) const
{
    if (pos >= 0 && pos < d_length)
        return d_data[pos];

    Covise::sendWarning("coFloatVectorParam index out of range");
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value

int coFloatVectorParam::getValue(float &data0, float &data1, float &data2) const
{
    if (d_length == 3)
    {
        data0 = d_data[0];
        data1 = d_data[1];
        data2 = d_data[2];
        return 1;
    }
    else
    {
        Covise::sendWarning("tried to access non-3D vector param as 3D");
        return 0;
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coFloatVectorParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coFloatVectorParam::getValString() const
{
    static char *valString = NULL;
    delete[] valString;
    valString = new char[d_length * 64 + 8];
    sprintf(valString, "%d", d_length);
    int i;
    for (i = 0; i < d_length; i++)
        sprintf(valString + strlen(valString), " %f", d_data[i]);
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coFloatVectorParam::setValString(const char *str)
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
        std::cerr << "coFloatVectorParam::setValString: sscanf failed" << std::endl;
        return;
    }

    if (newlength != d_length)
    {
        delete[] d_data;
        d_data = new float[d_length];
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
        retval = sscanf(str, "%f", &d_data[i]);
        if (retval != 1)
        {
            std::cerr << "coFloatVectorParam::setValString: sscanf failed" << std::endl;
            return;
        }

        // skip all non-blanks
        while (*str && !isspace(*str))
            str++;
    }
}
