/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__prescPnt.h"
#include <string.h>

/// read from file
coTetin__prescPnt::coTetin__prescPnt(istream &str, int binary)
    : coTetinCommand(coTetin::PRESCRIBED_POINT)
    , d_name(NULL)
    , d_family(NULL)
{
    if (binary)
    {
    }
    else
    {
        char lineBuf[4096];
        getLine(lineBuf, 4096, str);
        char *bPtr = lineBuf;
        coord[0] = readFloat(bPtr);
        coord[1] = readFloat(bPtr);
        coord[2] = readFloat(bPtr);
        if (!getOption(lineBuf, "name", d_name))
        {
            d_name = new char[1];
            d_name[0] = '\0';
        }
        if (!getOption(lineBuf, "family", d_family))
        {
            d_family = new char[1];
            d_family[0] = '\0';
        }
    }
}

coTetin__prescPnt::coTetin__prescPnt(float point_x, float point_y,
                                     float point_z,
                                     char *pp_name)
    : coTetinCommand(coTetin::PRESCRIBED_POINT)
    , d_name(NULL)
    , d_family(NULL)
{
    char *name = 0;
    coord[0] = point_x;
    coord[1] = point_y;
    coord[2] = point_z;
    if (pp_name)
    {
        name = (char *)new char[strlen(pp_name) + 1];
        strcpy(name, pp_name);
    }
    else
    {
        name = (char *)new char[1];
        name[0] = '\0';
    }
    d_name = name;
    d_family = new char[1];
    d_family[0] = '\0';
}

/// read from memory
coTetin__prescPnt::coTetin__prescPnt(int *&intDat, float *&floatDat,
                                     char *&charDat)
    : coTetinCommand(coTetin::PRESCRIBED_POINT)
{
    coord[0] = *floatDat++;
    coord[1] = *floatDat++;
    coord[2] = *floatDat++;
    d_name = getString(charDat);
    d_family = getString(charDat);
}

/// Destructor
coTetin__prescPnt::~coTetin__prescPnt()
{
    delete[] d_name;
    delete[] d_family;
}

/// check whether Object is valid ///
int coTetin__prescPnt::isValid() const
{
    if (!d_comm)
        return 0;
    else
        return 1;
}

/// count size required in fields
void coTetin__prescPnt::addSizes(int &numInt, int &numFloat, int &numChar) const
{

    numInt++; // d_comm
    numFloat += 3;

    // parameters
    numChar += strlen(d_name) + strlen(d_family) + 2;
}

/// put my data to a given set of pointers
void coTetin__prescPnt::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *floatDat++ = coord[0];
    *floatDat++ = coord[1];
    *floatDat++ = coord[2];

    strcpy(charDat, d_name);
    charDat += strlen(d_name) + 1;
    strcpy(charDat, d_family);
    charDat += strlen(d_family) + 1;
}

/// print to a stream in Tetin format
void coTetin__prescPnt::print(ostream &str) const
{
    if (isValid())
    {
        str << "prescribed_point"; // the command name
        str << " " << coord[0]
            << " " << coord[1]
            << " " << coord[2];
        if (*d_name)
            str << " name " << d_name;
        if (*d_family)
            str << " family " << d_family;
        str << endl;
    }
    else
        str << "// invalid prescribed_point command skipped" << endl;
}

// ===================== command-specific functions =====================
