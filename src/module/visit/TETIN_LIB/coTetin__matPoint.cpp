/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__matPoint.h"
#include <string.h>

/// read from file
coTetin__matPoint::coTetin__matPoint(istream &str, int binary)
    : coTetinCommand(coTetin::MATERIAL_POINT)
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
            ;
        {
            d_family = new char[1];
            d_family[0] = '\0';
        }
    }
}

/// read from memory
coTetin__matPoint::coTetin__matPoint(int *&intDat, float *&floatDat,
                                     char *&charDat)
    : coTetinCommand(coTetin::MATERIAL_POINT)
{
    coord[0] = *floatDat++;
    coord[1] = *floatDat++;
    coord[2] = *floatDat++;
    d_name = getString(charDat);
    d_family = getString(charDat);
}

/// Destructor
coTetin__matPoint::~coTetin__matPoint()
{
    delete[] d_name;
    delete[] d_family;
}

/// check whether Object is valid ///
int coTetin__matPoint::isValid() const
{
    if (!d_comm)
        return 0;
    else
        return 1;
}

/// count size required in fields
void coTetin__matPoint::addSizes(int &numInt, int &numFloat, int &numChar) const
{

    numInt++; // d_comm
    numFloat += 3;

    // parameters
    numChar += strlen(d_name) + strlen(d_family) + 2;
}

/// put my data to a given set of pointers
void coTetin__matPoint::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
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
void coTetin__matPoint::print(ostream &str) const
{
    if (isValid())
    {
        str << "material_point"; // the command name
        if (*d_name)
            str << " name " << d_name;
        if (*d_family)
            str << " family " << d_family;
        str << " " << coord[0]
            << " " << coord[1]
            << " " << coord[2] << endl;
    }
    else
        str << "// invalid material_point command skipped" << endl;
}

// ===================== command-specific functions =====================
