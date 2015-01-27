/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__period.h"
#include <string.h>

/// read from file
coTetin__period::coTetin__period(istream &str, int binary)
    : coTetinCommand(coTetin::PERIODIC)
{
    if (binary)
    {
    }
    else
    {
        char lineBuf[4096], *bPtr = lineBuf;
        getLine(lineBuf, 4096, str);
        d_x = readFloat(bPtr);
        d_y = readFloat(bPtr);
        d_z = readFloat(bPtr);
        d_nx = readFloat(bPtr);
        d_ny = readFloat(bPtr);
        d_nz = readFloat(bPtr);
        d_angle = readFloat(bPtr);
    }
}

/// read from memory
coTetin__period::coTetin__period(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::PERIODIC)
{
    d_x = *floatDat++;
    d_y = *floatDat++;
    d_z = *floatDat++;
    d_nx = *floatDat++;
    d_ny = *floatDat++;
    d_nz = *floatDat++;
    d_angle = *floatDat++;
}

/// Destructor
coTetin__period::~coTetin__period()
{
}

/// check whether Object is valid
int coTetin__period::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__period::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command's name
    numInt++;

    // parameters
    numFloat += 6;
}

/// put my data to a given set of pointers
void coTetin__period::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *floatDat++ = d_x;
    *floatDat++ = d_y;
    *floatDat++ = d_z;
    *floatDat++ = d_nx;
    *floatDat++ = d_ny;
    *floatDat++ = d_nz;
    *floatDat++ = d_angle;
}

/// print to a stream in Tetin format
void coTetin__period::print(ostream &str) const
{
    if (isValid())
        str << "periodic "
            << d_x << " "
            << d_y << " "
            << d_z << " "
            << d_nx << " "
            << d_ny << " "
            << d_nz << " "
            << d_angle
            << endl;

    else
        str << "// invalid periodic command skipped" << endl;
}

// ===================== command-specific functions =====================
