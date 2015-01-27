/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__trianTol.h"
#include <string.h>

/// read from file
coTetin__trianTol::coTetin__trianTol(istream &str, int binary)
    : coTetinCommand(coTetin::SET_TRIANGULATION_TOLERANCE)
{
    if (binary)
    {
    }
    else
    {
        str >> d_val;
    }
}

/// read from memory
coTetin__trianTol::coTetin__trianTol(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::SET_TRIANGULATION_TOLERANCE)
{
    d_val = *floatDat;
    floatDat++;
}

/// Destructor
coTetin__trianTol::~coTetin__trianTol()
{
}

/// check whether Object is valid
int coTetin__trianTol::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__trianTol::addSizes(int &numInt, int &numFloat, int &) const
{
    // command name + data
    numInt++;
    numFloat++;
}

/// put my data to a given set of pointers
void coTetin__trianTol::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *floatDat++ = d_val;
}

/// print to a stream in Tetin format
void coTetin__trianTol::print(ostream &str) const
{
    if (isValid())
        str << "set_triangulation_tolerance " << d_val << endl;
    else
        str << "// invalid set_triangulation_tolerance command skipped" << endl;
}

// ===================== command-specific functions =====================
