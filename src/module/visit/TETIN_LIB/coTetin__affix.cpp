/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__affix.h"
#include <string.h>

/// read from file
coTetin__affix::coTetin__affix(istream &str, int binary)
    : coTetinCommand(coTetin::AFFIX)
{
    if (binary)
    {
    }
    else
    {
        // (str>>d_flag) fails if eof or other error condition, e.g. nun-numeric
        if (!(str >> d_flag) || d_flag < 0 || d_flag > 1)
            d_flag = -1;
    }
}

/// read from memory
coTetin__affix::coTetin__affix(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::AFFIX)
{
    d_flag = *intDat;
    intDat++;
    if (d_flag < 0 || d_flag > 1)
        d_flag = -1;
}

/// Destructor
coTetin__affix::~coTetin__affix()
{
}

/// check whether Object is valid
int coTetin__affix::isValid() const
{
    if (!d_comm || d_flag == -1)
        return 0;
    else
        return 1;
}

/// count size required in fields
void coTetin__affix::addSizes(int &numInt, int &, int &) const
{
    // command name + data
    numInt += 2;
}

/// put my data to a given set of pointers
void coTetin__affix::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *intDat++ = d_flag;
}

/// print to a stream in Tetin format
void coTetin__affix::print(ostream &str) const
{
    if (isValid())
        str << "affix " << d_flag << endl;
    else
        str << "// invalid affix command skipped" << endl;
}

// ===================== command-specific functions =====================

coTetin__affix::coTetin__affix(int val)
    : coTetinCommand(coTetin::AFFIX)
{
    if (val == 0 || val == 1)
        d_flag = val;
    else
        d_flag = -1;
}

void coTetin__affix::setValue(int val)
{
    if (val == 0 || val == 1)
        d_flag = val;
    else
        d_flag = -1;
}
