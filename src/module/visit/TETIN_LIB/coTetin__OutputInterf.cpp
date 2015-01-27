/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__OutputInterf.h"
#include <string.h>

/// read from file
coTetin__OutputInterf::coTetin__OutputInterf(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::OUTPUT_INTERF)
{
    ;
}

/// read from memory
coTetin__OutputInterf::coTetin__OutputInterf(int *&intDat, float *&floatDat,
                                             char *&charDat)
    : coTetinCommand(coTetin::OUTPUT_INTERF)
{
    outp_intf = getString(charDat);
}

coTetin__OutputInterf::coTetin__OutputInterf(
    char *outp_intff)
    : coTetinCommand(coTetin::OUTPUT_INTERF)
{
    int len = (outp_intff) ? (strlen(outp_intff) + 1) : 1;
    outp_intf = new char[len];
    if (outp_intff)
    {
        strcpy(outp_intf, outp_intff);
    }
    else
    {
        outp_intf[0] = '\0';
    }
}

/// Destructor
coTetin__OutputInterf::~coTetin__OutputInterf()
{
    if (outp_intf)
        delete[] outp_intf;
    outp_intf = 0;
}

/// check whether Object is valid
int coTetin__OutputInterf::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__OutputInterf::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command name, + data
    numInt++;
    numChar += (outp_intf) ? (strlen(outp_intf) + 1) : 1;
}

/// put my data to a given set of pointers
void coTetin__OutputInterf::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    if (outp_intf)
    {
        strcpy(charDat, outp_intf);
        charDat += strlen(outp_intf) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
}

/// print to a stream in Tetin format
void coTetin__OutputInterf::print(ostream &str) const
{
    ;
}

// ===================== command-specific functions =====================
