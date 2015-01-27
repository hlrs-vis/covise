/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__rdFamBoco.h"
#include <string.h>

/// read from file
coTetin__rdFamBoco::coTetin__rdFamBoco(istream &str, int binary)
    : coTetinCommand(coTetin::READ_FAMILY_BOCO)
{
    if (binary)
    {
    }
    else
    {
        char buffer[512];
        getLine(buffer, 512, str);
        buffer[511] = '\0'; /// make sure it terminates
        d_boco = new char[strlen(buffer) + 1];
        strcpy(d_boco, buffer);
    }
}

/// read from memory
coTetin__rdFamBoco::coTetin__rdFamBoco(int *&, float *&, char *&charDat)
    : coTetinCommand(coTetin::READ_FAMILY_BOCO)
{
    d_boco = getString(charDat);
}

/// Destructor
coTetin__rdFamBoco::~coTetin__rdFamBoco()
{
    delete d_boco;
}

/// check whether Object is valid
int coTetin__rdFamBoco::isValid() const
{
    if (d_comm && d_boco && *d_boco)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__rdFamBoco::addSizes(int &numInt, int &, int &numChar) const
{
    // command name
    numInt++;

    numChar += strlen(d_boco) + 1;
}

/// put my data to a given set of pointers
void coTetin__rdFamBoco::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    strcpy(charDat, d_boco);
    charDat += strlen(d_boco) + 1;
}

/// print to a stream in Tetin format
void coTetin__rdFamBoco::print(ostream &str) const
{
    if (isValid())
        str << "read_family_boco " << d_boco << endl;
    else
        str << "// invalid read_family_boco command skipped" << endl;
}

// ===================== command-specific functions =====================
