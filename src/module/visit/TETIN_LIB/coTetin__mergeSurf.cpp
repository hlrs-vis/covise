/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__mergeSurf.h"
#include <string.h>

/// read from file
coTetin__mergeSurf::coTetin__mergeSurf(istream &str, int binary)
    : coTetinCommand(coTetin::MERGE_SURFACE)
{
    if (binary)
    {
    }
    else
    {
        char buffer[128];

        if (str >> buffer)
            d_fam[0] = strcpy(new char[strlen(buffer) + 1], buffer);
        else
            d_fam[0] = NULL;

        if (str >> buffer)
            d_fam[1] = strcpy(new char[strlen(buffer) + 1], buffer);
        else
        {
            delete d_fam[0];
            d_fam[1] = d_fam[0] = NULL;
        }
    }
}

/// read from memory
coTetin__mergeSurf::coTetin__mergeSurf(int *&, float *&, char *&chPtr)
    : coTetinCommand(coTetin::MERGE_SURFACE)
{
    if (chPtr)
    {
        d_fam[0] = getString(chPtr); // read 1st string and advance
        d_fam[1] = getString(chPtr); // read 2nd string and advance
    }
    else
        d_fam[0] = d_fam[1] = NULL;
}

/// Destructor
coTetin__mergeSurf::~coTetin__mergeSurf()
{
}

/// check whether Object is valid
int coTetin__mergeSurf::isValid() const
{
    if (!d_comm || !d_fam[0]) // no name or d_fam[0]=NULL
        return 0;
    else
        return 1;
}

/// count size required in fields
void coTetin__mergeSurf::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command's name
    numInt++;

    // parameters
    numChar += strlen(d_fam[0]) + strlen(d_fam[1]) + 2;
}

/// put my data to a given set of pointers
void coTetin__mergeSurf::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    strcpy(charDat, d_fam[0]);
    charDat += strlen(d_fam[0]) + 1;

    strcpy(charDat, d_fam[1]);
    charDat += strlen(d_fam[1]) + 1;
}

/// print to a stream in Tetin format
void coTetin__mergeSurf::print(ostream &str) const
{
    if (isValid())
        str << "merge_surface " // the command name
            << d_fam[0] << " " << d_fam[1]
            << endl;
    else
        str << "// invalid merge_surface command skipped" << endl;
}

// ===================== command-specific functions =====================

coTetin__mergeSurf::coTetin__mergeSurf(const char *fam0,
                                       const char *fam1)
{
    if (fam0 && fam1)
    {
        d_fam[0] = strcpy(new char[strlen(fam0) + 1], fam0);
        d_fam[1] = strcpy(new char[strlen(fam1) + 1], fam1);
    }
    else
        d_fam[0] = d_fam[1] = NULL;
}

const char *coTetin__mergeSurf::getFamily(int num) const
{
    if (num = 0 || num == 1)
        return d_fam[num];
    else
        return NULL;
}

void coTetin__mergeSurf::setFamily(int num, const char *name)
{
    if (num = 0 || num == 1)
    {
        delete d_fam[num];
        d_fam[num] = strcpy(new char[strlen(name) + 1], name);
    }
}
