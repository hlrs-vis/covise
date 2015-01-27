/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__trianFam.h"
#include <string.h>

/// read from file
coTetin__trianFam::coTetin__trianFam(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::TRIANGULATE_FAMILY)
{
    return;
}

/// read from memory
coTetin__trianFam::coTetin__trianFam(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::TRIANGULATE_FAMILY)
{
    n_families = *intDat++;
    family_names = 0;
    if (n_families > 0)
    {
        family_names = new char *[n_families];
        int i;
        for (i = 0; i < n_families; i++)
        {
            family_names[i] = getString(charDat);
        }
    }
}

coTetin__trianFam::coTetin__trianFam(int n_fam, char **names)
    : coTetinCommand(coTetin::TRIANGULATE_FAMILY)
{
    n_families = n_fam;
    family_names = 0;
    if (n_families > 0)
    {
        family_names = new char *[n_families];
        int i;
        for (i = 0; i < n_families; i++)
        {
            int len = (names && names[i]) ? (strlen(names[i]) + 1) : 1;
            family_names[i] = new char[len];
            if (names && names[i])
            {
                strcpy(family_names[i], names[i]);
            }
            else
            {
                family_names[i][0] = '\0';
            }
        }
    }
}

/// Destructor
coTetin__trianFam::~coTetin__trianFam()
{
    int i;
    if (family_names)
    {
        for (i = 0; i < n_families; i++)
        {
            if (family_names[i])
                delete[] family_names[i];
        }
        delete[] family_names;
    }
    family_names = 0;
    n_families = 0;
}

/// check whether Object is valid
int coTetin__trianFam::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__trianFam::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command name + n_families
    numInt += 2;
    int i;
    for (i = 0; i < n_families; i++)
    {
        numChar += (family_names && family_names[i]) ? (strlen(family_names[i]) + 1) : 1;
    }
}

/// put my data to a given set of pointers
void coTetin__trianFam::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *intDat++ = n_families;
    int i;
    for (i = 0; i < n_families; i++)
    {
        if (family_names && family_names[i])
        {
            strcpy(charDat, family_names[i]);
            charDat += strlen(family_names[i]) + 1;
        }
        else
        {
            *charDat++ = '\0';
        }
    }
}

/// print to a stream in Tetin format
void coTetin__trianFam::print(ostream &str) const
{
    return;
}

// ===================== command-specific functions =====================
