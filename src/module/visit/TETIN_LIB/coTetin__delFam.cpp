/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__delFam.h"
#include <string.h>

/// read from file
coTetin__delFam::coTetin__delFam(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::DELETE_FAMILY)
{
    return;
}

/// read from memory
coTetin__delFam::coTetin__delFam(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::DELETE_FAMILY)
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
    type_delete = *intDat++;
}

coTetin__delFam::coTetin__delFam(int n_fam, char **names, int del_type)
    : coTetinCommand(coTetin::DELETE_FAMILY)
{
    n_families = n_fam;
    type_delete = del_type;
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
coTetin__delFam::~coTetin__delFam()
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
    type_delete = FAMILY_DEL_INVALID;
}

/// check whether Object is valid
int coTetin__delFam::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__delFam::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command name + n_families + type_delete
    numInt += 3;
    int i;
    for (i = 0; i < n_families; i++)
    {
        numChar += (family_names && family_names[i]) ? (strlen(family_names[i]) + 1) : 1;
    }
}

/// put my data to a given set of pointers
void coTetin__delFam::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
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
    *intDat++ = type_delete;
}

/// print to a stream in Tetin format
void coTetin__delFam::print(ostream &str) const
{
    return;
}

// ===================== command-specific functions =====================
