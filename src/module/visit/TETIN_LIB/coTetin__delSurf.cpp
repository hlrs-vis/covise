/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__delSurf.h"
#include <string.h>

/// read from file
coTetin__delSurf::coTetin__delSurf(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::DELETE_SURFACE)
{
    return;
}

/// read from memory
coTetin__delSurf::coTetin__delSurf(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::DELETE_SURFACE)
{
    n_surfaces = *intDat++;
    surface_names = 0;
    if (n_surfaces > 0)
    {
        surface_names = new char *[n_surfaces];
        int i;
        for (i = 0; i < n_surfaces; i++)
        {
            surface_names[i] = getString(charDat);
        }
    }
}

coTetin__delSurf::coTetin__delSurf(int n_fam, char **names)
    : coTetinCommand(coTetin::DELETE_SURFACE)
{
    n_surfaces = n_fam;
    surface_names = 0;
    if (n_surfaces > 0)
    {
        surface_names = new char *[n_surfaces];
        int i;
        for (i = 0; i < n_surfaces; i++)
        {
            int len = (names && names[i]) ? (strlen(names[i]) + 1) : 1;
            surface_names[i] = new char[len];
            if (names && names[i])
            {
                strcpy(surface_names[i], names[i]);
            }
            else
            {
                surface_names[i][0] = '\0';
            }
        }
    }
}

/// Destructor
coTetin__delSurf::~coTetin__delSurf()
{
    int i;
    if (surface_names)
    {
        for (i = 0; i < n_surfaces; i++)
        {
            if (surface_names[i])
                delete[] surface_names[i];
        }
        delete[] surface_names;
    }
    surface_names = 0;
    n_surfaces = 0;
}

/// check whether Object is valid
int coTetin__delSurf::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__delSurf::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command name + n_surfaces
    numInt += 2;
    int i;
    for (i = 0; i < n_surfaces; i++)
    {
        numChar += (surface_names && surface_names[i]) ? (strlen(surface_names[i]) + 1) : 1;
    }
}

/// put my data to a given set of pointers
void coTetin__delSurf::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *intDat++ = n_surfaces;
    int i;
    for (i = 0; i < n_surfaces; i++)
    {
        if (surface_names && surface_names[i])
        {
            strcpy(charDat, surface_names[i]);
            charDat += strlen(surface_names[i]) + 1;
        }
        else
        {
            *charDat++ = '\0';
        }
    }
}

/// print to a stream in Tetin format
void coTetin__delSurf::print(ostream &str) const
{
    return;
}

// ===================== command-specific functions =====================
