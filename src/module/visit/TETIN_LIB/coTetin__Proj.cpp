/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__Proj.h"
#include <string.h>

/// read from file
coTetin__Proj::coTetin__Proj(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::PROJECT_POINT)
{
    return;
}

/// read from memory
coTetin__Proj::coTetin__Proj(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::PROJECT_POINT)
{
    n_points = *intDat++;
    points = 0;
    if (n_points > 0)
    {
        points = new float[3 * n_points];
        memcpy((void *)points, (void *)floatDat, 3 * n_points * sizeof(float));
        floatDat += 3 * n_points;
    }
    int dir_valid = *intDat++;
    dir = 0;
    if (dir_valid)
    {
        dir = new float[3];
        memcpy((void *)dir, (void *)floatDat, 3 * sizeof(float));
        floatDat += 3;
    }
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

coTetin__Proj::coTetin__Proj(int n_pnts, float *pnts, float *direct,
                             int n_fam, const char **names)
    : coTetinCommand(coTetin::PROJECT_POINT)
{
    n_points = n_pnts;
    points = 0;
    if (n_points > 0)
    {
        points = new float[3 * n_points];
        memcpy((void *)points, (void *)pnts, 3 * n_points * sizeof(float));
    }
    dir = 0;
    if (direct)
    {
        dir = new float[3];
        memcpy((void *)dir, (void *)direct, 3 * sizeof(float));
    }

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
coTetin__Proj::~coTetin__Proj()
{
    int i;
    if (points)
    {
        delete[] points;
    }
    if (dir)
    {
        delete[] dir;
    }
    if (family_names)
    {
        for (i = 0; i < n_families; i++)
        {
            if (family_names[i])
                delete[] family_names[i];
        }
        delete[] family_names;
    }
    n_points = 0;
    points = 0;
    dir = 0;
    family_names = 0;
    n_families = 0;
}

/// check whether Object is valid
int coTetin__Proj::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__Proj::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command name + n_points + dir_valid + n_families
    numInt += 4;
    if (points)
    {
        numFloat += 3 * n_points;
    }
    if (dir)
    {
        numFloat += 3;
    }
    int i;
    for (i = 0; i < n_families; i++)
    {
        numChar += (family_names && family_names[i]) ? (strlen(family_names[i]) + 1) : 1;
    }
}

/// put my data to a given set of pointers
void coTetin__Proj::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *intDat++ = n_points;
    if (points)
    {
        memcpy((void *)floatDat, (void *)points, 3 * n_points * sizeof(float));
        floatDat += 3 * n_points;
    }
    if (dir)
    {
        *intDat++ = 1;
        memcpy((void *)floatDat, (void *)dir, 3 * sizeof(float));
        floatDat += 3;
    }
    else
    {
        *intDat++ = 0;
    }

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
void coTetin__Proj::print(ostream &str) const
{
    return;
}

// ===================== command-specific functions =====================
