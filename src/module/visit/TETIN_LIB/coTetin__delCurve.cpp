/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__delCurve.h"
#include <string.h>

/// read from file
coTetin__delCurve::coTetin__delCurve(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::DELETE_CURVE)
{
    return;
}

/// read from memory
coTetin__delCurve::coTetin__delCurve(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::DELETE_CURVE)
{
    n_curves = *intDat++;
    curve_names = 0;
    if (n_curves > 0)
    {
        curve_names = new char *[n_curves];
        int i;
        for (i = 0; i < n_curves; i++)
        {
            curve_names[i] = getString(charDat);
        }
    }
}

coTetin__delCurve::coTetin__delCurve(int n_fam, char **names)
    : coTetinCommand(coTetin::DELETE_CURVE)
{
    n_curves = n_fam;
    curve_names = 0;
    if (n_curves > 0)
    {
        curve_names = new char *[n_curves];
        int i;
        for (i = 0; i < n_curves; i++)
        {
            int len = (names && names[i]) ? (strlen(names[i]) + 1) : 1;
            curve_names[i] = new char[len];
            if (names && names[i])
            {
                strcpy(curve_names[i], names[i]);
            }
            else
            {
                curve_names[i][0] = '\0';
            }
        }
    }
}

/// Destructor
coTetin__delCurve::~coTetin__delCurve()
{
    int i;
    if (curve_names)
    {
        for (i = 0; i < n_curves; i++)
        {
            if (curve_names[i])
                delete[] curve_names[i];
        }
        delete[] curve_names;
    }
    curve_names = 0;
    n_curves = 0;
}

/// check whether Object is valid
int coTetin__delCurve::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__delCurve::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command name + n_curves
    numInt += 2;
    int i;
    for (i = 0; i < n_curves; i++)
    {
        numChar += (curve_names && curve_names[i]) ? (strlen(curve_names[i]) + 1) : 1;
    }
}

/// put my data to a given set of pointers
void coTetin__delCurve::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *intDat++ = n_curves;
    int i;
    for (i = 0; i < n_curves; i++)
    {
        if (curve_names && curve_names[i])
        {
            strcpy(charDat, curve_names[i]);
            charDat += strlen(curve_names[i]) + 1;
        }
        else
        {
            *charDat++ = '\0';
        }
    }
}

/// print to a stream in Tetin format
void coTetin__delCurve::print(ostream &str) const
{
    return;
}

// ===================== command-specific functions =====================
