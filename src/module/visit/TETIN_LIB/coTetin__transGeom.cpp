/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__transGeom.h"
#include <string.h>

/// read from file
coTetin__transGeom::coTetin__transGeom(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::TRANSLATE_GEOM)
{
    return;
}

/// read from memory
coTetin__transGeom::coTetin__transGeom(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::TRANSLATE_GEOM)
{
    int i, j;
    type = *intDat++;
    trans_family = *intDat++;
    n_names = *intDat++;
    for (i = 0; i < 3; i++)
    {
        trans_vec[i] = *floatDat++;
    }
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            rot_matrix[i][j] = *floatDat++;
        }
    }
    names = 0;
    if (n_names > 0)
    {
        names = new char *[n_names];
        int i;
        for (i = 0; i < n_names; i++)
        {
            names[i] = getString(charDat);
        }
    }
}

coTetin__transGeom::coTetin__transGeom(int type_in,
                                       int trans_family_in,
                                       int n_names_in,
                                       const char **names_in,
                                       float trans_vec_in[3],
                                       float rot_matrix_in[3][3])
    : coTetinCommand(coTetin::TRANSLATE_GEOM)
{
    int i, j;

    type = type_in;
    trans_family = trans_family_in;
    for (i = 0; i < 3; i++)
    {
        trans_vec[i] = trans_vec_in[i];
    }
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            rot_matrix[i][j] = rot_matrix_in[i][j];
        }
    }
    n_names = n_names_in;
    names = 0;
    if (n_names > 0)
    {
        names = new char *[n_names];
        int i;
        for (i = 0; i < n_names; i++)
        {
            int len = (names_in && names_in[i]) ? (strlen(names_in[i]) + 1) : 1;
            names[i] = new char[len];
            if (names_in && names_in[i])
            {
                strcpy(names[i], names_in[i]);
            }
            else
            {
                names[i][0] = '\0';
            }
        }
    }
}

/// Destructor
coTetin__transGeom::~coTetin__transGeom()
{
    int i, j;
    if (names)
    {
        for (i = 0; i < n_names; i++)
        {
            if (names[i])
                delete[] names[i];
        }
        delete[] names;
    }
    type = ILLEGAL_TRANS;
    trans_family = 0;
    names = 0;
    n_names = 0;
    for (i = 0; i < 3; i++)
    {
        trans_vec[i] = 0.0;
        for (j = 0; j < 3; j++)
        {
            rot_matrix[i][j] = ((i == j) ? 1.0 : 0.0);
        }
    }
}

/// check whether Object is valid
int coTetin__transGeom::isValid() const
{
    if (d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__transGeom::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    // command name + n_names + type + trans_family
    numInt += 4;
    // 3 for trans_vec, 3*3 for rot_matrix
    numFloat += 12;
    int i;
    for (i = 0; i < n_names; i++)
    {
        numChar += (names && names[i]) ? (strlen(names[i]) + 1) : 1;
    }
}

/// put my data to a given set of pointers
void coTetin__transGeom::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    *intDat++ = type;
    *intDat++ = trans_family;
    *intDat++ = n_names;
    int i, j;
    for (i = 0; i < 3; i++)
    {
        *floatDat++ = trans_vec[i];
    }
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            *floatDat++ = rot_matrix[i][j];
        }
    }
    for (i = 0; i < n_names; i++)
    {
        if (names && names[i])
        {
            strcpy(charDat, names[i]);
            charDat += strlen(names[i]) + 1;
        }
        else
        {
            *charDat++ = '\0';
        }
    }
}

/// print to a stream in Tetin format
void coTetin__transGeom::print(ostream &str) const
{
    return;
}

// ===================== command-specific functions =====================
