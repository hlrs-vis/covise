/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__defDensPoly.h"
#include <string.h>

/// read from file
coTetin__defDensPoly::coTetin__defDensPoly(istream &str, int binary)
    : coTetinCommand(coTetin::DEFINE_DENSITY_POLYGON)
{
    if (binary)
    {
    }
    else
    {
        char lineBuf[4096];
        getLine(lineBuf, 4096, str);
        d_npnts = 0;
        d_size = 0.0;
        getOption(lineBuf, "npnts", d_npnts);
        getOption(lineBuf, "size", d_size);
        d_name = NULL;
        if (getOption(lineBuf, "name", d_name) == 0)
        {
            d_name = new char[1];
            d_name[0] = '\0';
        }

        // no points ...
        if (d_npnts <= 0)
        {
            d_coord = NULL;
            return;
        }

        // read the points the safe way...
        float *fPtr;
        fPtr = d_coord = new float[3 * d_npnts];
        int i;
        for (i = 0; i < d_npnts; i++)
        {
            getLine(lineBuf, 4096, str);
            char *cPtr = lineBuf;
            *fPtr = readFloat(cPtr);
            fPtr++;
            *fPtr = readFloat(cPtr);
            fPtr++;
            *fPtr = readFloat(cPtr);
            fPtr++;
        }
    }
}

/// read from memory
coTetin__defDensPoly::coTetin__defDensPoly(int *&intDat,
                                           float *&floatDat,
                                           char *&charDat)
    : coTetinCommand(coTetin::DEFINE_DENSITY_POLYGON)
{
    d_npnts = *intDat++;
    d_size = *floatDat++;
    d_name = getString(charDat);
    if (d_npnts <= 0)
    {
        d_coord = NULL;
        return;
    }

    // read the points
    d_coord = new float[3 * d_npnts];
    memcpy((void *)d_coord, (void *)floatDat, 3 * d_npnts * sizeof(float));
    floatDat += 3 * d_npnts;
}

/// construct it
coTetin__defDensPoly::coTetin__defDensPoly(int npts, float size, float *coord, char *name)
{
    d_npnts = npts;
    d_size = size;
    if (name)
        d_name = strcpy(new char[1 + strlen(name)], name);

    // read the points
    d_coord = new float[3 * d_npnts];
    memcpy((void *)d_coord, (void *)coord, 3 * d_npnts * sizeof(float));
}

/// Destructor
coTetin__defDensPoly::~coTetin__defDensPoly()
{
    delete[] d_name;
    delete[] d_coord;
}

/// check whether Object is valid
int coTetin__defDensPoly::isValid() const
{
    if (d_coord && d_comm)
        return 1;
    else
        return 0;
}

/// count size required in fields
void coTetin__defDensPoly::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    numChar += strlen(d_name) + 1; //  d_name + \0
    numInt += 2; //  d_comm + d_npnts
    numFloat += 1 + 3 * d_npnts; //  d_size + points
}

/// put my data to a given set of pointers
void coTetin__defDensPoly::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the command's name
    *intDat++ = d_comm;

    // copy the data
    strcpy(charDat, d_name);
    charDat += strlen(d_name) + 1;

    *intDat++ = d_npnts;
    *floatDat++ = d_size;

    if (d_npnts)
        memcpy((void *)floatDat, (void *)d_coord, 3 * d_npnts * sizeof(float));
    floatDat += 3 * d_npnts;
}

/// print to a stream in Tetin format
void coTetin__defDensPoly::print(ostream &str) const
{
    if (isValid())
    {
        str << "define_density_poly npnts " << d_npnts << " size " << d_size;
        if (*d_name)
            str << " name " << d_name << endl;
        else
            str << endl;

        int i;
        for (i = 0; i < d_npnts; i++)
            str << d_coord[3 * i] << "," << d_coord[3 * i + 1] << "," << d_coord[3 * i + 2]
                << endl;
    }
    else
        str << "// invalid define_density_poly command skipped" << endl;
}

// ===================== command-specific functions =====================
