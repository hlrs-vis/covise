/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS FillingData
//
// This class reads filling data
//
// Initial version: 2002-04-19 [sl]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "FillingData.h"
#include <util/coviseCompat.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef WIN32
#define DIR_SEP '\\'
#else
#define DIR_SEP '/'
#endif

// #define VERBOSE

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
FillingData::FillingData(const char *path,
                         int numFields, //int no_points,
                         const char *const *fieldNames,
                         const CadmouldData::FieldType *ftypes,
                         bool byteswap)
    : CadmouldData(byteswap)
    , d_fieldNames(fieldNames)
    , fieldTypes(ftypes)
    , d_numFields(numFields)
{
    d_state = 0;

    // check that file is there
    d_file = fopen(path, "r");
    if (!d_file)
    {
        if (errno)
            d_state = errno;
        else
            d_state = -1;
        return;
    }

    struct stat statRec;
    if (0 != fstat(fileno(d_file), &statRec))
    {
        if (errno)
            d_state = errno;
        else
            d_state = -1;
        return;
    }

    header hdr; // header of filling file

    // Read Header
    // @@@ assume binary
    if (fread(&hdr, sizeof(hdr), 1, d_file) != 1)
    {
        fprintf(stderr, "FillingData::FillingData: fread failed\n");
    }

    // to byte-swap or not to byte-swap, that is the question
    if (byte_swap)
    {
        byteSwap(1, &hdr.numnode);
    }
    /*
      if(hdr.maxfill < 0 || hdr.maxfill > 1000){
         byteSwap(1,&hdr.maxfill);
         if(hdr.maxfill < 0 || hdr.maxfill > 1000){
            Covise::sendError("FillingData: Even using byte-swapping a negative or greater than 1000 number of filling files is read.");
            d_state = -1;
            return;
         }
         byte_swap = true;
         byteSwap(1,&hdr.numnode);
      }
   */

    d_numVert = ((hdr.numnode + 3) / 4) * 4;

#ifdef VERBOSE
    fprintf(stderr, "opened %s : %d nodes\n", path, d_numVert);
#endif
}

CadmouldData::FieldType
FillingData::getFieldType(int fieldNo)
{
    return fieldTypes[fieldNo];
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

FillingData::~FillingData()
{
    fclose(d_file);
}

int
FillingData::numTimeSteps()
{
    return 0;
}

int
FillingData::numFields()
{
    return d_numFields;
}

int
FillingData::numVert()
{
    return d_numVert;
}

const char *
FillingData::getName(int fieldNo)
{
    return d_fieldNames[fieldNo];
}

int
FillingData::percent(int)
{
    return -1;
}

int
FillingData::readField(int fieldNo, int, void *buffer)
{
    rewind(d_file);
    // jump header
    header hdr;
    if (fread(&hdr, sizeof(hdr), 1, d_file) != 1)
    {
        fprintf(stderr, "FillingData::readField: fread failed\n");
    }

    switch (fieldNo)
    {
    case 0: // Fuellstand
    {
        float *value = static_cast<float *>(buffer);
        if (fread(value, sizeof(float), d_numVert, d_file) != d_numVert)
        {
            fprintf(stderr, "FillingData::readField: fread2 failed\n");
        }

        if (byte_swap)
            byteSwap(d_numVert, value);
    }
    break;
    case 1:
    {
        fseek(d_file, d_numVert * sizeof(float), SEEK_CUR);
        int *value = static_cast<int *>(buffer);
        if (fread(value, sizeof(int), d_numVert, d_file) != d_numVert)
        {
            fprintf(stderr, "FillingData::readField: fread3 failed\n");
        }
        if (byte_swap)
            byteSwap(d_numVert, value);
    }
    break;
    case 2:
    {
        fseek(d_file, d_numVert * (sizeof(float) + sizeof(int)), SEEK_CUR);
        header another_hdr;
        if (fread(&another_hdr, sizeof(another_hdr), 1, d_file) != 1)
        {
            fprintf(stderr, "FillingData::readField: fread4 failed\n");
        }
        float *value = static_cast<float *>(buffer);
        if (fread(value, sizeof(float), d_numVert, d_file) != d_numVert)
        {
            fprintf(stderr, "FillingData::readField: fread5 failed\n");
        }
        if (byte_swap)
            byteSwap(d_numVert, value);
    }
    break;
    }
    return 0;
}
