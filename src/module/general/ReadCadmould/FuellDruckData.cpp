/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS FuellDruckData
//
// This class @@@
//
// Initial version: 2002-03-26 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "FuellDruckData.h"
#include <util/coviseCompat.h>
#include <sys/stat.h>

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

FuellDruckData::FuellDruckData(const char *path, int numFields,
                               const char *const *fieldNames,
                               bool bswap)
    : CadmouldData(bswap)
    , d_fieldNames(fieldNames)
    , d_numFields(numFields)
    , path_(path)
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
        fclose(d_file);
        d_file = NULL;
        if (errno)
            d_state = errno;
        else
            d_state = -1;
        return;
    }
    fclose(d_file);
    d_file = NULL;

    // if we have a percentage, it is in the last two digits
    const char *number = path + strlen(path) - 2;
    if (number[0] >= '0' && number[0] <= '9'
        && number[1] >= '0' && number[1] <= '9')
    {
        d_percent = 0.1 * (number[0] - '0')
                    + 0.01 * (number[1] - '0');
    }
    else
        d_percent = 1.0;

    // length of file gives us # of nodes
    d_numVert = statRec.st_size / (6 * sizeof(float)) - 2;
#ifdef VERBOSE
    fprintf(stderr, "opened %s : %d nodes\n", path, d_numVert);
#endif
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

FuellDruckData::~FuellDruckData()
{
    if (d_file)
    {
        fclose(d_file);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// get number of timesteps - 0 = stationary
int FuellDruckData::numTimeSteps()
{
    return 0;
}

/// get number of data fields
int FuellDruckData::numVert()
{
    return d_numVert;
}

/// get number of data fields
int FuellDruckData::numFields()
{
    return d_numFields;
}

/// get name of data field
const char *FuellDruckData::getName(int fieldNo)
{
    return d_fieldNames[fieldNo];
}

/// get percentage of timestep, -1 if stationary
int FuellDruckData::percent(int stepNo)
{
    (void)stepNo;
    return int(d_percent);
}

/** get a certain step of a specified field into a given buffer
 * @return =0 ok, errno otherwise, stepNo is ignored
 */
int FuellDruckData::readField(int fieldNo, int stepNo, void *buffer)
{
    (void)stepNo;
    // this implementation wastes some storage, but is faster.
    // Cadmould simulations are vers small, so it's ok.
    float *filebuffer = new float[6 * d_numVert];

    // position behind the min/max fields (2x6)
    d_file = fopen(path_.c_str(), "r");
    if (!d_file
        || fseek(d_file, 12 * sizeof(float), SEEK_SET) != 0)
    {
        d_state = errno;
        return -1;
    }

    // read the complete file
    if (fread(filebuffer, sizeof(float), 6 * d_numVert, d_file) != 6 * d_numVert)
    {
        d_state = errno;
        fclose(d_file);
        d_file = NULL;
        return -1;
    }
    fclose(d_file);
    d_file = NULL;

    // now copy the correct field
    float *fPtr = filebuffer + fieldNo;
    float *bPtr = static_cast<float *>(buffer);
    int i;
    for (i = 0; i < d_numVert; i++)
    {
        *bPtr = *fPtr;
        ++bPtr;
        fPtr += 6;
    }

    // byteswap if we have to
    if (byte_swap)
    {
        byteSwap(d_numVert, buffer);
    }

    // scratch out parts
    //for (i=0;i<d_numVert;i++) {
    //   if (buffer[i]==-1.0)
    //      buffer[i]=FLT_MAX;
    //}

    return 0;
}
