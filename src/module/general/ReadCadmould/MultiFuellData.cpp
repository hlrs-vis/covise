/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS MultiFuellData
//
// This class @@@
//
// Initial version: 2002-03-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "MultiFuellData.h"
#include "FuellDruckData.h"
#include <util/coviseCompat.h>
#include <sys/param.h>
#include <sys/stat.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MultiFuellData::MultiFuellData(const char *path, int numFields,
                               const char *const *fieldNames, bool bswap)
    : CadmouldData(bswap)
{
    // find all existing files
    char filename[MAXPATHLEN];

    // @@@@@@@@@@@@@@@   look for filenames - later
    d_numSteps = 0;

    /// try to open all files
    int i;
    for (i = 0; i < MAXFILES; i++)
    {
        d_dataFields[i] = NULL;
    }
    for (i = 1; i <= MAXFILES; i++)
    {
        if (i < MAXFILES)
            sprintf(filename, "%s%02d", path, i);
        else
            sprintf(filename, "%s00", path);

        // try to open file if existent
        struct stat dummy;
        if (0 == stat(filename, &dummy))
        {
            d_dataFields[d_numSteps]
                = new FuellDruckData(filename, numFields, fieldNames, bswap);
            if (d_dataFields[d_numSteps]
                && d_dataFields[d_numSteps]->getState() == 0)
            {
                //fprintf(stderr,"   -> found data\n");
                d_numSteps++;
            }
            else
            {
                delete d_dataFields[d_numSteps];
                d_dataFields[d_numSteps] = NULL;
            }
        }
    }

    if (d_numSteps > 0)
        d_state = 0;
    else
        d_state = -1;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MultiFuellData::~MultiFuellData()
{
    int i;
    for (i = 0; i < MAXFILES; ++i)
    {
        delete d_dataFields[i];
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// get number of timesteps - 0 = stationary
int MultiFuellData::numTimeSteps()
{
    return d_numSteps;
}

/// get number of data fields
int MultiFuellData::numVert()
{
    return d_dataFields[0]->numVert();
}

/// get number of data fields
int MultiFuellData::numFields()
{
    return d_dataFields[0]->numFields();
}

/// get name of data field
const char *MultiFuellData::getName(int fieldNo)
{
    return d_dataFields[0]->getName(fieldNo);
}

/// get percentage of timestep, -1 if stationary
int MultiFuellData::percent(int stepNo)
{
    return d_dataFields[stepNo]->percent(stepNo);
}

int MultiFuellData::readField(int fieldNo, int stepNo,
                              void *buffer)
{
    return d_dataFields[stepNo]->readField(fieldNo, stepNo, buffer);
}
