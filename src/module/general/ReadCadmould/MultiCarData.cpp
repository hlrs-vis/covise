/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS MultiCarData
//
// Handle multiple timesteps of one result group
//
// Initial version: 2002-05-10 [sk]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include <string>
#include "MultiCarData.h"
#include "CarData.h"
#include <util/coviseCompat.h>
#ifndef WIN32
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/dir.h>
#endif

#ifdef __linux__
#include <dirent.h>
#endif

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MultiCarData::MultiCarData(CarFiles &list,
                           int grp,
                           bool bswap)
    : CadmouldData(bswap)
{
    d_numSteps = 0;

    /// try to open all files
    int i;
    for (i = 0; i < MAXFILES; i++)
    {
        d_dataFields[i] = NULL;
        d_names[i] = NULL;
    }

    if (list.numFiles(grp) > MAXFILES)
    {
        cerr << "Number of steps > MAXFILES " << endl;
        return;
    }

    for (i = 0; i < list.numFiles(grp); i++)
    {

        d_dataFields[i] = new CarData(list.get(i, grp), bswap);

        if (d_dataFields[i]
            && d_dataFields[i]->getState() != 0)
        {
            delete d_dataFields[i];
            d_dataFields[i] = NULL;
        }
        else
        {
            d_numSteps++;
        }
    }

    if (d_numSteps > 0)
    {
        d_state = 0;

        // get name of fields
        const char *grp_name = list.getName(grp);
        int grp_len = strlen(grp_name);

        for (i = 0; i < d_dataFields[0]->numFields(); i++)
        {
            d_names[i] = new char[grp_len + strlen(d_dataFields[0]->getName(i)) + 2];
            sprintf(d_names[i], "%s:%s", grp_name, d_dataFields[0]->getName(i));
        }
    }
    else
    {
        d_state = -1;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MultiCarData::~MultiCarData()
{
    int i;
    for (i = 0; i < MAXFILES; ++i)
    {
        delete d_dataFields[i];
        delete[] d_names[i];
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// get number of timesteps - 0 = stationary
int MultiCarData::numTimeSteps()
{
    if (d_numSteps == 1)
    {
        return 0;
    }
    else
    {
        return d_numSteps;
    }
}

/// get number of data fields
int MultiCarData::numVert()
{
    return d_dataFields[0]->numVert();
}

/// get number of data fields
int MultiCarData::numFields()
{
    return d_dataFields[0]->numFields();
}

/// get name of data field
const char *MultiCarData::getName(int fieldNo)
{
    return d_names[fieldNo];
}

/// get percentage of timestep, -1 if stationary
int MultiCarData::percent(int stepNo)
{
    return d_dataFields[stepNo]->percent(stepNo);
}

/// get type of field
CadmouldData::FieldType
MultiCarData::getFieldType(int fieldNo)
{
    return d_dataFields[0]->getFieldType(fieldNo);
}

int MultiCarData::readField(int fieldNo, int stepNo,
                            void *buffer)
{
    return d_dataFields[stepNo]->readField(fieldNo, stepNo, buffer);
}
