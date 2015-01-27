/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FUEL_DRUCK_DATA_H_
#define __FUEL_DRUCK_DATA_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS FuellDruckData
//
// Initial version: 2002-03-26 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "CadmouldData.h"
#include <util/coviseCompat.h>
#include <string>

/**
 * This Class models a single "FuellDruck" dara set from a Cadmould
 * simulation ouput deck. This may either be a step from a time set
 * or a non-timedependent data set.
 *
 */
class FuellDruckData : public CadmouldData
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    FuellDruckData(const char *path, int numFields,
                   const char *const *fieldNames, bool bswap);

    /// Destructor : virtual in case we derive objects
    virtual ~FuellDruckData();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// get number of timesteps - 0 = stationary
    virtual int numTimeSteps();

    /// get number of data fields
    virtual int numFields();

    /// get number of vertices saved
    virtual int numVert();

    /// get name of data field
    virtual const char *getName(int fieldNo);

    /// get percentage of timestep, -1 if stationary
    virtual int percent(int stepNo);

    /** get a certain step of a specified field into a given buffer
       * @return =0 ok, errno otherwise
       */
    virtual int readField(int fieldNo, int stepNo,
                          void *buffer);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // names of my fields
    const char *const *d_fieldNames;

    // number of fields used
    int d_numFields;

    // number of vertices in this data set
    int d_numVert;

    // percentage (timestep)
    float d_percent;

    // the open file
    FILE *d_file;

private:
    std::string path_;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED
    FuellDruckData(const FuellDruckData &);

    /// Assignment operator: NOT IMPLEMENTED
    FuellDruckData &operator=(const FuellDruckData &);

    /// Default constructor: NOT IMPLEMENTED
    FuellDruckData();
};
#endif
