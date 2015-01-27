/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __MULTI_FUELL_DATA_H_
#define __MULTI_FUELL_DATA_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS MultiFuellData
//
// Initial version: 2002-03-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

/**
 * Multi-file Cadmould "Fuelldruck" data
 *
 */
#include "CadmouldData.h"

class FuellDruckData;
class MultiFuellData : public CadmouldData
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    MultiFuellData(const char *path, int numFields,
                   const char *const *fieldNames, bool bswap);

    /// Destructor : virtual in case we derive objects
    virtual ~MultiFuellData();

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
    virtual int readField(int fieldNo, int stepNo, void *buffer);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // names of my fields
    const char *const *d_fieldNames;

    // number of time steps
    int d_numSteps;

    // Fuelldata files: max. 99, really d_numSteps
    static const int MAXFILES = 100;
    FuellDruckData *d_dataFields[MAXFILES];

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED
    MultiFuellData(const MultiFuellData &);

    /// Assignment operator: NOT IMPLEMENTED
    MultiFuellData &operator=(const MultiFuellData &);

    /// Default constructor: NOT IMPLEMENTED
    MultiFuellData();
};
#endif
