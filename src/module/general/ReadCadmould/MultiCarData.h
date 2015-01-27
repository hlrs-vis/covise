/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __MULTI_CAR_DATA_H_
#define __MULTI_CAR_DATA_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS MultiCarData
//
// Initial version: 2002-05-10 [sk]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

/**
 * Multi-file Cadmould "Car" data
 *
 */
#include "CarData.h"
#include "CarFiles.h"

class MultiCarData : public CadmouldData
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    MultiCarData(CarFiles &list, int grp, bool bswap);

    /// Destructor : virtual in case we derive objects
    virtual ~MultiCarData();

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

    /// get type of field
    virtual CadmouldData::FieldType getFieldType(int fieldNo);

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

    // number of time steps
    int d_numSteps;

    // Fuelldata files: max. 199, really d_numSteps
    static const int MAXFILES = 200;
    CarData *d_dataFields[MAXFILES];

    // field names
    char *d_names[MAXFILES];

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED
    MultiCarData(const MultiCarData &);

    /// Assignment operator: NOT IMPLEMENTED
    MultiCarData &operator=(const MultiCarData &);

    /// Default constructor: NOT IMPLEMENTED
    MultiCarData();
};
#endif
