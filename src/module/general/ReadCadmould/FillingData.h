/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FILLING_DATA_H_
#define __FILLING_DATA_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS FillingData
//
// Initial version: 2002-04-19 [sl]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "CadmouldData.h"
#include <util/coviseCompat.h>
/**
 * This Class models the data found in the simulation file as
 * given in the browser parameter.
 */

class FillingData : public CadmouldData
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    FillingData(const char *path, int numFields, //int no_points,
                const char *const *fieldNames, const CadmouldData::FieldType *,
                bool swap);

    /// Destructor : virtual in case we derive objects
    virtual ~FillingData();

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

    virtual CadmouldData::FieldType getFieldType(int fieldNo);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // names of my fields
    const char *const *d_fieldNames;

    // field types
    const CadmouldData::FieldType *fieldTypes;

    // number of fields used
    int d_numFields;

    // number of vertices in this data set
    int d_numVert;

    // the open file
    FILE *d_file;

    struct header
    {
        int numnode;
        float min, max;
        int maxfill;
    };

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED
    FillingData(const FillingData &);

    /// Assignment operator: NOT IMPLEMENTED
    FillingData &operator=(const FillingData &);

    /// Default constructor: NOT IMPLEMENTED
    FillingData();
};
#endif
