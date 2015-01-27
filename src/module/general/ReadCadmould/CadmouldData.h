/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CADMOULD_DATA_H_
#define __CADMOULD_DATA_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS CadmouldData
//
// Initial version: 2002-03-25 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

/**
 * Virtual Base class for all Cadmould data files
 *
 */
class CadmouldData
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor : abstract base class, c'tor and d'tor empty
       */
    CadmouldData(bool swap);

    /// Destructor : virtual in case we derive objects
    virtual ~CadmouldData();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// get status. 0 if ok, >0 errno <0 other error
    int getState();

    /// get number of timesteps - 0 = stationary
    virtual int numTimeSteps() = 0;

    /// get number of data fields
    virtual int numFields() = 0;

    /// get number of vertices saved
    virtual int numVert() = 0;

    /// get name of data field
    virtual const char *getName(int fieldNo) = 0;

    /// get character of the magnitude
    enum FieldType
    {
        SCALAR_FLOAT,
        SCALAR_INT
    };

    virtual FieldType getFieldType(int fieldNo);

    /// get percentage of timestep, -1 if stationary
    virtual int percent(int stepNo) = 0;

    /** get a certain step of a specified field into a given buffer
       * @return =0 ok, errno otherwise
       */
    virtual int readField(int fieldNo, int stepNo, void *buffer) = 0;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    static void byteSwap(int no_points, void *buffer);

    bool byte_swap;

    // state: 0=ok, -1 unknown error, >1 errno
    int d_state;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED
    CadmouldData(const CadmouldData &);

    /// Assignment operator: NOT IMPLEMENTED
    CadmouldData &operator=(const CadmouldData &);
};
#endif
