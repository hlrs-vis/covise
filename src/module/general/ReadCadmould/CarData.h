/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CAR_DATA_H_
#define __CAR_DATA_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS CarData
//
// Initial version: 2002-05-06 [sk]
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

class CarData : public CadmouldData
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    CarData(const char *path, bool swap);

    /// Destructor : virtual in case we derive objects
    virtual ~CarData();

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

    typedef struct /* text */
    {
        int idx;
        char txt[36];
    } STR;

    typedef struct /* value is either int or float */
    {
        float f;
        int i;
    } VAL;

    typedef struct
    {
        char ver[8]; /* Version CAR_MAGIC */
        char prg[32]; /* name of simulation program*/
        float tim; /* cycle time [s] */
        float lev; /* 0 <= Fuellstand <= 1 */
        int lin; /* number of lines */
        int num_col; /* number of colomns */
        int floatg; /* flags */
        int mod; /* number of DISPLAY_MODES */
    } CAR_HEADER;

    typedef struct
    {
        char typ; /* = typ[col] - typ of colomn( FLOAT or INT ) */
        STR fmt; /* = fmt[col] - C printout format */
        STR tit; /* = tit[col] - title */
        STR uni; /* = uni[col] - unit */
        STR xpl; /* = xpl[col] - explanation for non-existing values */
        int cmp; /* = cmp[col] - Spaltenkompatibilitaet > 0 */
        unsigned char bit; /* = bit[col][(mod+8)/8] - bits of display mode */
        VAL sca; /* = sca[col] - scale factor of values */
        VAL min; /* = min[col] - min of values in this column */
        VAL max; /* = max[col] - max  of values in this column*/
        // VAL_P val;		/* = val[col][lin] - results */
    } CAR_COL;

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // names of my fields
    char **d_fieldNames;

    char *d_path;

    // field types
    CadmouldData::FieldType *d_fieldTypes;

    // number of fields used
    int d_numFields;

    // number of vertices in this data set
    int d_numVert;

    // the open file
    FILE *d_file;

private:
    CAR_COL col;
    CAR_HEADER car_hdr; // header
    STR *title; // list of titles

    char *buf; // buffer to read file in
    long fpos; // file position

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED
    CarData(const CarData &);

    /// Assignment operator: NOT IMPLEMENTED
    CarData &operator=(const CarData &);

    /// Default constructor: NOT IMPLEMENTED
    CarData();

    /// read STR from file
    void readSTR(STR *str);

    /// read from buffer
    void getString(STR *tgt, int fieldNo);
    void getObject(void *tgt, size_t size, int fieldNo);
    void getValue(VAL *tgt, char type, int fieldNo);
};
#endif
