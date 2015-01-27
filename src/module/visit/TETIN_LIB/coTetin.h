/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN_H_
#define _CO_TETIN_H_

#include <iostream.h>
#include <string.h>
#include <covise/covise_binary.h>

// 01.06.99

class coTetinCommand;

/**
 * Class
 *
 */
class coTetin : public coBinaryObject
{

public:
    enum Command
    {
        ILLEGAL = 0,
        BOCO_FILE,
        READ_FAMILY_BOCO,
        AFFIX,
        DEFINE_CURVE,
        DEFINE_DENSITY_POLYGON,
        DEFINE_MODEL,
        DEFINE_SURFACE,
        DEFINE_THIN_CUT,
        MATERIAL_POINT,
        MERGE_SURFACE,
        PERIODIC,
        PRESCRIBED_POINT,
        SET_TRIANGULATION_TOLERANCE,
        TRANSLATIONAL,
        TETIN_FILENAME,
        REPLAY_FILENAME,
        CONFIGDIR_NAME,
        START_HEXA,
        OUTPUT_INTERF,
        TRIANGULATE_FAMILY,
        APPROXIMATE_CURVE,
        GET_PRESCPNT,
        TRANSLATE_GEOM,
        DELETE_FAMILY,
        DELETE_SURFACE,
        DELETE_CURVE,
        PROJECT_POINT
    };

private:
    /// Max. number of include levels: currently 64
    enum
    {
        MAX_INCLUDE_LEVELS = 64
    };

    // currently used stream
    istream *actStream;

    // read a tetin file: called recursively for include statements
    coTetinCommand *readFile(istream &str, int binary, int twoPass);

    // chain of Commands
    coTetinCommand *d_commands;

    /// Copy-Constructor: NOT  IMPLEMENTED
    coTetin(const coTetin &){};

    // Assignment operator: NOT  IMPLEMENTED
    coTetin &operator=(const coTetin &)
    {
        return *this;
    }

protected:
public:
    coTetinCommand *getCommands()
    {
        return d_commands;
    };
    /** read a tetin file
       *  @param  str       stream to read from
       *  @param  binary    =1 for binary, =0 for text file (binary not yet impl.)
       *  @param  twoPass   =1 for two-, =0 for one-pass (two-pass not yet impl.)
       */

    // Default constructor
    coTetin()
    {
        d_commands = 0;
    };

    coTetin(istream &str, int binary = 0, int twoPass = 0);

    /** read from shared memory
       *  @param  intPtr   Pointer in integer field, advanced after read
       *  @param  floatPtr Pointer in float   field, advanced after read
       *  @param  charPtr  Pointer in char    field, advanced after read
       */
    coTetin(const int *&intPtr, const float *&floatPtr, const char *&chPtr);

    /// make a Tetin object from a covise_binary
    coTetin(DO_BinData *binObj);

    /// Destructor
    virtual ~coTetin();

    // virtual functions from coBinaryObject

    /// get the type
    const char *getType() const;

    /// find out how much data space must be allocated
    virtual void getSize(int &numInt, int &numFloat, int &numChar) const;

    /// copy object into binary storage
    virtual void getBinary(int *intDat, float *floatDat, char *charDat) const;

    virtual void append(coTetinCommand *new_com);

    /// print to a stream
    void print(ostream &str);
};
#endif
