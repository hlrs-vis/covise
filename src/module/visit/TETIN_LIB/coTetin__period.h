/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__PERIODIC_H_
#define _CO_TETIN__PERIODIC_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 09.06.99

/**
 * Class coTetin__period implements Tetin file "periodic" command
 *
 */
class coTetin__period : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__period(const coTetin__period &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__period &operator=(const coTetin__period &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__period(){};

    // ===================== the command's data =====================

    float d_x, d_y, d_z, d_nx, d_ny, d_nz, d_angle;

public:
    /// read from file
    coTetin__period(istream &str, int binary);

    /// read from memory
    coTetin__period(int *&, float *&, char *&);

    /// Destructor
    virtual ~coTetin__period();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================
};
#endif
