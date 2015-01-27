/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__TRANSLATIONAL_H_
#define _CO_TETIN__TRANSLATIONAL_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 09.06.99

/**
 * Class coTetin__transl implements Tetin file "translational" command
 *
 */
class coTetin__transl : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__transl(const coTetin__transl &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__transl &operator=(const coTetin__transl &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__transl(){};

    // ===================== the command's data =====================

    float d_x, d_y, d_z, d_nx, d_ny, d_nz, d_angle;

public:
    /// read from file
    coTetin__transl(istream &str, int binary);

    /// read from memory
    coTetin__transl(int *&, float *&, char *&);

    /// Destructor
    virtual ~coTetin__transl();

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
