/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__SET_TRIANGULATION_TOLERANCE_H_
#define _CO_TETIN__SET_TRIANGULATION_TOLERANCE_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 04.06.99

/**
 * Class coTetin__trianTol implements Tetin file "set_triangulation_tolerance" command
 *
 */
class coTetin__trianTol : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__trianTol(const coTetin__trianTol &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__trianTol &operator=(const coTetin__trianTol &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__trianTol(){};

    // ===================== the command's data =====================

    // value
    float d_val;

public:
    /// read from file
    coTetin__trianTol(istream &str, int binary);

    /// read from memory
    coTetin__trianTol(int *&intDat, float *&floatDat, char *&charDat);

    /// Destructor
    virtual ~coTetin__trianTol();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;

    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================

    /// construct from value
    coTetin__trianTol(float val)
        : coTetinCommand(coTetin::SET_TRIANGULATION_TOLERANCE)
    {
        d_val = val;
    }

    /// get the value
    float getValue() const
    {
        return d_val;
    }

    /// set the value
    void setValue(float val)
    {
        d_val = val;
    }
};
#endif
