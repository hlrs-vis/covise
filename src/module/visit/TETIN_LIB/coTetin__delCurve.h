/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__DELETE_CURVE_H_
#define _CO_TETIN__DELETE_CURVE_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 07.10.99

/**
 * Class coTetin__delCurve implements "delete curves" command
 *
 */

class coTetin__delCurve : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__delCurve(const coTetin__delCurve &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__delCurve &operator=(const coTetin__delCurve &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__delCurve()
        : n_curves(0)
        , curve_names(0){};

    // ===================== the command's data =====================

    // value
    int n_curves;
    char **curve_names;

public:
    /// read from file
    coTetin__delCurve(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__delCurve(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__delCurve(int n_fam, char **names);

    /// Destructor
    virtual ~coTetin__delCurve();

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
