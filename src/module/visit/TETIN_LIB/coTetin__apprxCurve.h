/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__APPROXIMATE_CURVE_H_
#define _CO_TETIN__APPROXIMATE_CURVE_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 02.02.00

/**
 * Class coTetin__apprxCurve implements "approx. curve" command
 *
 */
class coTetin__apprxCurve : public coTetinCommand
{

private:
    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__apprxCurve(const coTetin__apprxCurve &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__apprxCurve &operator=(const coTetin__apprxCurve &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__apprxCurve()
        : apprx_family(0)
        , n_names(0)
        , names(0){};

    // ===================== the command's data =====================

    // value
    int apprx_family; // if <> 0: apprx. family_names else curve_names
    int n_names;
    char **names;

public:
    /// read from file
    coTetin__apprxCurve(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__apprxCurve(int *&intDat, float *&floatDat, char *&charDat);

    coTetin__apprxCurve(int apprx_family_in,
                        int n_names_in, const char **names_in);

    /// Destructor
    virtual ~coTetin__apprxCurve();

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
