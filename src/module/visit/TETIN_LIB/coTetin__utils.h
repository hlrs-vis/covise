/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__UTILS_H_
#define _CO_TETIN__UTILS_H_

#include "iostream.h"
#include "coTetinCommand.h"

// 08.06.99

/**
 * utilities for coTetin__
 *
 */

class coTetin__BSpline
{
public:
    int ncps[2]; // number of control points
    int ord[2]; // order in u and v
    int rat; // rational flag
    float *knot[2]; // knots
    float *cps;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// put given set of pointers to my data
    virtual void putBinary(int *&intDat, float *&floatDat, char *&charDat);
    coTetin__BSpline();
    ~coTetin__BSpline();
};

coTetin__BSpline *read_spline(istream &str, ostream &ostr);
coTetin__BSpline *read_spline_curve(istream &str, ostream &ostr);
int comment_getc(istream &str, ostream &ostr);
int getgplfloat(istream &str, ostream &ostr, float *val);
int coTetin__break_line(char *temp, char *linev[]);
#endif
