/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          solve.h  -  solve heat conduction
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __SOLVE_H_

#define __SOLVE_H_

#include "arrays.h"
#include "classext.h"

// ***************************************************************************
// tridiagonal matrix class a*ndTemp(i-1)+b*ndTemp(i)+c*ndTemp(i+1) = rhs
// ***************************************************************************

class TTridag
{
public:
    TTridag()
    {
        at = bt = ct = rhst = 0;
        a = b = c = rhs = 0;
        ndTemp = qvol = 0;
        fa = 1;
    }

    prec a, at, b, bt, c, ct, rhs, rhst; // matrix coefficients
    prec ndTemp; // temperature
    prec qvol; // volumetric source
    prec fa; // diffusivity ratio
};

class TTridagMatrix : public Vector<TTridag>
{
public:
    TTridagMatrix()
        : Vector<TTridag>(){};
    TTridagMatrix(unsigned i)
        : Vector<TTridag>(i){};

    int Solve(int);
};

// ***************************************************************************
// class for solving heat conduction in a column
// ***************************************************************************

class HeatConduction
{
public:
    HeatConduction();

    int Solve(int, int, prec &); // solve heat conduction in depth
    int CalcTemp(int, int, prec &, prec &); // calc temp profile for given
    // recession
    // find root in interval
    int FindRoot(int, int, prec &, prec &, prec &,
                 prec &, prec &, prec);
    // calc recession
    void CalcRecession(int i, int j, prec &, prec &);
    // speed

    Vector<TPoint3D> TempSai; // temperature derivative at node k
    rmvector PowerDensity; // absorbed power density at node k
    TPoint3D saitmax; // maximum displacement
    TTridagMatrix m; // tridiagonal matrix
    prec abszta; // magnitude of gradient zta
};
#endif
