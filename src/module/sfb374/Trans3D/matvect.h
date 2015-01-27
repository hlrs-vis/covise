/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          matvect.h  -  matrix operations
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/*
    Copyright (c) 1995 Namir C. Shammas

  Version 1.0                                Date 8/9/94

  C++ module for basic vect and array operations:

  + Add matrices
  + Subtract matrices
  + Multiply matrices
  + Solve a set of linear equarions using the
Gauss-Jordan method
+ Solve a set of linear equarions using the
LU decomposition method
+ Solve a set of linear equarions using the
Gauss-Seidel method
+ Obtain the inverse of a matrix
+ Obatin the determinant of a matrix
*/

#ifndef _MATVECT_H_
#define _MATVECT_H_

#include "arrays.h"

#define MATVECT_EPSILON 1.0e-15
#define MATVECT_BAD_RESULT -1.0e+30

class MatrixOp
{
public:
    enum MatErrType
    {
        matErr_None,
        matErr_Size,
        matErr_Singular,
        matErr_IllConditioned,
        matErr_IterLimit
    };

    void Copy(rmatrix &MatA, rmatrix &MatB);
    MatErrType AddMat(rmatrix &aMat, rmatrix &MatB, int numRows, int numCols);
    MatErrType SubMat(rmatrix &MatA, rmatrix &MatB, int numRows, int numCols);
    MatErrType MulMat(rmatrix &MatA, rmatrix &MatB, rmatrix &MatC,
                      int numRowsA, int numColsA, int numRowsB, int numColsB);
    MatErrType GaussJordan(rmatrix &A, rmatrix &B, int numRows, int numCols);
    MatErrType LUDecomp(rmatrix &A, ivector &Index,
                        int numRows, int &rowSwapFlag);
    void LUBackSubst(rmatrix &A, ivector &Index,
                     int numRows, rvector &B);
    MatErrType GaussSeidel(rmatrix &A, rvector &B, rvector &X,
                           int numRows, int maxIter,
                           double eps1, double eps2);
    void LUInverse(rmatrix &A, rmatrix &InvA,
                   ivector &Index, int numRows);
    MatErrType MatInverse(rmatrix &A, int numRows);
    double LUDeterminant(rmatrix &A, int numRows, int rowSwapFlag);
    double MatDeterminant(rmatrix &A, int numRows);
    MatErrType GaussElim(rmatrix &A, rmatrix &B, int numRows, int numCols);
};
#endif
