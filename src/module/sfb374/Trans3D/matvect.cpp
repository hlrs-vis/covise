/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          matvect.cpp  -  description
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include "matvect.h"

// ***************************************************************************
// matrix operation class
// ***************************************************************************

// copy matrix

void MatrixOp::Copy(rmatrix &MatA, rmatrix &MatB)
{
    unsigned i, j;

    for (i = 0; i < MatA.GetiMax(); i++)
        for (j = 0; j < MatA.GetjMax(); j++)
            MatA(i, j) = MatB(i, j);
}

// add matrices

MatrixOp::MatErrType MatrixOp::AddMat(rmatrix &MatA, rmatrix &MatB,
                                      int numRows, int numCols)
{
    int row, col;

    if (!(MatA.CheckIndex(numRows - 1, numCols - 1) && MatB.CheckIndex(numRows - 1, numCols - 1) && numCols != numRows))
        return matErr_Size;

    for (row = 0; row < numRows; row++)
        for (col = 0; col < numCols; col++)
            MatA(row, col) += MatB(row, col);

    return matErr_None;
}

// subtract matrices

MatrixOp::MatErrType MatrixOp::SubMat(rmatrix &MatA, rmatrix &MatB,
                                      int numRows, int numCols)
{
    int row, col;

    if (!(MatA.CheckIndex(numRows - 1, numCols - 1) && MatB.CheckIndex(numRows - 1, numCols - 1) && numCols != numRows))
        return matErr_Size;

    for (row = 0; row < numRows; row++)
        for (col = 0; col < numCols; col++)
            MatA(row, col) -= MatB(row, col);

    return matErr_None;
}

// multiply matrices

MatrixOp::MatErrType MatrixOp::MulMat(rmatrix &MatA, rmatrix &MatB, rmatrix &MatC,
                                      int numRowsA, int numColsA,
                                      int numRowsB, int numColsB)
{
    int row, col, k;

    if (!(MatA.CheckIndex(numRowsA - 1, numColsA - 1) && MatB.CheckIndex(numRowsA - 1, numColsA - 1) && numColsA == numRowsB))
        return matErr_Size;

    for (row = 0; row < numRowsB; row++)
    {
        for (col = 0; col < numColsA; col++)
            MatA(row, col) = 0;
        for (k = 0; k < numColsA; k++)
            MatA(row, col) += MatB(row, k) * MatC(row, k);
    }
    return matErr_None;
}

// Gauss-Jordan elimination

MatrixOp::MatErrType MatrixOp::GaussJordan(rmatrix &A, rmatrix &B,
                                           int numRows, int numCols)
{
    int *rowIndex = new int[A.GetiMax()];
    int *colIndex = new int[A.GetjMax()];
    int *pivotIndex = new int[A.GetjMax()];
    int i, j, k, n, m;
    int row, col;
    double large, z, oneOverPiv;

    if (!A.CheckIndex(numRows - 1, numCols - 1))
        return matErr_Size;

    // initialize the row and column indices
    for (i = 0; i < numRows; i++)
    {
        rowIndex[i] = i;
        colIndex[i] = i;
    }

    // initialize the pivot index array
    for (i = 0; i < numRows; i++)
        pivotIndex[i] = -1;

    for (i = 0; i < numRows; i++)
    {
        large = 0;

        // look for Pivot element
        for (j = 0; j < numRows; j++)
            if (pivotIndex[j] != 0)
                for (k = 0; k < numCols; k++)
                {
                    if (pivotIndex[k] == -1)
                    {
                        if (fabs(A(j, k)) >= large)
                        {
                            large = fabs(A(j, k));
                            row = j;
                            col = k;
                        }
                    }
                    else if (pivotIndex[k] > 0)
                    {
                        delete[] colIndex;
                        delete[] rowIndex;
                        delete[] pivotIndex;
                        return matErr_Singular;
                    }
                }
        pivotIndex[col] += 1;
        if (row != col) // shift element to diagonal
        {
            for (n = 0; n < numRows; n++)
                swap(A(row, n), A(col, n));
            for (n = 0; n < numCols; n++)
                swap(B(row, n), B(col, n));
        }
        rowIndex[i] = row;
        colIndex[i] = col;
        if (fabs(A(col, col)) < 1.e-10)
        {
            delete[] colIndex;
            delete[] rowIndex;
            delete[] pivotIndex;
            return matErr_Singular;
        }
        oneOverPiv = 1 / A(col, col);
        A(col, col) = 1;
        for (n = 0; n < numRows; n++)
            A(col, n) *= oneOverPiv;
        for (n = 0; n < numCols; n++)
            B(col, n) *= oneOverPiv;
        for (m = 0; m < numRows; m++)
            if (m != col)
            {
                z = A(m, col);
                A(m, col) = 1;
                for (n = 0; n < numRows; n++)
                    A(m, n) -= A(col, n) * z;
                for (n = 0; n < numCols; n++)
                    B(m, n) -= B(col, n) * z;
            }
    }
    for (n = numRows - 1; n >= 0; n--)
    {
        if (rowIndex[n] != colIndex[n])
            for (k = 0; k < numRows; k++)
                swap(A(k, rowIndex[n]), A(k, colIndex[n]));
    }
    delete[] colIndex;
    delete[] rowIndex;
    delete[] pivotIndex;
    return matErr_None;
}

// LU decomposition

MatrixOp::MatErrType MatrixOp::LUDecomp(rmatrix &A, ivector &Index,
                                        int numRows, int &rowSwapFlag)
{
    int i, j, k, iMax;
    double large, sum, z, z2;
    rvector scaleVect(A.GetjMax());

    // initialize row interchange flag
    rowSwapFlag = 1;
    // loop to obtain the scaling element
    for (i = 0; i < numRows; i++)
    {
        large = 0;
        for (j = 0; j < numRows; j++)
        {
            z2 = fabs(A(i, j));
            large = (z2 > large) ? z2 : large;
        }
        // no non-zero large value? then exit with an error code
        if (large == 0)
            return matErr_Singular;
        scaleVect[i] = 1 / large;
    }
    for (j = 0; j < numRows; j++)
    {
        for (i = 0; i < j; i++)
        {
            sum = A(i, j);
            for (k = 0; k < i; k++)
                sum -= A(i, k) * A(k, j);
            A(i, j) = sum;
        }
        large = 0;
        for (i = j; i < numRows; i++)
        {
            sum = A(i, j);
            for (k = 0; k < j; k++)
                sum -= A(i, k) * A(k, j);
            A(i, j) = sum;
            z = scaleVect[i] * fabs(sum);
            if (z >= large)
            {
                large = z;
                iMax = i;
            }
        }
        if (j != iMax)
        {
            for (k = 0; k < numRows; k++)
            {
                z = A(iMax, k);
                A(iMax, k) = A(j, k);
                A(j, k) = z;
            }
            rowSwapFlag *= -1;
            scaleVect[iMax] = scaleVect[j];
        }
        Index[j] = iMax;
        if (A(j, j) == 0)
            A(j, j) = MATVECT_EPSILON;
        if (j != numRows)
        {
            z = 1 / A(j, j);
            for (i = j + 1; i < numRows; i++)
                A(i, j) *= z;
        }
    }
    return matErr_None;
}

// LU back substitution

void MatrixOp::LUBackSubst(rmatrix &A, ivector &Index,
                           int numRows, rvector &B)
{
    int i, j, idx, k = -1;
    double sum;

    for (i = 0; i < numRows; i++)
    {
        idx = Index[i];
        sum = B[idx];
        B[idx] = B[i];
        if (k > -1)
            for (j = k; j < i; j++)
                sum -= A(i, j) * B[j];
        else if (sum != 0)
            k = i;
        B[i] = sum;
    }
    for (i = numRows - 1; i >= 0; i--)
    {
        sum = B[i];
        for (j = i + 1; j < numRows; j++)
            sum -= A(i, j) * B[j];
        B[i] = sum / A(i, i);
    }
}

// Gauss-Seidel elimination

MatrixOp::MatErrType MatrixOp::GaussSeidel(rmatrix &A, rvector &B, rvector &X,
                                           int numRows, int maxIter,
                                           double eps1, double eps2)
{
    enum opType
    {
        opContinue,
        opConverge,
        opSingular,
        opError
    };
    rvector Xold(numRows);
    double denom, dev, devMax;
    int i, j, iter = 0;
    enum opType operType = opContinue;

    // normalize matrix A and vector B
    for (i = 0; i < numRows; i++)
    {
        denom = A(i, i);
        if (denom < eps1)
            return matErr_Singular;
        B[i] /= denom;
        for (j = 0; j < numRows; j++)
            A(i, j) /= denom;
    }

    // perform Gauss-Seidel iteration
    while (operType == opContinue)
    {
        for (i = 0; i < numRows; i++)
        {
            Xold[i] = X[i];
            X[i] = 0;
            for (j = 0; j < numRows; j++)
                if (j != i)
                    X[i] -= A(i, j) * X[j];
            X[i] += B[i];
        }

        // check for the convergence
        devMax = fabs(Xold[0] - X[0]) / X[0];
        for (i = 1; i < numRows; i++)
        {
            dev = fabs(Xold[i] - X[i]) / X[i];
            devMax = (dev > devMax) ? dev : devMax;
        }
        if (devMax <= eps2)

            operType = opConverge;
        else
        {
            iter++;
            if (iter > maxIter)
                operType = opError;
        }
    }

    switch (operType)
    {
    case opConverge:
        return matErr_None;

    case opSingular:
        return matErr_Singular;

    case opError:
        return matErr_IterLimit;

    default:
        return matErr_None;
    }
}

// inverse of LU matrix

void MatrixOp::LUInverse(rmatrix &A, rmatrix &InvA, ivector &Index, int numRows)
{
    rvector colVect(numRows);
    int i, j;

    for (j = 0; j < numRows; j++)
    {
        for (i = 0; i < numRows; i++)
            colVect[i] = 0;
        colVect[j] = 1;
        LUBackSubst(A, Index, numRows, colVect);
        for (i = 0; i < numRows; i++)
            InvA(i, j) = colVect[i];
    }
}

// inverse of matrix

MatrixOp::MatErrType MatrixOp::MatInverse(rmatrix &A, int numRows)
{
    rvector colVect(numRows);
    ivector Index(numRows);
    int i, j;
    int rowSwapFlag;
    MatErrType err;

    err = LUDecomp(A, Index, numRows, rowSwapFlag);
    if (err != matErr_None)
        return err;

    for (j = 0; j < numRows; j++)
    {
        for (i = 0; i < numRows; i++)
            colVect[i] = 0;
        colVect[j] = 1;
        LUBackSubst(A, Index, numRows, colVect);
        for (i = 0; i < numRows; i++)
            A(i, j) = colVect[i];
    }

    return matErr_None;
}

// determinant of LU matrix

double MatrixOp::LUDeterminant(rmatrix &A, int numRows, int rowSwapFlag)
{
    double result = (double)rowSwapFlag;
    int i;

    for (i = 0; i < numRows; i++)
        result *= A(i, i);

    return result;
}

// determinant of matrix

double MatrixOp::MatDeterminant(rmatrix &A, int numRows)
{
    ivector Index(numRows);
    int i;
    int rowSwapFlag;
    MatErrType err;
    double result;

    err = LUDecomp(A, Index, numRows, rowSwapFlag);
    if (err != matErr_None)
        return MATVECT_BAD_RESULT;

    result = (double)rowSwapFlag;

    for (i = 0; i < numRows; i++)
        result *= A(i, i);

    return result;
}

// Gauss elimination

MatrixOp::MatErrType MatrixOp::GaussElim(rmatrix &MatA, rmatrix &MatB,
                                         int numRows, int numCols)
{
    int i, j, k, n;
    int row, numColsB = MatB.GetjMax();
    double large, oneOverPiv;

    if (!MatA.CheckIndex(numRows - 1, numCols - 1))
        return matErr_Size;

    n = 0;
    for (j = 0; j < numCols; j++)
    {
        large = 0;
        for (i = n; i < numRows; i++) // look for Pivot element
        {
            if (fabs(MatA(i, j)) > large)
            {
                large = fabs(MatA(i, j));
                row = i;
            }
        }

        if (large > 0)
        {
            oneOverPiv = 1.0 / MatA(row, j); // Pivot factor
            if (row != n)
            {
                for (i = 0; i < numCols; i++) // switch rows
                    swap(MatA(row, i), MatA(n, i));
                for (i = 0; i < numColsB; i++)
                    swap(MatB(row, i), MatB(n, i));
                row = n;
            }
            n++;
            for (i = j; i < numCols; i++)
                MatA(row, i) *= oneOverPiv;
            for (i = 0; i < numColsB; i++)
                MatB(row, i) *= oneOverPiv;

            for (k = n; k < numRows; k++)
            {
                large = MatA(k, j);
                MatA(k, j) = 0.0;
                for (i = j + 1; i < numCols; i++)
                    MatA(k, i) -= large * MatA(row, i);
                for (i = 0; i < numColsB; i++)
                    MatB(k, i) -= large * MatB(row, i);
            }
        }
    }

    return matErr_None;
}
