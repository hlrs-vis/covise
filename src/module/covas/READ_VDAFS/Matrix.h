/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MATRIX_H
#define MATRIX_H

#include <math.h>
#include <string.h>
#include <assert.h>

#define EPS 1.0e-15
#ifndef REAL
typedef double REAL;
#endif

////////////////////////////// 2x2-Matrix //////////////////////////////
class Matrix_2x2
{
    REAL mat[2][2];

public:
    inline Matrix_2x2(){};
    inline Matrix_2x2(REAL a00, REAL a01, REAL a10, REAL a11);
    inline Matrix_2x2(const Matrix_2x2 &m);

    // Set/get
    inline REAL get(int row, int col) const;
    inline void set(int row, int col, REAL val);
    inline void set(const Matrix_2x2 &m);

    // Operators
    inline REAL *operator[](int i);
    inline const REAL *operator[](int i) const;
    inline Matrix_2x2 &operator=(const Matrix_2x2 &m);

    // Math
    inline REAL det(const Matrix_2x2 &m);
    inline REAL det();
    inline void invert(const Matrix_2x2 &m);
};

///////////////////////// Inlines of Matrix2x2 /////////////////////////
inline REAL Matrix_2x2::get(int row, int col) const
{
    if (row >= 0 && row < 2 && col >= 0 && col < 2)
        return mat[row][col];
    else // !!!!!!!! Should WARN !!!!!!!!!!
        return 0.0f;
    //    else	!!!!!!!! Should WARN !!!!!!!!!!
}

inline void Matrix_2x2::set(int row, int col, REAL val)
{
    if (row >= 0 && row < 2 && col >= 0 && col < 2)
        mat[row][col] = val;
    //    else	!!!!!!!! SHould WARN !!!!!!!!!!
}

inline void Matrix_2x2::set(const Matrix_2x2 &v)
{
    memcpy(mat, v.mat, (size_t)(sizeof(REAL) * 4));
}

inline Matrix_2x2::Matrix_2x2(REAL a00, REAL a01, REAL a10, REAL a11)
{
    set(0, 0, a00);
    set(0, 1, a01);
    set(1, 0, a10);
    set(1, 1, a11);
}

inline Matrix_2x2::Matrix_2x2(const Matrix_2x2 &m)
{
    set(m);
}

inline REAL *Matrix_2x2::operator[](int i)
{
    return mat[i];
}

inline const REAL *Matrix_2x2::operator[](int i) const
{
    return mat[i];
}

inline Matrix_2x2 &Matrix_2x2::operator=(const Matrix_2x2 &m)
{
    set(m);
    return *this;
}

inline REAL Matrix_2x2::det(const Matrix_2x2 &m)
{
    return (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
}

inline REAL Matrix_2x2::det()
{
    return (mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]);
}

inline void Matrix_2x2::invert(const Matrix_2x2 &m)
{
    REAL valDet;
    valDet = det(m);

    if (fabs(valDet) > EPS)
    {
        mat[0][0] = 1 / valDet * m[1][1];
        mat[0][1] = -1 / valDet * m[0][1];
        mat[1][0] = -1 / valDet * m[1][0];
        mat[1][1] = 1 / valDet * m[0][0];
    }
    else
        cerr << "ERROR: singular Matrix_2x2 " << endl;
}

//////////////////////////////// Matrix ////////////////////////////////

class Matrix // n x n Matrix with arbitrary dimension n
{

private:
    REAL **mat;
    int dim; // dimension of matrix

public:
    // constructors and destructors
    inline Matrix()
    {
        // cout << "Matrix standard constructor" << endl;
        dim = 0;
        mat = NULL;
    };
    inline Matrix(int n);
    inline Matrix(const Matrix &_m);
    inline ~Matrix();

    // set/get
    inline REAL get(int row, int col) const;

    // operators
    inline REAL *operator[](int index);
    inline REAL *operator[](int index) const;
    inline Matrix &operator=(const Matrix &_m);

    // Math functions
    inline void transpose(const Matrix &_m);

    //-------- for debugging
    inline void output()
    {
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
                cout << mat[i][j] << "\t";
            cout << endl;
        }
    }
    //--------
};

/////////////////////////// Inlines of Matrix ///////////////////////////
inline Matrix::Matrix(int n)
{
    // cout << "Matrix non-trivial constructor" << endl;
    dim = n;

    mat = new REAL *[dim];
    assert(mat != 0);
    //variables must be declared outside 'for'
    //since otheriwse under Linux we get
    //name lookup of `i' changed for new ANSI `for' scoping
    int i;
    for (i = 0; i < dim; i++)
    {
        mat[i] = new REAL[dim];
        assert(mat[i] != 0);
    }

    // initialize
    for (i = 0; i < dim; i++)
    {
        //variables must be declared outside 'for'
        //since otheriwse under Linux we get
        //name lookup of `j' changed for new ANSI `for' scoping
        int j;
        for (j = 0; j < dim; j++)
            mat[i][j] = 0.0;
    }
}

inline Matrix::Matrix(const Matrix &_m)
    : dim(_m.dim)
{
    // cout << "Matrix copy constructor" << endl;
    mat = new REAL *[dim];
    assert(mat != 0);

    for (int i = 0; i < dim; i++)
    {
        mat[i] = new REAL[dim];
        assert(mat[i] != 0);
        memcpy(mat[i], _m.mat[i], (size_t)(sizeof(REAL) * dim));
    }
}

inline Matrix::~Matrix()
{
    // cout << "Matrix destructor" << endl;
    for (int i = 0; i < dim; i++)
        delete[] mat[i];
    delete[] mat;
}

inline REAL Matrix::get(int row, int col) const
{
    if (row >= 0 && row < dim && col >= 0 && col < dim)
        return mat[row][col];
    else // !!!!!!!! Should WARN !!!!!!!!!!
        return 0.0f;
    //    else	!!!!!!!! Should WARN !!!!!!!!!!
}

inline REAL *Matrix::operator[](int index)
{
    return (mat[index]);
}

inline REAL *Matrix::operator[](int index) const
{
    return (mat[index]);
}

inline Matrix &Matrix::operator=(const Matrix &_m)
{
    if (this == &_m)
        return *this;
    //variables must be declared outside 'for'
    //since otheriwse under Linux we get
    //name lookup of `i' changed for new ANSI `for' scoping
    int i;
    for (i = 0; i < dim; i++)
        delete[] mat[i];
    delete[] mat;

    dim = _m.dim;
    mat = new REAL *[dim];
    assert(mat != 0);

    for (i = 0; i < dim; i++)
    {
        mat[i] = new REAL[dim];
        assert(mat[i] != 0);
        memcpy(mat[i], _m.mat[i], (size_t)(sizeof(REAL) * dim));
    }
    return *this;
}

inline void Matrix::transpose(const Matrix &_m)
{
    int i, j;

    if (this == &_m)
    {
        Matrix temp = _m;
        for (i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
                mat[i][j] = temp.mat[j][i];
    }
    else
    {
        for (i = 0; i < dim; i++)
            delete[] mat[i];
        delete[] mat;

        dim = _m.dim;
        mat = new REAL *[dim];
        assert(mat != 0);

        for (i = 0; i < dim; i++)
        {
            mat[i] = new REAL[dim];
            assert(mat[i] != 0);

            for (j = 0; j < dim; j++)
                mat[i][j] = _m.mat[j][i];
        }
    }
}
#endif // MATRIX_H
