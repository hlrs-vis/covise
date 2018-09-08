// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "vvvecmath.h"

#ifdef __sun
#define sinf sin
#define cosf cos
#define acosf acos
#define sqrtf sqrt
#define expf exp
#define atanf atan
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

using namespace std;

//============================================================================
// vvVecmath Class Methods
//============================================================================

//----------------------------------------------------------------------------
/** Signum function.
  @param a number to test for sign
  @return 1 if a is positive or zero, -1 if a is negative
*/
float vvVecmath::sgn(float a)
{
  if (a>=0) return 1.0f;
  else return -1.0f;
}

//============================================================================
// vvMatrix Class Methods
//============================================================================

//----------------------------------------------------------------------------
/// Constructor for empty matrix (all values 0.0)
vvMatrix::vvMatrix()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = 0.0;
}

vvMatrix::vvMatrix(float d0, float d1, float d2, float d3)
{
  e[0][0] = d0;
  e[0][1] = 0;
  e[0][2] = 0;
  e[0][3] = 0;
  e[1][0] = 0;
  e[1][1] = d1;
  e[1][2] = 0;
  e[1][3] = 0;
  e[2][0] = 0;
  e[2][1] = 0;
  e[2][2] = d2;
  e[2][3] = 0;
  e[3][0] = 0;
  e[3][1] = 0;
  e[3][2] = 0;
  e[3][3] = d3;
}

//----------------------------------------------------------------------------
/// Constructor to init matrix from GLfloat array (GLfloat[16])
vvMatrix::vvMatrix(float* glf)
{
  int row, col, ite;

  ite = 0;
  for (row=0; row<4; ++row)
  {
    for (col=0; col<4; ++col)
    {
      e[row][col] = glf[ite];
      ite++;
    }
  }
}

//----------------------------------------------------------------------------
/** Subscript operator.
 */
float& vvMatrix::operator()(size_t row, size_t col)
{
  assert(row < 4);
  assert(col < 4);
  return e[row][col];
}

//---------------------------------------------------------------------------
/** Subscript operator.
 */
float const& vvMatrix::operator()(size_t row, size_t col) const
{
  assert(row < 4);
  assert(col < 4);
  return e[row][col];
}

//----------------------------------------------------------------------------
/** Multiplication. Operands will be multiplied from left to right.
 */
vvMatrix vvMatrix::operator+(const vvMatrix& operand) const
{
  vvMatrix result = *this;
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      result.e[i][j] += operand.e[i][j];
    }
  }
  return result;
}


//----------------------------------------------------------------------------
/** Multiplication. Operands will be multiplied from left to right.
 */
vvMatrix vvMatrix::operator-(const vvMatrix& operand) const
{
  vvMatrix result = *this;
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      result.e[i][j] -= operand.e[i][j];
    }
  }
  return result;
}

//----------------------------------------------------------------------------
/** Multiplication. Operands will be multiplied from left to right.
 */
vvMatrix vvMatrix::operator*(const vvMatrix& RHS) const
{
  vvMatrix LHS(*this);
  LHS.multiplyRight(RHS);
  return LHS;
}

//----------------------------------------------------------------------------
/// Print the matrix to stdout for debugging
void vvMatrix::print(const char* title) const
{
  int row, col;

  cerr << title << endl;

  cerr.setf(ios::fixed, ios::floatfield);
  cerr.precision(3);

  for (row=0; row<4; ++row)
  {
    for (col=0; col<4; ++col)
      cerr << setw(10) << e[row][col];
    cerr << endl;
  }
}

//----------------------------------------------------------------------------
/// Set identity matrix (diagonal=1, rest=0)
void vvMatrix::identity()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      if (row==col) e[row][col] = 1.0;
  else e[row][col] = 0.0;
}

//----------------------------------------------------------------------------
/// Set all matrix elements to zero
void vvMatrix::zero()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = 0.0;
}

//----------------------------------------------------------------------------
/// Apply a translation.
vvMatrix& vvMatrix::translate(float x, float y, float z)
{
  e[0][3] += x;
  e[1][3] += y;
  e[2][3] += z;

  return *this;
}

//----------------------------------------------------------------------------
/// Apply a translation.
vvMatrix& vvMatrix::translate(const vvVector3& t)
{
  return translate(t[0], t[1], t[2]);
}

//----------------------------------------------------------------------------
/// Apply a non-uniform scale.
vvMatrix& vvMatrix::scaleLocal(float x, float y, float z)
{
  vvMatrix s;                                     // scaling matrix

  s.e[0][0] = x;
  s.e[1][1] = y;
  s.e[2][2] = z;
  s.e[3][3] = 1.0f;

  return multiplyRight(s);
}

//----------------------------------------------------------------------------
/** Apply a uniform scale.
  @param a scale factor
*/
vvMatrix& vvMatrix::scaleLocal(float s)
{
  return scaleLocal(s, s, s);
}

//----------------------------------------------------------------------------
/** Rotation about vector x/y/z by angle a (radian).
 (Source: Foley et.al. page 227)
 @return rotation matrix
*/
vvMatrix vvMatrix::rotate(float a, float x, float y, float z)
{
  vvMatrix rot;                                   // rotation matrix
  rot.identity();
  float cosfa, sinfa;                             // shortcuts

  // normalize vector:
  float d = sqrtf(x * x + y * y + z * z);         // divisor
  if (d == 0.0) 
  {
    cerr << "vvMatrix::rotate: invalid rotation vector" << endl;
    return rot;                       // division by zero error
  }
  float dInv = 1.0f / d;
  x *= dInv;
  y *= dInv;
  z *= dInv;

  // Precompute recurring values:
  cosfa = cosf(a);
  sinfa = sinf(a);

  // Compute rotation matrix:
  rot.e[0][0] = x * x + cosfa * (1 - x * x);
  rot.e[0][1] = x * y * (1 - cosfa) - z * sinfa;
  rot.e[0][2] = x * z * (1 - cosfa) + y * sinfa;
  rot.e[0][3] = 0.0;

  rot.e[1][0] = x * y * (1 - cosfa) + z * sinfa;
  rot.e[1][1] = y * y + cosfa * (1 - y * y);
  rot.e[1][2] = y * z * (1 - cosfa) - x * sinfa;
  rot.e[1][3] = 0.0;

  rot.e[2][0] = x * z * (1 - cosfa) - y * sinfa;
  rot.e[2][1] = y * z * (1 - cosfa) + x * sinfa;
  rot.e[2][2] = z * z + cosfa * (1 - z * z);
  rot.e[2][3] = 0.0;

  rot.e[3][0] = 0.0;
  rot.e[3][1] = 0.0;
  rot.e[3][2] = 0.0;
  rot.e[3][3] = 1.0;

  // Perform rotation:
  multiplyRight(rot);
  return rot;
}

//----------------------------------------------------------------------------
/** Rotation about vector v by angle a (radian).
  @return rotation matrix
 */
vvMatrix vvMatrix::rotate(float a, const vvVector3& v)
{
  return vvMatrix::rotate(a, v[0], v[1], v[2]);
}


//----------------------------------------------------------------------------
// Multiplies this matrix from the left with the given matrix: this = RHS * this
vvMatrix& vvMatrix::multiplyLeft(const vvMatrix& LHS)
{
  vvMatrix RHS(*this);

  for (size_t row = 0; row < 4; ++row)
  {
    for (size_t col = 0; col < 4; ++col)
    {
      e[row][col] = LHS(row, 0) * RHS(0, col)
                  + LHS(row, 1) * RHS(1, col)
                  + LHS(row, 2) * RHS(2, col)
                  + LHS(row, 3) * RHS(3, col);
    }
  }

  return *this;
}

//----------------------------------------------------------------------------
// Multiplies this matrix from the right with the given matrix: this = this * RHS
vvMatrix& vvMatrix::multiplyRight(const vvMatrix &RHS)
{
  vvMatrix LHS(*this);

  for (size_t row = 0; row < 4; ++row)
  {
    for (size_t col = 0; col < 4; ++col)
    {
      e[row][col] = LHS(row, 0) * RHS(0, col)
                  + LHS(row, 1) * RHS(1, col)
                  + LHS(row, 2) * RHS(2, col)
                  + LHS(row, 3) * RHS(3, col);
    }
  }

  return *this;
}

//----------------------------------------------------------------------------
/** Inverts an _orthogonal_ matrix
 Orthogonal means: columns and rows are perpenducular to each other and are unit vectors
*/
void vvMatrix::invertOrtho()
{
  int row, col;
  vvMatrix bak(*this);                             // backup of current matrix

  for (row=0; row<3; ++row)
    for (col=0; col<3; ++col)
      e[row][col] = bak.e[col][row];
}

//----------------------------------------------------------------------------
/// Inverts _only_ the 2D part of a matrix
void vvMatrix::invert2D()
{
  vvMatrix bak(*this);                             // backup of current matrix
  float factor;                                   // constant factor
  float det;                                      // 2D part determinant

  det = e[0][0] * e[1][1] - e[0][1] * e[1][0];
  if (det == 0.0) return;                         // determinant zero error
  factor = 1.0f / det;

  e[0][0] =  factor * bak.e[1][1];
  e[0][1] = -factor * bak.e[0][1];
  e[1][0] = -factor * bak.e[1][0];
  e[1][1] =  factor * bak.e[0][0];
}

//----------------------------------------------------------------------------
/// Transposes a matrix (=mirror at diagonal)
void vvMatrix::transpose()
{
  int row, col;
  vvMatrix bak(*this);                             // backup of current matrix

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = bak.e[col][row];
}

//----------------------------------------------------------------------------
/// Returns the multiplied diagonal values
float vvMatrix::diagonal()
{
  float mult = 1.0f;
  int i;

  for (i=0; i<4; ++i)
    mult *= e[i][i];
  return mult;
}

//----------------------------------------------------------------------------
/// Copies only the translational part of a matrix and keeps the rest untouched
void vvMatrix::copyTrans(const vvMatrix& m)
{
  e[0][3] = m.e[0][3];
  e[1][3] = m.e[1][3];
  e[2][3] = m.e[2][3];
}

//----------------------------------------------------------------------------
/// Copies only the rotational part of a matrix and keeps the rest untouched
void vvMatrix::copyRot(const vvMatrix& m)
{
  int row, col;

  for (row=0; row<3; ++row)
    for (col=0; col<3; ++col)
      e[row][col] = m.e[row][col];
}

//----------------------------------------------------------------------------
/// Make pure translational matrix.
void vvMatrix::transOnly()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      if (row==col) e[row][col] = 1.0f;
  else if (col!=3) e[row][col] = 0.0f;
}

//----------------------------------------------------------------------------
/// Make pure rotational matrix.
void vvMatrix::rotOnly()
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      if (row==3 || col==3)
      {
        if (row==3 && col==3) e[row][col] = 1.0f;
        else e[row][col] = 0.0f;
      }
}

//----------------------------------------------------------------------------
/// Overwrites the translational part of a matrix with 0's
void vvMatrix::killTrans()
{
  e[0][3] = e[1][3] = e[2][3] = 0.0;
}

//----------------------------------------------------------------------------
/// Overwrites the rotational part of a matrix with 0's
void vvMatrix::killRot()
{
  int row, col;

  for (row=0; row<3; ++row)
    for (col=0; col<3; ++col)
      e[row][col] = 0.0;
}

//----------------------------------------------------------------------------
/// Compares two matrices. Returns true if equal, otherwise false
bool vvMatrix::equal(const vvMatrix& m) const
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      if (e[row][col] != m.e[row][col])
      {
        return false;
      }

  return true;
}

//----------------------------------------------------------------------------
/** Stores the matrix elements in a linear array in column-major order.
  Column-Major order means (if matrix is written with translational
  part on the right side):
  <PRE>
  e0  e4  e8  e12
  e1  e5  e9  e13
  e2  e6  e10 e14
  e3  e7  e11 e15
  </PRE>
 Important: the array pointer must provide space for 16 float values!<BR>
 This function is needed to convert the matrix values to the OpenGL
format used e.g. in the glLoadMatrixf() command.
@see getGL
*/
void vvMatrix::getGL(float* array) const
{
  int row, col, i=0;

  for (col=0; col<4; ++col)
    for (row=0; row<4; ++row)
      array[i++] = e[row][col];
}

//----------------------------------------------------------------------------
/** Converts an OpenGL matrix to the vecmath matrix format
  @see getGL
*/
void vvMatrix::setGL(const float* glmatrix)
{
  int row, col, i=0;

  for (col=0; col<4; ++col)
    for (row=0; row<4; ++row)
      e[row][col] = glmatrix[i++];
}

//----------------------------------------------------------------------------
/** @see getGL
 */
void vvMatrix::setGL(const double* glmatrix)
{
  float mat[16];

  for (int i=0; i<16; ++i)
  {
    mat[i] = float(glmatrix[i]);
  }
  setGL(mat);
}

//----------------------------------------------------------------------------
/** Returns the matrix in float format.
 A float array with space for 16 float elements must be given!
*/
void vvMatrix::get(float* elements) const
{
  int col, row, i=0;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
  {
    elements[i] = e[row][col];
    ++i;
  }
}

//----------------------------------------------------------------------------
/** Copies the matrix from a source which is in float format.
 A float array with 16 float elements must be given!
*/
void vvMatrix::set(const float* elements)
{
  int col, row, i=0;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
  {
    e[row][col] = elements[i];
    ++i;
  }
}

//----------------------------------------------------------------------------
/** Returns the matrix in double format.
 A double array with space for 16 double elements must be given!
*/
void vvMatrix::get(double* elements) const
{
  int col, row, i=0;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
  {
    elements[i] = (double)e[row][col];
    ++i;
  }
}

//----------------------------------------------------------------------------
/** Copies the matrix from a source which is in double format.
 A double array with 16 double elements must be given!
*/
void vvMatrix::set(const double* elements)
{
  int col, row, i=0;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
  {
    e[row][col] = (float)elements[i];
    ++i;
  }
}

//----------------------------------------------------------------------------
/** Set matrix values for a specific row.
  @param row      row index
  @param a,b,c,d  new values (left to right)
*/
void vvMatrix::setRow(int row, float a, float b, float c, float d)
{
  e[row][0] = a;
  e[row][1] = b;
  e[row][2] = c;
  e[row][3] = d;
}

//----------------------------------------------------------------------------
/** Set matrix values of a specific row from a vector.
  The rightmost matrix element of the row is not changed.
  @param row  row index
  @param vec  vector with new elements
*/
void vvMatrix::setRow(int row, const vvVector3& vec)
{
  e[row][0] = vec[0];
  e[row][1] = vec[1];
  e[row][2] = vec[2];
}

//----------------------------------------------------------------------------
/** Set matrix values for a specific column.
  @param col      column index
  @param a,b,c,d  new values (top to bottom)
*/
void vvMatrix::setColumn(int col, float a, float b, float c, float d)
{
  e[0][col] = a;
  e[1][col] = b;
  e[2][col] = c;
  e[3][col] = d;
}

//----------------------------------------------------------------------------
/** Set matrix values of a specific column from a vector.
  The bottom matrix element of the column is not changed.
  @param col  column index
  @param vec  vector with new elements
*/
void vvMatrix::setColumn(int col, const vvVector3& vec)
{
  e[0][col] = vec[0];
  e[1][col] = vec[1];
  e[2][col] = vec[2];
}

//----------------------------------------------------------------------------
/** Get matrix values of a specific row.
  @param row row index
  @return Row values are found in a,b,c,d (left to right)
*/
void vvMatrix::getRow(int row, float* a, float* b, float* c, float* d) const
{
  *a = e[row][0];
  *b = e[row][1];
  *c = e[row][2];
  *d = e[row][3];
}

//----------------------------------------------------------------------------
/** Get matrix values of a specific row and store them in a vector.
  The rightmost matrix element of the row is ignored.
  @param row  row index
  @param vec  vector to obtain matrix elements
*/
void vvMatrix::getRow(int row, vvVector3* vec) const
{
  (*vec)[0] = e[row][0];
  (*vec)[1] = e[row][1];
  (*vec)[2] = e[row][2];
}

//----------------------------------------------------------------------------
/** Get matrix values of a specific column.
  @param col column index
  @return Column values are found in a,b,c,d (top to bottom)
*/
void vvMatrix::getColumn(int col, float* a, float* b, float* c, float* d)
{
  *a = e[0][col];
  *b = e[1][col];
  *c = e[2][col];
  *d = e[3][col];
}

//----------------------------------------------------------------------------
/** Get matrix values of a specific column and store them in a vector.
  The bottom matrix element of the column is ignored.
  @param col  column index
  @param vec  vector to obtain matrix elements
*/
void vvMatrix::getColumn(int col, vvVector3& vec)
{
  vec[0] = e[0][col];
  vec[1] = e[1][col];
  vec[2] = e[2][col];
}

//----------------------------------------------------------------------------
/// Creates matrix with random integer numbers in range from..to
void vvMatrix::random(int from, int to)
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = (float)(from + (rand() % (to - from + 1)));
}

//----------------------------------------------------------------------------
/// Creates matrix with random float numbers in range from..to
void vvMatrix::random(float from, float to)
{
  int row, col;

  for (row=0; row<4; ++row)
    for (col=0; col<4; ++col)
      e[row][col] = (float)from + ((float)rand()/(float)RAND_MAX * (float)(to - from));
}

//-----------------------------------------------------------------------
/** LU decomposition.
  From  Numerical Recipes, p. 46.<BR>
  Given a matrix a[n][n], this routine replaces the matrix by the
  LU decomposition of a rowwise permutation of itself. n and a are
  input. a is output. indx[n] is an output vector that records
  the row permutation effected by the partial pivoting; d is output
  +-1 depending on wether the number of eow interchanges was even
  or odd, respectively.
  @author Daniel Weiskopf
  @see LUBackSubstitution
*/
void vvMatrix::LUDecomposition(int index[4], float &d)
{
  const float TINY = 1.0e-20f;
  const int N = 4;                                // Special case for n=4, could be any positive number
  int i, imax, j, k;
  float big, dum, sum, temp;
  float vv[N];                                    // Stores the implicit scaling of each row

  d = 1.0f;

  // Loop over rows to get the implicit scaling information
  for(i = 0; i < N; i++)
  {
    big = 0.0f;
    for (j = 0; j < N; j++)
    {
      if ((temp = (float) fabs(e[i][j])) > big)
        big = temp;
    }
    if (big == 0.0)
      cerr << "Singular matrix in routine LUdcmp " << e[i][0] << " "
        << e[i][1] << " " << e[i][2] << " " << e[i][3];
    vv[i] = 1.0f / big;                           // Save the scaling
  }

  // Loop over columns for Crout's method
  for (j = 0; j < N; j++)
  {
    for (i = 0; i < j; i++)
    {
      sum = e[i][j];
      for (k = 0; k < i; k++)
        sum -= e[i][k] * e[k][j];
      e[i][j] = sum;
    }

    // Finds the pivot point
    big = 0.0f;
    imax = 0;
    for (i = j; i < N; i++)
    {
      sum = e[i][j];
      for (k = 0; k < j; k++)
        sum -= e[i][k] * e[k][j];
      e[i][j] = sum;
      if ((dum = vv[i] * (float) fabs(sum)) >= big)
      {
        big = dum;
        imax = i;
      }
    }

    // Do we need to interchange rows?
    if (j != imax)
    {
      for (k = 0; k < N; k++)
      {
        dum = e[imax][k];
        e[imax][k] = e[j][k];
        e[j][k] = dum;
      }
      d = -d;
      vv[imax] = vv[j];                           // Also interchange the scale factor
    }
    index[j] = imax;
    if (e[j][j] == 0.0)
      e[j][j] = TINY;
    if (j != N)
    {
      dum = 1/(e[j][j]);
      for (i = j+1; i < N; i++)
        e[i][j] *= dum;
    }
  }
}

//----------------------------------------------------------------------------
/** LU backsubstitution.
  @author Daniel Weiskopf
  @see LUDecomposition
*/
void vvMatrix::LUBackSubstitution(int index[4], float b[4])
{
  int   i, j;
  int   ii = -1;
  float sum;

  // Special case for n=4, could be any positive number
  const int n = 4;

  for (i = 0; i < n; i++)
  {
    const int ip    = index[i];
    sum   = b[ip];
    b[ip] = b[i];
    if (ii >= 0)
      for (j = ii; j <= i-1; j++)
        sum -= e[i][j] * b[j];
    else if (sum)
      ii = i;
    b[i] = sum;
  }

  for (i = n-1; i >= 0; i--)
  {
    sum = b[i];
    for (j = i+1; j < n; j++)
      sum -= e[i][j] * b[j];
    b[i] = sum/e[i][i];
  }
}

//----------------------------------------------------------------------------
/** Invert a matrix (according to Numerical Recipes, p. 48)
  @author Daniel Weiskopf
*/
void vvMatrix::invert(void)
{
  float    d;
  int      index[4];
  float    col[4];
  float    y[4][4];
  int      i;
  int      j;

  LUDecomposition(index, d);
  for ( j = 0; j < 4; j++)
  {
    for ( i = 0; i < 4; i++)
      col[i] = 0.0f;
    col[j] = 1.0f;
    LUBackSubstitution(index, col);
    for ( i = 0; i < 4; i++)
      y[i][j] = col[i];
  }

  for ( j = 0; j < 4; j++)
    for ( i = 0; i < 4; i++)
      e[i][j] = y[i][j];
}

//----------------------------------------------------------------------------
/** Swap two matrix rows
  @param row1, row2  rows to swap [0..3]
*/
void vvMatrix::swapRows(int row1, int row2)
{
  int col;
  float buffer;

  for (col=0; col<4; ++col)
  {
    buffer = e[row1][col];
    e[row1][col] = e[row2][col];
    e[row2][col] = buffer;
  }
}

//----------------------------------------------------------------------------
/** Swap two matrix columns
  @param col1, col2 columns to swap [0..3]
*/
void vvMatrix::swapColumns(int col1, int col2)
{
  int row;
  float buffer;

  for (row=0; row<4; ++row)
  {
    buffer = e[row][col1];
    e[row][col1] = e[row][col2];
    e[row][col2] = buffer;
  }
}

//----------------------------------------------------------------------------
/// Set OpenGL compatible projection matrix for orthogonal projection
void vvMatrix::setProjOrtho(float left, float right, float bottom, float top,
float nearPlane, float farPlane)
{
  e[0][0] = 2.0f / (right - left);
  e[0][1] = 0.0f;
  e[0][2] = 0.0f;
  e[0][3] = (left + right) / (left - right);
  e[1][0] = 0.0f;
  e[1][1] = 2.0f / (top - bottom);
  e[1][2] = 0.0f;
  e[1][3] = (bottom + top) / (bottom - top);
  e[2][0] = 0.0f;
  e[2][1] = 0.0f;
  e[2][2] = 2.0f / (nearPlane - farPlane);
  e[2][3] = (nearPlane + farPlane) / (nearPlane - farPlane);
  e[3][0] = 0.0f;
  e[3][1] = 0.0f;
  e[3][2] = 0.0f;
  e[3][3] = 1.0f;
}

//----------------------------------------------------------------------------
/// Get parameters of OpenGL parallel projection matrix
void vvMatrix::getProjOrtho(float* left, float* right, float* bottom, float* top,
float* nearPlane, float* farPlane)
{
  *left       = 2.0f / (e[0][0] * ( (e[0][3] - 1.0f) / (e[0][3] + 1.0f) - 1.0f ));
  *right      = (*left) + 2.0f / e[0][0];
  *bottom     = 2.0f / (e[1][1] * ( (e[1][3] - 1.0f) / (e[1][3] + 1.0f) - 1.0f ));
  *top        = (*bottom) + 2.0f / e[1][1];
  *nearPlane  = 2.0f / (e[2][2] * ( (1.0f - e[2][3]) / (1.0f + e[2][3]) + 1.0f ));
  *farPlane   = (*nearPlane) - 2.0f / e[2][2];
}

//----------------------------------------------------------------------------
/// Set OpenGL compatible projection matrix for perspective projection
void vvMatrix::setProjPersp(float left, float right, float bottom, float top,
float nearPlane, float farPlane)
{
  e[0][0] = (nearPlane + nearPlane)  / (right - left);
  e[0][1] = 0.0f;
  e[0][2] = (right + left) / (right - left);
  e[0][3] = 0.0f;
  e[1][0] = 0.0f;
  e[1][1] = (nearPlane + nearPlane)  / (top - bottom);
  e[1][2] = (top + bottom) / (top - bottom);
  e[1][3] = 0.0f;
  e[2][0] = 0.0f;
  e[2][1] = 0.0f;
  e[2][2] = (farPlane + nearPlane) / (nearPlane - farPlane);
  e[2][3] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
  e[3][0] = 0.0f;
  e[3][1] = 0.0f;
  e[3][2] = -1.0f;
  e[3][3] = 0.0f;
}

//----------------------------------------------------------------------------
/// Get parameters of OpenGL perspective projection matrix
void vvMatrix::getProjPersp(float* left, float* right, float* bottom, float* top,
float* nearPlane, float* farPlane)
{
  *nearPlane  = e[2][3] / 2.0f * ((e[2][2] + 1.0f) / (e[2][2] - 1.0f) - 1.0f);
  *farPlane   = e[2][3] * (*nearPlane) / (2.0f * (*nearPlane) + e[2][3]);
  *left       = 2.0f * (*nearPlane) / (e[0][0] * ((e[0][2] + 1.0f) / (e[0][2] - 1.0f) - 1.0f));
  *right      = (*left) * (e[0][2] + 1.0f) / (e[0][2] - 1.0f);
  *bottom     = 2.0f * (*nearPlane) / (e[1][1] * ((e[1][2] + 1.0f) / (e[1][2] - 1.0f) - 1.0f));
  *top        = (*bottom) * (e[1][2] + 1.0f) / (e[1][2] - 1.0f);
}

//----------------------------------------------------------------------------
/** Checks if the matrix could be used as an orthogonal projection matrix.
  Strategy: Given a matrix m:<PRE>
    A0 A1 A2 A3
    B0 B1 B2 B3
    C0 C1 C2 C3
    D0 D1 D2 D3
  </PRE>
  The distinguishing matrix elements are D0, D1, and D2. They are called
  perspective scale factors.<BR>
  D0 is the perspective scale factor for the X axis,<BR>
  D1 is the perspective scale factor for the Y axis,<BR>
D2 is the perspective scale factor for the Z axis.<BR>
If all scale factors are zero, an orthogonal projection is used.
@return true if matrix is an orthogonal projection matrix, otherwise false
*/
bool vvMatrix::isProjOrtho() const
{
  return (e[3][0]==0.0f && e[3][1]==0.0f && e[3][2]==0.0f);
}

//-----------------------------------------------------------------------------
/** This function works exactly the same as gluLookAt: it creates a viewing
  matrix derived from an eye point, a reference point indicating the center
  of the scene, and an UP vector.
*/
void vvMatrix::makeLookAt(float eyeX, float eyeY, float eyeZ,
                          float centerX, float centerY, float centerZ,
                          float upX, float upY, float upZ)
{
  vvVector3 f, up, s, u, center, eye;

  center[0] = centerX;
  center[1] = centerY;
  center[2] = centerZ;
  eye[0] = eyeX;
  eye[1] = eyeY;
  eye[2] = eyeZ;
  f = center - eye;
  f.normalize();
  up[0] = upX;
  up[1] = upY;
  up[2] = upZ;
  up.normalize();
  s = f ^ up;
  u = s ^ f;

  identity();
  e[0][0] =  s[0];
  e[0][1] =  u[0];
  e[0][2] = -f[0];
  e[1][0] =  s[1];
  e[1][1] =  u[1];
  e[1][2] = -f[1];
  e[2][0] =  s[2];
  e[2][1] =  u[2];
  e[2][2] = -f[2];

  vvMatrix trans;
  trans.identity();
  eye = -eye;
  trans.setColumn(3, eye);
  *this = trans * (*this);
}

//-----------------------------------------------------------------------------
/** Determine Z coordinate of near plane from glFrustum generated projection matrices.
  @return z coordinate of near plane
*/
float vvMatrix::getNearPlaneZ() const
{
  return e[2][3] / 2.0f * ((e[2][2] + 1.0f) / (e[2][2] - 1.0f) - 1.0f);
}

//-----------------------------------------------------------------------------
/** Rotates the matrix according to a fictitious trackball, placed in
    the middle of the given window.
    The trackball is approximated by a Gaussian curve.
    The trackball coordinate system is: x=right, y=up, z=to viewer<BR>
    The origin of the mouse coordinates zero (0,0) is considered to be top left.
  @param width, height  window size in pixels
  @param fromX, fromY   mouse starting position in pixels
  @param toX, toY       mouse end position in pixels
*/
vvMatrix vvMatrix::trackballRotation(int width, int height, int fromX, int fromY, int toX, int toY)
{
  const float TRACKBALL_SIZE = 1.3f;              // virtual trackball size (empirical value)
  vvMatrix mInv;                                  // inverse of ObjectView matrix
  vvVector3 v1, v2;                               // mouse drag positions in normalized 3D space
  float smallSize;                                // smaller window size between width and height
  float halfWidth, halfHeight;                    // half window sizes
  float angle;                                    // rotational angle
  float d;                                        // distance

  // Compute mouse coordinates in window and normalized to -1..1
  // ((0,0)=window center, (-1,-1) = bottom left, (1,1) = top right)
  halfWidth   = (float)width  / 2.0f;
  halfHeight  = (float)height / 2.0f;
  smallSize   = (halfWidth < halfHeight) ? halfWidth : halfHeight;
  v1[0]       = ((float)fromX - halfWidth)  / smallSize;
  v1[1]       = ((float)(height-fromY) - halfHeight) / smallSize;
  v2[0]       = ((float)toX   - halfWidth)  / smallSize;
  v2[1]       = ((float)(height-toY)   - halfHeight) / smallSize;

  // Compute z-coordinates on Gaussian trackball:
  d       = sqrtf(v1[0] * v1[0] + v1[1] * v1[1]);
  v1[2]   = expf(-TRACKBALL_SIZE * d * d);
  d       = sqrtf(v2[0] * v2[0] + v2[1] * v2[1]);
  v2[2]   = expf(-TRACKBALL_SIZE * d * d);

  // Compute rotational angle:
  angle = v1.angle(v2);                           // angle = angle between v1 and v2

  // Compute rotational axis:
  v2.cross(v1);                                   // v2 = v2 x v1 (cross product)

  // Convert axis coordinates (v2) from WCS to OCS:
  mInv.identity();
  mInv.copyRot(*this);                            // copy rotational part of mv to mInv
  mInv.invertOrtho();                             // invert orthogonal matrix mInv
  v2.multiply(mInv);                              // v2 = v2 x mInv (matrix multiplication)
  v2.normalize();                                 // normalize v2 before rotation

  // Perform acutal model view matrix modification:
  return rotate(-angle, v2[0], v2[1], v2[2]);      // rotate model view matrix
}

/** Compute Euler angles for a matrix. The angles are returned in Radians.
  Source: http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q37
*/
void vvMatrix::computeEulerAngles(float* angleX, float* angleY, float* angleZ)
{
  float tx, ty, c;

  *angleY = asinf(e[2][0]);                       // Calculate Y-axis angle
  c =  cosf(*angleY);
  if (fabs( c ) > 0.005f)                         // Gimball lock?
  {
    tx      =  e[2][2] / c;                       // No, so get X-axis angle
    ty      = -e[2][1] / c;
    *angleX = atan2f(ty, tx);
    tx      =  e[0][0] / c;                       // Get Z-axis angle
    ty      = -e[1][0] / c;
    *angleZ = atan2f( ty, tx );
  }
  else                                            // Gimball lock has occurred
  {
    *angleX = 0.0f;                               // Set X-axis angle to zero
    tx      = e[1][1];                            // And calculate Z-axis angle
    ty      = e[0][1];
    *angleZ = atan2f( ty, tx );
  }

  // Return only positive angles in [0, 2*VV_PI]:
  if (*angleX < 0.0f) *angleX += 2.0f * VV_PI;
  if (*angleY < 0.0f) *angleY += 2.0f * VV_PI;
  if (*angleZ < 0.0f) *angleZ += 2.0f * VV_PI;
}


//----------------------------------------------------------------------------
/// Default constructor.
vvPlane::vvPlane()
{
  _point = vvVector3(0.0f, 0.0f, 0.0f);
  _normal = vvVector3(0.0f, 0.0f, 1.0f);
}

//----------------------------------------------------------------------------
/// Constructor for point-normal format.
vvPlane::vvPlane(const vvVector3& p, const vvVector3& n)
{
  _point = p;
  _normal = n;
  _normal.normalize();
}

//----------------------------------------------------------------------------
/// Constructor for point-vector-vector format.
vvPlane::vvPlane(const vvVector3& p, const vvVector3& dir1, const vvVector3& dir2)
{
  _point = p;
  _normal = dir1;
  _normal.cross(dir2);
  _normal.normalize();
}

//----------------------------------------------------------------------------
/** Check if two points are on the same side of the plane.
  @param p1,p2 points to check
  @return true if points are on the same side
*/
bool vvPlane::isSameSide(const vvVector3& p1, const vvVector3& p2) const
{
  if (vvVecmath::sgn(dist(p1)) == vvVecmath::sgn(dist(p2))) return true;
  else return false;
}

//----------------------------------------------------------------------------
/** Computes the distance of a point from the plane.
  @param p   normal and point defining a plane
  @return distance of point to plane, sign determines side of the plane on which the
          point lies: positive=side into which normal points, negative=other side
*/
float vvPlane::dist(const vvVector3& p) const
{
  const float d = -_normal.dot(_point); // scalar component of hessian form of plane
  return (_normal.dot(p) + d);
}

//============================================================================
// Functions for STANDALONE mode
//============================================================================

#ifdef VV_STANDALONE

//----------------------------------------------------------------------------
/// test routines for vvMatrix
void testMatrix()
{
  vvMatrix m1;
  vvMatrix m2;
  vvMatrix m3;
  float glmatrix[16] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
  float left, right, bottom, top, nearPlane, farPlane;

  m1.identity();
  m1.print("m1 =");
  m2.identity();
  m2.scale(3.0f);
  m2.print("m2 =");
  m2.translate(2.0, 3.0, 4.0);
  m2.print("m2.translate(2.0, 3.0, 4.0)=");
  m3.e[0][0] = 1.0;
  m3.e[0][2] = 2.0;
  m3.print("m3 =");
  m3.rotate((float)VV_PI / 2.0f, 1.0f, 0.0f, 0.0f);
  m3.print("m3.rotate(M_PI / 2.0, 1.0, 0.0, 0.0)=");
  m1.random(0, 5);
  m1.print("m1.random(0, 5)=");
  m2.random(0, 5);
  m2.print("m2.random(0, 5)=");
  m1.multiplyPre(&m2);
  m1.print("m1.multiply(&m2)");
  m2.random(1.0f, 10.0f);
  m2.print("m2.random(1.0, 10.0)=");
  m1.getGL((float*)m2.e);
  m2.print("m1.getGL((float*)m2.e)=");
  m2.getGL(glmatrix);
  m2.print("m2.getGL(glmatrix)=");
  m2.transpose();
  m2.print("m2.transpose()=");

  m1.random(1, 9);
  m1.print("m1.random(1, 9) =");
  m1.rotOnly();
  m1.print("m1.rotOnly()= ");

  m1.random(1, 9);
  m1.print("m1.random(1, 9) =");
  m1.transOnly();
  m1.print("m1.transOnly()= ");

  m1.random(1, 9);
  m1.print("m1.random(1, 9) =");
  m1.invert();
  m1.print("m1.invert()=");

  m1.setProjOrtho(-2.0, 5.0, -4.0, 7.0, -250.0, 999.0);
  m1.print("m1.setProjOrtho(-2.0, 5.0, -4.0, 7.0, -250.0, 999.0) =");
  m1.getProjOrtho(&left, &right, &bottom, &top, &nearPlane, &farPlane);
  cerr << "left=" << left << ", right=" << right << ", bottom=" << bottom <<
    ", top=" << top << ", nearPlane=" << nearPlane << ", farPlane=" << farPlane << endl;

  m1.setProjPersp(-12.0, 15.0, -14.0, 17.0, -25.0, 99.0);
  m1.print("m1.setProjPersp(-12.0, 15.0, -14.0, 17.0, -25.0, 99.0) =");
  m1.getProjPersp(&left, &right, &bottom, &top, &nearPlane, &farPlane);
  cerr << "left=" << left << ", right=" << right << ", bottom=" << bottom <<
    ", top=" << top << ", nearPlane=" << nearPlane << ", farPlane=" << farPlane << endl;
}

//----------------------------------------------------------------------------
/// test routines for vvVector3 and vvVector4
void testVector()
{
  vvMatrix m;
  vvVector3 v1(1.0, 2.0, 3.0);
  vvVector3 v2, v3, v4, v5, v6;
  float result;

  v3.print("v3=");
  v4.random(0.0, 9.0);
  v4.print("v4.random(0.0, 0.9)= ");
  v1.print("v1=");
  v2.random(1, 5);
  v2.print("v2.random(1, 5)= ");

  v3.copy(&v2);
  v3.add(&v1);
  v3.print("v2.add(&v1)= ");

  v3.copy(&v2);
  v3.sub(&v1);
  v3.print("v2.sub(&v1)= ");

  v3.copy(&v2);
  result = v3.dot(&v1);
  cerr << "v2.dot(&v1)= " << result << endl;

  v3.copy(&v2);
  v3.cross(&v1);
  v3.print("v2.cross(&v1)= ");

  v3.copy(&v2);
  result = v3.distance(&v1);
  cerr << "v2.distance(&v1)= " << result << endl;

  v3.copy(&v2);
  v3.scale(3.0);
  v3.print("v2.scale(3.0)= ");

  v3.copy(&v2);
  v3.normalize();
  v3.print("v2.normalize= ");

  m.random(1, 9);
  m.print("Matrix m=");
  v3.copy(&v2);
  v3.multiply(&m);
  v3.print("v2.multiply(&m)= ");

  v2.set(2.0, 3.0, 0.0);
  v2.print("v2= ");
  v3.set(5.0, 0.0, 0.0);
  v3.print("v3= ");
  cerr << "v3.angle(&v2)= " << v3.angle(&v2) * 180.0 / VV_PI << endl;

  v3.set(1.0, 2.0, 3.0);
  v3.print("v3=");
  v4.set(-6.0, -2.0, 9.0);
  v4.print("v4=");
  v3.swap(&v4);
  cerr << "v3.swap(&v4)" << endl;
  v3.print("v3=");
  v4.print("v4=");

  int numIntersections;
  v2.set(0.0, 0.0, 0.0);
  v3.set(0.0, 0.0, 1.0);
  v4.set(-2.0, -3.0, 0.0);
  v5.set(0.0, 1.0, 0.0);
  numIntersections = v1.isectRayCylinder(&v2, &v3, 3.0f, &v4, &v5);

  v1.set(1.0, 1.0, 1.0);
  v1.print("v1=");
  v2.set(1.0, 1.0, 1.0);
  v2.print("v2=");
  v3.set(0.0, 0.0, 1.0);
  v3.print("v3=");
  v4.set(1.0, 0.0, 1.0);
  v4.print("v4=");
  v3.isectPlaneRay(&v1, &v2, &v4, &v3);
  v3.print("isectPlaneRay(&v1, &v2, &v4, &v3)=");

  v1.set(1.0, 1.0, 1.0);
  v2.set(4.0, 1.0, 1.0);
  v3.set(4.0, 3.0, 1.0);
  v4.set(0.9f, 1.0, 0.0);
  v5.set(0.0, 0.0, 1.0);
  v6.isectRayTriangle(&v4, &v5, &v1, &v2, &v3);
  v6.print("isectRayTriangle");

  v1.set(0.0, 0.0, 0.0);
  v2.set(0.0, 0.0, 1.0);
  v3.set(1.0, 2.0, 0.0);
  v4.set(0.0, 0.0, 1.0);
  cerr << "dist=" << v5.isectLineLine(v1, v2, v3, v4) << endl;
  v5.print("isectLineLine");
}

//----------------------------------------------------------------------------
/// main routine, only used in STANDALONE mode
int main(int, char*)
{
  srand(42);
  testMatrix();
  testVector();
  return 1;
}
#endif

/////////////////
// End of File
/////////////////
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
