/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Interface classes for application modules to the COVISE   **
 **              software environment                                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C)1997 RUS                                **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author: D. Rantzau                                                     **
 ** Date:  15.08.97  V1.0                                                  **
\**************************************************************************/
#include <assert.h>
#include "coVector.h"
#include "coMatrix.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define coMAX(a, b) a > b ? a : b

using namespace covise;

//==========================================================================
//
//==========================================================================
coVector coVector::operator+(const coVector &v) const
{
    coVector c(dim);
    for (int i = 0; i < dim; i++)
        c[i] = vec[i] + v.vec[i];
    return c;
}

//==========================================================================
//
//==========================================================================
coVector coVector::operator-(const coVector &v) const
{
    coVector c(dim);
    for (int i = 0; i < dim; i++)
        c[i] = vec[i] - v.vec[i];
    return c;
}

coVector coVector::operator-() const
{
    return negate();
}

//==========================================================================
//
//==========================================================================

double coVector::operator*(const coVector &v) const
{
    double r = 0.0;
    for (int i = 0; i < dim; i++)
        r += vec[i] * v.vec[i];
    return r;
}

//==========================================================================
//
//==========================================================================

coVector coVector::operator*(const coMatrix &m) const
{
    //supresses the use of higherdimensional vectors
    assert(dim == 3 || dim == 4);

    coVector result;
    for (int j = 0; j < 4; j++)
    {
        result.vec[j] = 0;
        for (int k = 0; k < 4; k++)
        {
            result.vec[j] += vec[k] * m.val[k][j];
        }
    }
    if (dim == 3)
    {
        for (int i = 0; i < 4; i++)
        {
            result.vec[i] /= result.vec[3];
        }
    }
    return result;
}

//==========================================================================
//
//==========================================================================

coVector coVector::operator*(double r) const
{
    coVector c(dim);
    for (int i = 0; i < dim; i++)
        c[i] = vec[i] * r;
    return c;
}

//==========================================================================
//
//==========================================================================
coVector coVector::cross(const coVector &v) const
{
    coVector c(dim);
    if (dim == 3)
    {
        c[0] = vec[1] * v.vec[2] - vec[2] * v.vec[1];
        c[1] = vec[2] * v.vec[0] - vec[0] * v.vec[2];
        c[2] = vec[0] * v.vec[1] - vec[1] * v.vec[0];
        return c;
    }
    else
        return 0;
}

//==========================================================================
//
//==========================================================================
coVector coVector::eval(const coVector &v) const
{
    coVector c(dim);
    for (int i = 0; i < dim; i++)
        c[i] = vec[i] * v.vec[i];
    return c;
}

//==========================================================================
//
//==========================================================================
coVector coVector::unitize() const
{
    coVector c(dim);
    double r = length();
    if (r == 0.)
        r = 1.; // avoiding numerical overflows
    for (int i = 0; i < dim; i++)
        c[i] = vec[i] / r;
    return c;
}

//==========================================================================
//
//==========================================================================
coVector coVector::negate() const
{
    coVector c(dim);
    for (int i = 0; i < dim; i++)
        c[i] = -vec[i];
    return c;
}

//==========================================================================
//
//==========================================================================
void coVector::print(FILE *f, char *n)
{
    if (dim == 3)
        fprintf(f, "%s{%f %f %f} ", n ? n : "", vec[0], vec[1], vec[2]);
    if (dim == 4)
        fprintf(f, "%s{%f %f %f %f} ", n ? n : "", vec[0], vec[1], vec[2], vec[3]);
}

//==========================================================================
//
//==========================================================================
coVector coVector::maximal(const coVector &v) const
{
    // return a vector with the maximal value of both vectors in EACH direction:
    coVector c(dim);
    for (int i = 0; i < dim; i++)
        c[i] = coMAX(vec[i], v.vec[i]);
    return c;
}

//==========================================================================
//
//==========================================================================

int coVector::operator==(coVector &v)
{
    double r = 0.0;
    for (int i = 0; i < dim; i++)
        r += (vec[i] - v.vec[i]) * (vec[i] - v.vec[i]);
    return (r == 0);
}

//==========================================================================
//
//==========================================================================
double coVector::dot(const coVector &v) const
{
    double c = 0.0;

    for (int i = 0; i < dim; i++)
        c += (vec[i] * v.vec[i]);
    return c;
}

//==========================================================================
//
//==========================================================================
double coVector::length() const
{
    double c = 0.0;

    for (int i = 0; i < dim; i++)
        c += (vec[i] * vec[i]);

    c = sqrt(c);

    return c;
}

bool coVector::isZero() const
{

    const double eps = 0.00000000001;

    for (int i = 0; i < dim; ++i)
    {
        if ((vec[i] >= eps) || (vec[i] <= -eps))
            return false;
    }

    return true;
}

double coVector::enclosedAngle(const coVector &v) const
{

    const double eps = 0.00000000001;

    if ((dim != v.dim) || isZero() || v.isZero())
        return 0.0;

    double acosAngle = dot(v) / (length() * v.length());

    if (acosAngle + eps > 1.0)
        return 0.0;
    if (acosAngle - eps < -1.0)
        return M_PI;

    return acos(acosAngle);
}

coVector &coVector::normalize()
{

    const double eps = 0.00000000001;

    double len = length();

    if (fabs(len) < eps)

    {
        len = 1;
    }

    else

    {

        for (int i = 0; i < dim; ++i)
        {
            vec[i] /= len;
        }
    }

    return *this;
}
