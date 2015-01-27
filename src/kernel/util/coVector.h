/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_VECTOR_H
#define COVISE_VECTOR_H

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

#include <math.h>
#include <string.h>
#include <iostream>
#include <stdio.h>
#include "coExport.h"

using std::ostream;
namespace covise
{

class coMatrix;

class UTILEXPORT coVector
{

    friend class coMatrix;

public:
    int dim;
    double vec4[4];
    double *vec;

    coVector()
    {
        dim = 3;
        vec = vec4;
    }

    coVector(const coVector &v)
    {
        dim = v.dim;
        if (dim < 5)
            vec = vec4;
        else
            vec = new double[dim];
        for (int i = 0; i < dim; i++)
            vec[i] = v[i];
    }

    coVector &operator=(const coVector &v)
    {
        dim = v.dim;
        if (dim < 5)
            vec = vec4;
        else
            vec = new double[dim];
        for (int i = 0; i < dim; i++)
            vec[i] = v[i];
        return *this;
    }

    coVector(int d)
    {
        dim = d;
        if (d < 5)
            vec = vec4;
        else
            vec = new double[d];
    }
    coVector(int d, double *_val)
    {
        dim = d;
        if (d < 5)
            vec = vec4;
        else
            vec = new double[d];
        for (int ctr = 0; ctr < d; ++d)
            vec[ctr] = _val[ctr];
    }

    ~coVector()
    {
        if (dim > 4)
            delete[] vec;
    }

    coVector(double _val[3])
    {
        dim = 3;
        vec = vec4;
        vec[0] = _val[0];
        vec[1] = _val[1];
        vec[2] = _val[2];
    }

    coVector(double a, double b, double c)
    {
        dim = 3;
        vec = vec4;
        vec[0] = a;
        vec[1] = b;
        vec[2] = c;
        vec[3] = 1.;
    }
    coVector(double a, double b, double c, double d)
    {
        dim = 4;
        vec = vec4;
        vec[0] = a;
        vec[1] = b;
        vec[2] = c;
        vec[3] = d;
    }

    int operator==(coVector &v);

    coVector operator+(const coVector &v) const;

    coVector operator-(const coVector &v) const;

    coVector operator-() const;

    double operator*(const coVector &v) const;

    coVector operator*(const coMatrix &m) const;

    coVector operator*(double r) const;

    double &operator[](int i)
    {
        return vec[i];
    }

    double operator[](int i) const
    {
        return vec[i];
    }

    //
    // vector operators:
    //
    double length() const;

    coVector scale(double f)
    {
        return *this * f;
    }

    double dot(const coVector &v) const;

    coVector cross(const coVector &v) const;

    coVector eval(const coVector &v) const;

    coVector unitize() const;

    coVector negate() const;

    // print the vector:

    void print(FILE *f, char *n = 0);

    void get(double f[3])
    {
        f[0] = vec[0];
        f[1] = vec[1];
        f[2] = vec[2];
    }

    void get(float f[3])
    {
        f[0] = (float)vec[0];
        f[1] = (float)vec[1];
        f[2] = (float)vec[2];
    }

    coVector maximal(const coVector &v) const;

    double enclosedAngle(const coVector &v) const;

    bool isZero() const;

    coVector &normalize();

    friend ostream &operator<<(ostream &O, const coVector &v)
    {
        O << "[" << v[0] << "|" << v[1] << "|" << v[2] << "]";
        return O;
    }
    /*{\Mbinopfunc  writes $v$ componentwise to the output stream $O$.}*/

    //friend istream& operator>>(istream& I, coVector& v);
    /*{\Mbinopfunc  reads $v$ componentwise from the input stream $I$.}*/
};
}
#endif // COVISE_VECTOR_H
