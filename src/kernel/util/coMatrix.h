/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_MATRIX_H
#define COVISE_MATRIX_H

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

#include "coExport.h"
#include "coVector.h"

namespace covise
{

class UTILEXPORT coMatrix
{

    friend class coVector;

    double val[4][4];

public:
    int changed;

    // this is a flag for optimization
    // if 0 ->matrix is unity matrix
    //coMatrix() { unity(); };
    coMatrix()
    {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                val[i][j] = 0.0;
    };

    coMatrix(double *m)
    {
        int i, j;
        for (i = 0; i < 4; i++)
            for (j = 0; j < 4; j++)
                val[i][j] = m[i * 4 + j];
    };

    coMatrix(const coMatrix &m)
    {
        int i, j;
        for (i = 0; i < 4; i++)
            for (j = 0; j < 4; j++)
                val[i][j] = m.val[i][j];
    };

    ~coMatrix(){};

    //
    // Matrix operators:
    //
    coMatrix &operator=(const coMatrix &m)
    {
        int i, j;
        for (i = 0; i < 4; i++)
            for (j = 0; j < 4; j++)
                val[i][j] = m.val[i][j];
        return *this;
    };

    int operator==(const coMatrix &) const;
    double &operator()(int i, int j)
    {
        return val[i][j];
    };
    coMatrix operator+(const coMatrix &) const;
    coMatrix operator-(const coMatrix &) const;
    coMatrix operator*(const coMatrix &) const;
    coVector operator*(const coVector &) const;
    coMatrix operator*(const double &) const;

    void set(int i, int j, double d)
    {
        val[i][j] = d;
        changed = 1;
    };

    double get(int i, int j) const
    {
        return val[i][j];
    };

    coMatrix invers() const;
    coMatrix transpose() const;
    void fromQuat(const float, const float, const float, const float);

    //
    // Matrix transformations
    //

    coMatrix scaleS(const double &d) const
    {
        coMatrix m = *this * d;
        m.changed = 1;
        return m;
    };

    coMatrix invScaleS(const double &d)
    {
        coMatrix m = *this * -d;
        m.changed = 1;
        return m;
    };

    coMatrix scale(const coVector &v) const
    {
        coMatrix m1, m2;
        m1.set(0, 0, v.vec[0]);
        m1.set(1, 1, v.vec[1]);
        m1.set(2, 2, v.vec[2]);
        m2 = *this * m1;
        m2.changed = 1;
        return m2;
    };

    coMatrix invScale(const coVector &v) const
    {
        coMatrix m1, m2;
        m1.set(0, 0, 1 / v.vec[0]);
        m1.set(1, 1, 1 / v.vec[1]);
        m1.set(2, 2, 1 / v.vec[2]);
        m2 = m1 * *this;
        m2.changed = 1;
        return m2;
    };

    coMatrix translation(const coVector &) const;
    coMatrix invTranslation(const coVector &) const;

    coMatrix rotationX(const double) const;
    coMatrix invRotationX(const double) const;

    coMatrix rotationY(const double) const;
    coMatrix invRotationY(const double) const;

    coMatrix rotationZ(const double) const;
    coMatrix invRotationZ(const double) const;

    coMatrix rotation(const coVector &v) const
    {
        coMatrix m1, m2;
        double sinx = sin(v.vec[0]);
        double siny = sin(v.vec[1]);
        double sinz = sin(v.vec[2]);

        double cosx = cos(v.vec[0]);
        double cosy = cos(v.vec[1]);
        double cosz = cos(v.vec[2]);

        m1.val[0][0] = cosy * cosz;
        m1.val[0][1] = cosy * sinz;
        m1.val[0][2] = -siny;

        m1.val[1][0] = cosz * sinx * siny - cosx * sinz;
        m1.val[1][1] = cosx * cosz + sinx * siny * sinz;
        m1.val[1][2] = cosy * sinx;

        m1.val[2][0] = cosx * cosz * siny + sinx * sinz;
        m1.val[2][1] = -(cosz * sinx) + cosx * siny * sinz;
        m1.val[2][2] = cosx * cosy;

        m2 = *this * m1;
        m2.changed = 1;
        return m2;
    };

    coMatrix invRotation(const coVector &v) const
    {
        coMatrix m1, m2;
        double sinx = sin(-v.vec[0]);
        double siny = sin(-v.vec[1]);
        double sinz = sin(-v.vec[2]);

        double cosx = cos(-v.vec[0]);
        double cosy = cos(-v.vec[1]);
        double cosz = cos(-v.vec[2]);

        m1.val[0][0] = cosy * cosz;
        m1.val[0][1] = cosz * sinx * siny + cosx * sinz;
        m1.val[0][2] = -(cosx * cosz * siny) + sinx * sinz;

        m1.val[1][0] = -(cosy * sinz);
        m1.val[1][1] = cosx * cosz - sinx * siny * sinz;
        m1.val[1][2] = cosz * sinx + cosx * siny * sinz;

        m1.val[2][0] = siny;
        m1.val[2][1] = -(cosy * sinx);
        m1.val[2][2] = cosx * cosy;

        m2 = m1 * *this;
        m2.changed = 1;
        return m2;
    };

    coMatrix &setRotation(const coVector &axis, double angle);

    void get(float f[4][4])
    {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                f[i][j] = (float)val[i][j];
    };

    // print the matrix:
    void print(FILE *f, char *n = 0)
    {
        fprintf(f, "%s{ %f %f %f %f ", n ? n : "", val[0][0], val[0][1], val[0][2], val[0][3]);
        fprintf(f, "%f %f %f %f ", val[1][0], val[1][1], val[1][2], val[1][3]);
        fprintf(f, "%f %f %f %f ", val[2][0], val[2][1], val[2][2], val[2][3]);
        fprintf(f, "%f %f %f %f } ", val[3][0], val[3][1], val[3][2], val[3][3]);
    };

    void unity()
    {
        changed = 0;
        val[0][1] = val[0][2] = val[0][3] = val[1][0] = val[1][2] = val[1][3] = val[2][0] = val[2][1] = val[2][3] = val[3][0] = val[3][1] = val[3][2] = 0;
        val[0][0] = val[1][1] = val[2][2] = val[3][3] = 1;
    };

    friend ostream &operator<<(ostream &O, const coMatrix &m)
    {
        O << m.val[0][0] << " " << m.val[0][1] << " " << m.val[0][2] << " " << m.val[0][3] << ","
          << m.val[1][0] << " " << m.val[1][1] << " " << m.val[1][2] << " " << m.val[1][3] << ","
          << m.val[2][0] << " " << m.val[2][1] << " " << m.val[2][2] << " " << m.val[2][3] << ","
          << m.val[3][0] << " " << m.val[3][1] << " " << m.val[3][2] << " " << m.val[3][3];
        return O;
    };
};
}
#endif // COVISE_MATRIX_H
