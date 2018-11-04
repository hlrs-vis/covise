/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_VEC_H
#define VRUI_VEC_H

#include <util/coTypes.h>
#include <math.h>

namespace vrui
{

class OPENVRUIEXPORT vruiVec
{

public:
    vruiVec(double x, double y, double z)
    {
        this->size = 3;
        vec[0] = x;
        vec[1] = y;
        vec[2] = z;
    }

    vruiVec(double x, double y, double z, double w)
    {
        this->size = 4;
        vec[0] = x;
        vec[1] = y;
        vec[2] = z;
        vec[3] = w;
    }

    vruiVec(int size)
    {
        this->size = size;
        this->vec = new double(size);
    }

    vruiVec(const vruiVec &second)
    {
        this->size = second.size;
        this->vec = new double(this->size);
        for (int ctr = 0; ctr < this->size; ++ctr)
        {
            this->vec[ctr] = second.vec[ctr];
        }
    }

    ~vruiVec();

    double &operator[](int index)
    {
        return vec[index];
    }

    double &operator[](int index) const
    {
        return vec[index];
    }

    vruiVec &normalize()
    {

        double sum = 0.0;
        for (int ctr = 0; ctr < this->size; ++ctr)
        {
            sum += vec[ctr] * vec[ctr];
        }

        sum = sqrt(sum);

        for (int ctr = 0; ctr < this->size; ++ctr)
        {
            vec[ctr] /= sum;
        }

        return *this;
    }

    int size;
    double *vec;
};

vruiVec operator-(const vruiVec &first, const vruiVec &second);
}
#endif
