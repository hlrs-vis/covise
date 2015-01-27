/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SPHEREDATA_PLUGIN_H
#define SPHEREDATA_PLUGIN_H
#include <assert.h>
struct SphereData
{
    int n;
    float *x;
    float *y;
    float *z;
    float *r;

    SphereData()
        : n(0)
        , x(NULL)
        , y(NULL)
        , z(NULL)
        , r(NULL){};

    SphereData(int size, float *x, float *y, float *z, float *r)
    {
        /*      this->n = size;
      this->x = x;
      this->y = y;
      this->z = z;
      this->r = r;*/
        setData(size, x, y, z, r);
    }

    SphereData(const SphereData &data)
    {
        setData(data.n, data.x, data.y, data.z, data.r);
    }

    SphereData &operator=(const SphereData &data)
    {
        setData(data.n, data.x, data.y, data.z, data.r);
        return *this;
    }

    void setData(int size, float *x, float *y, float *z, float *r)
    {
        this->n = size;
        this->x = new float[size];
        this->y = new float[size];
        this->z = new float[size];
        this->r = new float[size];

        memcpy(this->x, x, size * sizeof(float));
        memcpy(this->y, y, size * sizeof(float));
        memcpy(this->z, z, size * sizeof(float));
        memcpy(this->r, r, size * sizeof(float));
    };

    ~SphereData()
    {
        delete[] this->x;
        delete[] this->y;
        delete[] this->z;
        delete[] this->r;
    };
};
#endif
