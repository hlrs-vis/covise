/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Point.h"
#include <stdlib.h>

Q *
Factory::NewQs(int no_data, const float *pt0, const float *pt1, const float *pt2)
{
    switch (no_data)
    {
    case 0: // geometry only
        return (new Qs<0>(pt0, pt1, pt2));
    case 1: // geometry and scalar data
        return (new Qs<1>(pt0, pt1, pt2));
    case 3: // geometry and vector data
        return (new Qs<3>(pt0, pt1, pt2));
    default:
        abort();
    }
    return NULL;
}

Q *
Factory::NewQs(int no_data, const double *pt0, const double *pt1, const double *pt2)
{
    switch (no_data)
    {
    case 0: // geometry only
        return (new Qs<0>(pt0, pt1, pt2));
    case 1: // geometry and scalar data
        return (new Qs<1>(pt0, pt1, pt2));
    case 3: // geometry and vector data
        return (new Qs<3>(pt0, pt1, pt2));
    default:
        abort();
    }
    return NULL;
}

Point *
Factory::NewPointAndData(int no_data, float x, float y, float z, const float *data)
{
    switch (no_data)
    {
    case 0: // geometry only
        return (new PointAndData<0>(x, y, z, data));
    case 1: // geometry and scalar data
        return (new PointAndData<1>(x, y, z, data));
    case 3: // geometry and vector data
        return (new PointAndData<3>(x, y, z, data));
    default:
        abort();
    }
    return NULL;
}

double
ScalarProd(int dimension, const float *data0, const float *data1)
{
    int i;
    double ret = 0.0;
    for (i = 0; i < dimension; ++i)
    {
        ret += data0[i] * data1[i];
    }
    return ret;
}

double
ScalarProd(int dimension, const double *data0, const double *data1)
{
    int i;
    double ret = 0.0;
    for (i = 0; i < dimension; ++i)
    {
        ret += data0[i] * data1[i];
    }
    return ret;
}

void
vect_prod(float *normal, const float *e0, const float *e1)
{
    normal[0] = e0[1] * e1[2] - e0[2] * e1[1];
    normal[1] = e0[2] * e1[0] - e0[0] * e1[2];
    normal[2] = e0[0] * e1[1] - e0[1] * e1[0];
}

void
vect_prod(double *normal, const double *e0, const double *e1)
{
    normal[0] = e0[1] * e1[2] - e0[2] * e1[1];
    normal[1] = e0[2] * e1[0] - e0[0] * e1[2];
    normal[2] = e0[0] * e1[1] - e0[1] * e1[0];
}

bool
Normalise(float *normal)
{
    double dnormal[3];
    dnormal[0] = normal[0];
    dnormal[1] = normal[1];
    dnormal[2] = normal[2];
    double len = ScalarProd(3, dnormal, dnormal);
    len = sqrt(len);
    if (len == 0.0)
    {
        return false;
    }
    len = 1.0 / len;
    dnormal[0] *= len;
    dnormal[1] *= len;
    dnormal[2] *= len;
    normal[0] = (float)dnormal[0];
    normal[1] = (float)dnormal[1];
    normal[2] = (float)dnormal[2];
    return true;
}

bool
Normalise(float *normal, int dim)
{
	double dnormal[32]; // FIXME
	for (int i = 0; i < dim; i++)
		dnormal[i] = normal[i];
	double len = ScalarProd(dim, dnormal, dnormal);
	len = sqrt(len);
	if (len == 0.0)
	{
		return false;
	}
	len = 1.0 / len;
	int i;
	for (i = 0; i < dim; ++i)
	{
		dnormal[i] *= len;
		normal[i] = float(dnormal[i]);
	}
    return true;
}
