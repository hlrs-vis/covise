/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SPHERE_H
#define CO_SPHERE_H

#include <util/coExport.h>
#include <sysdep/math.h>
#include <do/coDoSpheres.h>

namespace covise
{

class ALGEXPORT coSphere
{
public:
    enum
    {
        TESSELATION = 17
    };

private:
    float gcos_72_[5];
    float gcos_72_36_[5];
    float gsin_72_[5];
    float gsin_72_36_[5];
    float gsin45_;

    // a unit sphere
    float xSphere_[TESSELATION];
    float ySphere_[TESSELATION];
    float zSphere_[TESSELATION];
    float normalsOutX_[TESSELATION];
    float normalsOutY_[TESSELATION];
    float normalsOutZ_[TESSELATION];
    int tsl_[6];
    int vl_[42];

    float xScaleSphere_[TESSELATION];
    float yScaleSphere_[TESSELATION];
    float zScaleSphere_[TESSELATION];

    float gsin(float angle)
    {
        return ((float)(sin((angle * M_PI) / 180.0)));
    };
    float gcos(float angle)
    {
        return ((float)(cos((angle * M_PI) / 180.0)));
    };
    void buildOneUnitSphere();

public:
    coSphere();
    void deformSpheres(int numPoints, float *dataIn[3], float *xPoints, float *yPoints, float *zPoints, float *xSpheres, float *ySpheres, float *zSpheres, float *normalsOut[3], float scale);

    void scaleSphere(float radius);

    void useScaledSphere(const float *sourceX, const float *sourceY, const float *sourceZ, const float *data_in, float radscal, float *xSpheres, float *ySpheres, float *zSpheres, float *xPoints, float *yPoints, float *zPoints, float *normalsOut[3], int *tsl, int *vl, int numPoints, float *dataOut, float *colorIn, int hasColor);

    void computeSpheres(float scale, float radius, float *data_in[3], float *xSpheres, float *ySpheres, float *zSpheres, float *xPoints, float *yPoints, float *zPoints, float *normalsOut[3], int *tsl, int *vl, int numPoints, float *dataOut, float *colorIn, int hasColor, int hasData);
    coDistributedObject *amplifyData(const coDistributedObject *, const coObjInfo &newObj, int);
};
}
#endif
