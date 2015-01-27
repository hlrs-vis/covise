/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coSphere.h"

#include <do/coDoIntArr.h>
#include <do/coDoData.h>

using namespace covise;

coSphere::coSphere()
{
    int b;
    for (b = 0; b < 5; b++)
    {
        gcos_72_[b] = gcos((float)b * 72);
        gsin_72_[b] = gsin((float)b * 72);
        gcos_72_36_[b] = gcos((float)b * 72 + 36);
        gsin_72_36_[b] = gsin((float)b * 72 + 36);
    }
    gsin45_ = gsin(45);

    buildOneUnitSphere();
}

void
coSphere::buildOneUnitSphere()
{
    int b;
    for (b = 0; b < 5; b++)
    {
        xSphere_[b] = gcos_72_[b];
        ySphere_[b] = gsin_72_[b];
        zSphere_[b] = 0.0;
        normalsOutX_[b] = gcos_72_[b];
        normalsOutY_[b] = gsin_72_[b];
        normalsOutZ_[b] = 0.0;

        xSphere_[b + 5] = gcos_72_36_[b] * gsin45_;
        ySphere_[b + 5] = gsin_72_36_[b] * gsin45_;
        zSphere_[b + 5] = gsin45_;
        normalsOutX_[b + 5] = gcos_72_36_[b] * gsin45_;
        normalsOutY_[b + 5] = gsin_72_36_[b] * gsin45_;
        normalsOutZ_[b + 5] = gsin45_;

        xSphere_[b + 10] = gcos_72_36_[b] * gsin45_;
        ySphere_[b + 10] = gsin_72_36_[b] * gsin45_;
        zSphere_[b + 10] = -gsin45_;
        normalsOutX_[b + 10] = gcos_72_36_[b] * gsin45_;
        normalsOutY_[b + 10] = gsin_72_36_[b] * gsin45_;
        normalsOutZ_[b + 10] = -gsin45_;
    }
    xSphere_[15] = 0.0;
    ySphere_[15] = 0.0;
    zSphere_[15] = 1.0;
    normalsOutX_[15] = 0.0;
    normalsOutY_[15] = 0.0;
    normalsOutZ_[15] = 1.0;

    xSphere_[16] = 0.0;
    ySphere_[16] = 0.0;
    zSphere_[16] = -1.0;
    normalsOutX_[16] = 0.0;
    normalsOutY_[16] = 0.0;
    normalsOutZ_[16] = -1.0;

    // triangle strips
    tsl_[0] = 0;
    tsl_[1] = 12;
    tsl_[2] = 12 + 12;
    tsl_[3] = 12 + 12 + 5;
    tsl_[4] = 12 + 12 + 5 + 4;
    tsl_[5] = 12 + 12 + 5 + 4 + 5;

    // vertex list
    vl_[0] = 5;
    vl_[1] = 1;
    vl_[2] = 6;
    vl_[3] = 2;
    vl_[4] = 7;
    vl_[5] = 3;
    vl_[6] = 8;
    vl_[7] = 4;
    vl_[8] = 9;
    vl_[9] = 0;
    vl_[10] = 5;
    vl_[11] = 1;

    vl_[12] = 0;
    vl_[13] = 10;
    vl_[14] = 1;
    vl_[15] = 11;
    vl_[16] = 2;
    vl_[17] = 12;
    vl_[18] = 3;
    vl_[19] = 13;
    vl_[20] = 4;
    vl_[21] = 14;
    vl_[22] = 0;
    vl_[23] = 10;

    vl_[24] = 9;
    vl_[25] = 5;
    vl_[26] = 15;
    vl_[27] = 6;
    vl_[28] = 7;

    vl_[29] = 9;
    vl_[30] = 15;
    vl_[31] = 8;
    vl_[32] = 7;

    vl_[33] = 12;
    vl_[34] = 11;
    vl_[35] = 16;
    vl_[36] = 10;
    vl_[37] = 14;

    vl_[38] = 12;
    vl_[39] = 16;
    vl_[40] = 13;
    vl_[41] = 14;
}

inline void addTessel(float *array, float num)
{
    for (int i = 0; i < coSphere::TESSELATION; i++)
    {
        array[i] += num;
    }
}

inline void prodTessel(float *array, float num)
{
    for (int i = 0; i < coSphere::TESSELATION; i++)
    {
        array[i] *= num;
    }
}

void
coSphere::computeSpheres(float scale, float radius, float *dataIn[3],
                         float *xSpheres, float *ySpheres, float *zSpheres, float *xPoints, float *yPoints, float *zPoints,
                         float *normalsOut[3], int *tsl, int *vl, int numPoints, float *dataOut, float *colorIn, int hasColor, int hasData)
{

    switch (hasData)
    {
    case 0:
        scaleSphere(radius);

        useScaledSphere(xScaleSphere_, yScaleSphere_, zScaleSphere_, NULL, 0.0, xSpheres, ySpheres, zSpheres, xPoints, yPoints, zPoints, normalsOut, tsl, vl, numPoints, dataOut, colorIn, hasColor);
        break;
    case 1:
        useScaledSphere(xSphere_, ySphere_, zSphere_, dataIn[0], scale,
                        xSpheres, ySpheres, zSpheres, xPoints, yPoints, zPoints,
                        normalsOut, tsl, vl, numPoints, dataOut, colorIn, hasColor);
        break;
    case 2:
        scaleSphere(radius);

        useScaledSphere(xScaleSphere_, yScaleSphere_, zScaleSphere_, NULL, 0.0, xSpheres, ySpheres, zSpheres, xPoints, yPoints, zPoints, normalsOut, tsl, vl, numPoints, dataOut, colorIn, hasColor);

        deformSpheres(numPoints, dataIn, xPoints, yPoints, zPoints, xSpheres, ySpheres, zSpheres, normalsOut, scale);
        break;
    default:
        cerr << "This is a bug" << endl;
    }
}

coDistributedObject *
coSphere::amplifyData(const coDistributedObject *data_in, const coObjInfo &newObj, int numPoints)
{
    coDistributedObject *data_out = NULL;
    if (const coDoAbstractData *d_in = dynamic_cast<const coDoAbstractData *>(data_in))
    {
        coDoAbstractData *d_out = d_in->cloneType(newObj, TESSELATION * numPoints);
        data_out = d_out;
        int c = 0;
        for (int i = 0; i < numPoints; i++)
        {
            for (int b = 0; b < TESSELATION; b++, c++)
                d_out->cloneValue(c, d_in, i);
        }
    }
    else if (const coDoIntArr *intarr_in = dynamic_cast<const coDoIntArr *>(data_in))
    {
        int numDims = intarr_in->getNumDimensions();
        if (numDims != 1)
        {
            fprintf(stderr, "IntArr with dims>1 not supported by amplifyData in coSphere\n");
            data_out = NULL;
        }
        else
        {
            int dim = numPoints * TESSELATION;
            coDoIntArr *intarr_out = new coDoIntArr(newObj, 1, &dim);
            data_out = intarr_out;
            int *alt = intarr_in->getAddress();
            int *neu = intarr_out->getAddress();
            int c = 0;
            for (int i = 0; i < numPoints; i++)
            {
                for (int b = 0; b < TESSELATION; b++)
                {
                    neu[c] = alt[i];
                    c++;
                }
            }
        }
    }
    else
    {
        fprintf(stderr, "Data type not supported by amplifyData in coSphere\n");
        data_out = NULL;
    }
    return data_out;
}

void
coSphere::useScaledSphere(const float *sourceX, const float *sourceY, const float *sourceZ,
                          const float *data_in, float radscal,
                          float *xSpheres, float *ySpheres, float *zSpheres, float *xPoints, float *yPoints, float *zPoints,
                          float *normalsOut[3], int *tsl, int *vl, int numPoints, float *dataOut, float *colorIn, int hasColor)
{
    int c, v;

    c = v = 0;
    float *thisXSphere = xSpheres;
    float *thisYSphere = ySpheres;
    float *thisZSphere = zSpheres;
    float *thisNormalsOutX = normalsOut[0];
    float *thisNormalsOutY = normalsOut[1];
    float *thisNormalsOutZ = normalsOut[2];
    int *thisTsl = tsl;
    int *thisVl = vl;
    if (!data_in)
    {
        for (int i = 0; i < numPoints; i++, thisXSphere += TESSELATION, thisYSphere += TESSELATION, thisZSphere += TESSELATION,
                 thisNormalsOutX += TESSELATION, thisNormalsOutY += TESSELATION, thisNormalsOutZ += TESSELATION,
                 thisTsl += 6, v += 42, thisVl += 42, c += TESSELATION)
        {
            // coordinates&normals
            memcpy(thisXSphere, sourceX, TESSELATION * sizeof(float));
            memcpy(thisYSphere, sourceY, TESSELATION * sizeof(float));
            memcpy(thisZSphere, sourceZ, TESSELATION * sizeof(float));
            memcpy(thisNormalsOutX, normalsOutX_, TESSELATION * sizeof(float));
            memcpy(thisNormalsOutY, normalsOutY_, TESSELATION * sizeof(float));
            memcpy(thisNormalsOutZ, normalsOutZ_, TESSELATION * sizeof(float));

            memcpy(thisTsl, tsl_, 6 * sizeof(int));
            for (int j = 0; j < 6; j++)
            {
                thisTsl[j] += v;
            }

            memcpy(thisVl, vl_, 42 * sizeof(int));
            for (int j = 0; j < 42; j++)
            {
                thisVl[j] += c;
            }

            addTessel(thisXSphere, xPoints[i]);
            addTessel(thisYSphere, yPoints[i]);
            addTessel(thisZSphere, zPoints[i]);
        }
    }
    else
    {
        for (int i = 0; i < numPoints; i++, thisXSphere += TESSELATION, thisYSphere += TESSELATION, thisZSphere += TESSELATION,
                 thisNormalsOutX += TESSELATION, thisNormalsOutY += TESSELATION, thisNormalsOutZ += TESSELATION,
                 thisTsl += 6, v += 42, thisVl += 42, c += TESSELATION)
        {
            // coordinates&normals
            memcpy(thisXSphere, sourceX, TESSELATION * sizeof(float));
            memcpy(thisYSphere, sourceY, TESSELATION * sizeof(float));
            memcpy(thisZSphere, sourceZ, TESSELATION * sizeof(float));
            memcpy(thisNormalsOutX, normalsOutX_, TESSELATION * sizeof(float));
            memcpy(thisNormalsOutY, normalsOutY_, TESSELATION * sizeof(float));
            memcpy(thisNormalsOutZ, normalsOutZ_, TESSELATION * sizeof(float));
            memcpy(thisTsl, tsl_, 6 * sizeof(int));
            for (int j = 0; j < 6; j++)
            {
                thisTsl[j] += v;
            }

            memcpy(thisVl, vl_, 42 * sizeof(int));
            for (int j = 0; j < 42; j++)
            {
                thisVl[j] += c;
            }

            float prod = radscal * data_in[i];
            prodTessel(thisXSphere, prod);
            prodTessel(thisYSphere, prod);
            prodTessel(thisZSphere, prod);
            addTessel(thisXSphere, xPoints[i]);
            addTessel(thisYSphere, yPoints[i]);
            addTessel(thisZSphere, zPoints[i]);
        }
    }
    // the data
    if (hasColor && dataOut && colorIn)
    {
        c = 0;
        for (int i = 0; i < numPoints; i++)
        {
            for (int b = 0; b < TESSELATION; b++, c++)
                dataOut[c] = colorIn[i]; //dataIn[0][i];
        }
    }
}

void
coSphere::scaleSphere(float radius)
{
    int i;
    memcpy(xScaleSphere_, xSphere_, TESSELATION * sizeof(float));
    memcpy(yScaleSphere_, ySphere_, TESSELATION * sizeof(float));
    memcpy(zScaleSphere_, zSphere_, TESSELATION * sizeof(float));
    for (i = 0; i < TESSELATION; i++)
    {
        xScaleSphere_[i] *= radius;
        yScaleSphere_[i] *= radius;
        zScaleSphere_[i] *= radius;
    }
}

void coSphere::deformSpheres(int numPoints, float *dataIn[3], float *xPoints, float *yPoints, float *zPoints,
                             float *xSpheres, float *ySpheres, float *zSpheres, float *normalsOut[3], float scale)
{
    int i;
    int b;
    float s;
    float x, y, z;
    float dx, dy, dz;
    float u, v, w, mag;
    float d;

    for (i = 0; i < numPoints; i++)
    {
        u = dataIn[0][i];
        v = dataIn[1][i];
        w = dataIn[2][i];
        x = xPoints[i];
        y = yPoints[i];
        z = zPoints[i];
        mag = sqrt(u * u + v * v + w * w);

        for (b = 0; b < TESSELATION; b++)
        {
            dx = xSpheres[(i * TESSELATION) + b] - x;
            dy = ySpheres[(i * TESSELATION) + b] - y;
            dz = zSpheres[(i * TESSELATION) + b] - z;

            // compute distance from point to plane
            s = sqrt(dx * dx + dy * dy + dz * dz);
            d = ((dx / s) * (u / mag) + (dy / s) * (v / mag) + (dz / s) * (w / mag));

            // compute scale-factor
            s = d * scale * mag;

            // deform
            xSpheres[(i * TESSELATION) + b] = x + dx + u * s;
            ySpheres[(i * TESSELATION) + b] = y + dy + v * s;
            zSpheres[(i * TESSELATION) + b] = z + dz + w * s;

            // and the normals (looks great without adjusted normals)
            normalsOut[0][(i * TESSELATION) + b] -= u * s;
            normalsOut[1][(i * TESSELATION) + b] -= v * s;
            normalsOut[2][(i * TESSELATION) + b] -= w * s;
        }
    }
}
