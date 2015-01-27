/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "Vessels.h"

#include <string.h>

//////
////// we must provide main to init covise
//////

int main(int argc, char *argv[])
{
    // init
    Vessels *application = new Vessels(argc, argv);

    // and back to covise
    application->run();

    // done
    return (0);
}

coDistributedObject **Vessels::compute(coDistributedObject **in, char **outNames)
{
    coDistributedObject **returnObject = NULL;
    coDoUniformGrid *gridInObj = NULL;
    coDoFloat *dataInObj = NULL;
    coDoFloat *dataOutObj = NULL;

    // get parameters
    Covise::get_scalar_param("value", &value);
    Covise::get_vector_param("pointinside", 0, &pointInside[0]);
    Covise::get_vector_param("pointinside", 1, &pointInside[1]);
    Covise::get_vector_param("pointinside", 2, &pointInside[2]);

    Covise::get_scalar_param("shrink_times", &shrinkTimes);
    Covise::get_scalar_param("blow_times", &blowTimes);
    Covise::get_choice_param("filter1", &filter1);

    // get input objects
    gridInObj = (coDoUniformGrid *)in[0];
    dataInObj = (coDoFloat *)in[1];

    // get input data
    dataInObj->getGridSize(&numX, &numY, &numZ);
    dataInObj->getAddress(&dataIn);

    // compute point

    // das hier sind werte die manuell ermittelt wurden
    point[0] = 141;
    point[1] = 335;
    point[2] = 36;

    // prepare output data
    dataOutObj = new coDoFloat(outNames[0], numX, numY, numZ);
    dataOutObj->getAddress(&dataOut);
    returnObject = new coDistributedObject *[1];
    returnObject[0] = dataOutObj;
    dataBfr = new float[numX * numY * numZ];

    // build output
    buildOutput();

    // done
    delete[] dataBfr;
    return (returnObject);
}

void Vessels::buildOutput()
{
    int i;

    blackOrWhite_filter(dataBfr, dataIn, value);
    switch (filter1)
    {
    case 2: // blow'n'shrink
        for (i = 0; i < blowTimes; i++)
        {
            blow_filter(dataOut, dataBfr);
            copyBuffer(dataBfr, dataOut);
        }
        for (i = 0; i < shrinkTimes; i++)
        {
            shrink_filter(dataOut, dataBfr);
            copyBuffer(dataBfr, dataOut);
        }
        break;
    default: // shrink'n'blow
        for (i = 0; i < shrinkTimes; i++)
        {
            shrink_filter(dataOut, dataBfr);
            copyBuffer(dataBfr, dataOut);
        }
        for (i = 0; i < blowTimes; i++)
        {
            blow_filter(dataOut, dataBfr);
            copyBuffer(dataBfr, dataOut);
        }
    }
}

void Vessels::copyBuffer(float *tgt, const float *src)
{
    memcpy(tgt, src, (numX * numY * numZ) * sizeof(float));
    return;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void Vessels::blackOrWhite_filter(float *tgt, const float *src, const float t)
{
    int i, n;
    n = numX * numY * numZ;
    for (i = 0; i < n; i++)
    {
        if (src[i] >= t)
            tgt[i] = 1.0;
        else
            tgt[i] = 0.0;
    }
    return;
}

void Vessels::shrink_filter(float *tgt, const float *src)
{
    int x, y, z;
    int o, n;
    int nYZ = numY * numZ;

    // reset
    memset(tgt, 0, (numX * nYZ) * sizeof(float));

    // ok
    for (z = 0; z < numZ; z++)
    {
        for (y = 1; y < numY - 1; y++)
        {
            for (x = 1; x < numX - 1; x++)
            {
                n = x * nYZ + y * numZ + z;
                // untere reihe pruefen
                o = n - numZ;
                if (src[o - nYZ] && src[o] && src[o + nYZ])
                {
                    // und eigene reihe pruefen
                    if (src[n - nYZ] && src[n + nYZ])
                    {
                        // obere reihe
                        o = n + numZ;
                        if (src[o - nYZ] && src[o] && src[o + nYZ])
                        {
                            // alle umgebenden punkte sind 1, also nicht
                            // shrinken d.h. dieser punkt ist auch 1
                            // (er bleibt 0, wenn einer der umgebenden auch 0 ist)
                            tgt[n] = 1.0;
                        }
                    }
                }
            }
        }
    }

    return;
}

void Vessels::blow_filter(float *tgt, const float *src)
{
    int x, y, z;
    int o, n;
    int nYZ = numY * numZ;

    // reset
    memset(tgt, 0, (numX * nYZ) * sizeof(float));

    // ok
    for (z = 0; z < numZ; z++)
    {
        for (y = 1; y < numY - 1; y++)
        {
            for (x = 1; x < numX - 1; x++)
            {
                n = x * nYZ + y * numZ + z;
                // untere reihe pruefen
                o = n - numZ;
                if (src[o - nYZ] || src[o] || src[o + nYZ])
                    tgt[n] = 1.0;
                else
                {
                    // und eigene reihe pruefen
                    if (src[n - nYZ] || src[n + nYZ])
                        tgt[n] = 1.0;
                    else
                    {
                        // obere reihe
                        o = n + numZ;
                        if (src[o - nYZ] || src[o] || src[o + nYZ])
                            tgt[n] = 1.0;
                    }
                }
            }
        }
    }

    return;
}
