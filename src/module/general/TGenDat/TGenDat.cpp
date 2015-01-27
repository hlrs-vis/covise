/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TGenDat.h"

#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

// transient1: singleblock+TIMESTEP data, transient2: multiblock+TIMESTEP data
// moving: TIMESTEP grid+data

TGenDat::TGenDat(int argc, char *argv[])
    : coModule(argc, argv, "Generate timestep data")
{
    oPGrid = addOutputPort("grid", "UnstructuredGrid|StructuredGrid|RectilinearGrid", "the generated grid");
    oPData = addOutputPort("data", "Vec3|Vec3", "the generated data");

    static const char *dGridChoice[] = { "rectilinear", "structured", "unstructured" };
    pGridChoice = addChoiceParam("gridType", "selects which type of grid to generate");
    pGridChoice->setValue(3, dGridChoice, 0);

    static const char *dStyleChoice[] = { "circular", "multiblock", "transient1", "transient2", "moving", "weird" };
    pStyleChoice = addChoiceParam("case", "selects which testcase should be generated");
    pStyleChoice->setValue(6, dStyleChoice, 0);

    // done
    return;
}

TGenDat::~TGenDat()
{
    // dummy
}

int TGenDat::compute(const char *)
{
    switch (pGridChoice->getValue())
    {
    case 0: // rectilinear
        rectGrid();
        break;
    case 1: // structured
        strGrid();
        break;
    case 2: // unstructured
        unstrGrid();
        break;
    default:
        Covise::sendInfo("invalid gridType selected. no output generated.");
    }

    /*
         sprintf(bfr, "%s_%d", outPort1->getObjName(), i);
         points = new coDoPoints(bfr, o);
         sprintf(bfr, "%s_%d", outPort3->getObjName(), i);
         id = new coDoFloat(bfr, o);
         points->getAddresses(&x, &y, &z);
         points->addAttribute( "COLOR", "blue" );
         id->getAddress(&d);
   */

    // done
    return (coModule::SUCCESS);
}

void TGenDat::rectGrid()
{
    switch (pStyleChoice->getValue())
    {
    case 0: // circular
    {
        float *xCoord, *yCoord, *zCoord;
        float *uData, *vData, *wData;
        int numX, numY, numZ, x, y, z;

        numX = 20;
        numY = 24;
        numZ = 17;

        coDoRectilinearGrid *grid = genRectGrid(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, numX, numY, numZ, 3214, (char *)oPGrid->getObjName());
        grid->getAddresses(&xCoord, &yCoord, &zCoord);

        uData = new float[numX * numY * numZ];
        vData = new float[numX * numY * numZ];
        wData = new float[numX * numY * numZ];

        for (x = 0; x < numX; x++)
        {
            for (y = 0; y < numY; y++)
            {
                for (z = 0; z < numZ; z++)
                {
                    if (yCoord[y] > 0.5)
                    {
                        wData[x * numY * numZ + y * numZ + z] = xCoord[x] + zCoord[z] * (yCoord[y] - 0.5f);
                        uData[x * numY * numZ + y * numZ + z] = -zCoord[z] + xCoord[x] * (yCoord[y] - 0.5f);
                        vData[x * numY * numZ + y * numZ + z] = 0.0f;
                    }
                    else if (yCoord[y] < -0.5)
                    {
                        wData[x * numY * numZ + y * numZ + z] = xCoord[x] + zCoord[z] * (yCoord[y] + 0.5f);
                        uData[x * numY * numZ + y * numZ + z] = -zCoord[z] + xCoord[x] * (yCoord[y] + 0.5f);
                        vData[x * numY * numZ + y * numZ + z] = 0.0f;
                    }
                    else
                    {
                        wData[x * numY * numZ + y * numZ + z] = xCoord[x];
                        uData[x * numY * numZ + y * numZ + z] = -zCoord[z];
                        vData[x * numY * numZ + y * numZ + z] = 0.0f;
                    }
                }
            }
        }

        coDoVec3 *data = new coDoVec3(oPData->getObjName(), numX * numY * numZ, uData, vData, wData);

        delete[] uData;
        delete[] vData;
        delete[] wData;
        delete data;
        delete grid;
    }
    break;
    case 1: // multiblock
    {
        coDistributedObject **setGObjs = NULL, **setDObjs = NULL;
        char *objName, *elemName;
        int i;

        setGObjs = new coDistributedObject *[6];
        setDObjs = new coDistributedObject *[6];

        float *xCoord, *yCoord, *zCoord;
        float *uData, *vData, *wData;
        int numX, numY, numZ, x, y, z;

        coDoRectilinearGrid *grid;
        coDoVec3 *data;

        for (i = 0; i < 5; i++)
        {
            objName = (char *)oPGrid->getObjName();
            elemName = new char[strlen(objName) + 3];
            sprintf(elemName, "%s_%d", objName, i);

            switch (i)
            {
            case 0:
                numX = 20;
                numY = 24;
                numZ = 16;
                grid = genRectGrid(1.0, 1.0, 0.0, -1.0, 0.0, 1.0, numX, numY, numZ, 3214, elemName);
                break;
            case 1:
                numX = 10;
                numY = 6;
                numZ = 8;
                grid = genRectGrid(0.0, 1.0, 0.0, 1.0, 0.0, -1.0, numX, numY, numZ, 637, elemName);
                break;
            case 2:
                numX = 8;
                numY = 30;
                numZ = 12;
                grid = genRectGrid(1.0, 0.0, 1.0, 0.0, -1.0, -1.0, numX, numY, numZ, 843, elemName);
                break;
            case 3:
                numX = 15;
                numY = 24;
                numZ = 6;
                grid = genRectGrid(-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, numX, numY, numZ, 12, elemName);
                break;
            case 4:
                numX = 8;
                numY = 14;
                numZ = 6;
                grid = genRectGrid(-1.0, 1.0, 0.0, 0.0, -1.0, -1.0, numX, numY, numZ, 96, elemName);
                break;
            default:
                Covise::sendInfo("invalid case selected. no output generated.");
                return;
                break;
            }
            delete[] elemName;
            grid->getAddresses(&xCoord, &yCoord, &zCoord);

            uData = new float[numX * numY * numZ];
            vData = new float[numX * numY * numZ];
            wData = new float[numX * numY * numZ];

            for (x = 0; x < numX; x++)
            {
                for (y = 0; y < numY; y++)
                {
                    for (z = 0; z < numZ; z++)
                    {
                        if (yCoord[y] > 0.5)
                        {
                            wData[x * numY * numZ + y * numZ + z] = xCoord[x] + zCoord[z] * (yCoord[y] - 0.5f);
                            uData[x * numY * numZ + y * numZ + z] = -zCoord[z] + xCoord[x] * (yCoord[y] - 0.5f);
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                        else if (yCoord[y] < -0.5)
                        {
                            wData[x * numY * numZ + y * numZ + z] = xCoord[x] + zCoord[z] * (yCoord[y] + 0.5f);
                            uData[x * numY * numZ + y * numZ + z] = -zCoord[z] + xCoord[x] * (yCoord[y] + 0.5f);
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                        else
                        {
                            wData[x * numY * numZ + y * numZ + z] = xCoord[x];
                            uData[x * numY * numZ + y * numZ + z] = -zCoord[z];
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                    }
                }
            }

            objName = (char *)oPData->getObjName();
            elemName = new char[strlen(objName) + 3];
            sprintf(elemName, "%s_%d", objName, i);

            data = new coDoVec3(elemName, numX * numY * numZ, uData, vData, wData);

            delete[] elemName;
            delete[] uData;
            delete[] vData;
            delete[] wData;

            setGObjs[i] = grid;
            setDObjs[i] = data;

            //delete data;
            //delete grid;
        }

        setGObjs[i] = NULL;
        setDObjs[i] = NULL;

        coDoSet *tSet = new coDoSet(oPGrid->getObjName(), setGObjs);
        for (i = 0; i < 5; i++)
            delete setGObjs[i];
        delete[] setGObjs;
        delete tSet;

        tSet = new coDoSet(oPData->getObjName(), setDObjs);
        for (i = 0; i < 5; i++)
            delete setDObjs[i];
        delete[] setDObjs;
        delete tSet;
    }

    break;

    case 2: // transient1 (static grid, timestep data)
    {
        coDistributedObject **setDObjs = NULL;
        char *objName, *elemName;
        int i;
        setDObjs = new coDistributedObject *[33];

        float *xCoord, *yCoord, *zCoord;
        float *uData, *vData, *wData;
        int numX, numY, numZ, x, y, z;
        float u, v, w;

        numX = 20;
        numY = 24;
        numZ = 17;

        coDoRectilinearGrid *grid = genRectGrid(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, numX, numY, numZ, 3214, (char *)oPGrid->getObjName());
        grid->getAddresses(&xCoord, &yCoord, &zCoord);

        coDoVec3 *data = NULL;
        float phase = 0.0, phStep = (float)(8.0 * M_PI) / 64.0;

        for (i = 0; i < 32; i++)
        {
            uData = new float[numX * numY * numZ];
            vData = new float[numX * numY * numZ];
            wData = new float[numX * numY * numZ];

            for (x = 0; x < numX; x++)
            {
                for (y = 0; y < numY; y++)
                {
                    for (z = 0; z < numZ; z++)
                    {

                        // overlayed with orthogonal sine
                        u = xCoord[x];
                        v = -zCoord[z];
                        w = (float)sin(phase) * 0.3f;

                        wData[x * numY * numZ + y * numZ + z] = u;
                        uData[x * numY * numZ + y * numZ + z] = v;
                        vData[x * numY * numZ + y * numZ + z] = w;
                    }
                }
            }
            phase += phStep;

            objName = (char *)oPData->getObjName();
            elemName = new char[strlen(objName) + 3];
            sprintf(elemName, "%s_%d", objName, i);

            data = new coDoVec3(elemName, numX * numY * numZ, uData, vData, wData);

            delete[] elemName;
            delete[] uData;
            delete[] vData;
            delete[] wData;

            setDObjs[i] = data;
        }

        setDObjs[i] = NULL;

        coDoSet *tSet = new coDoSet(oPData->getObjName(), setDObjs);
        tSet->addAttribute("TIMESTEP", "1 32");
        for (i = 0; i < 32; i++)
            delete setDObjs[i];
        delete[] setDObjs;
        delete tSet;
    }

    break;

    case 3: // transient2 (multiblock grid+timestep data)
    {
        // for debugging we're generating timestep singleblock grid+data (to validate
        //   velocityfield for case 3)
        coDistributedObject **setDObjs = NULL, **setGObjs = NULL;
        char *objName, *elemName;
        int i;
        setDObjs = new coDistributedObject *[33];
        setGObjs = new coDistributedObject *[33];

        float *xCoord, *yCoord, *zCoord;
        float *uData, *vData, *wData;
        int numX, numY, numZ, x, y, z;
        float u, v, w;

        numX = 20;
        numY = 24;
        numZ = 17;

        coDoRectilinearGrid *grid = NULL;

        coDoVec3 *data = NULL;
        float phase = 0.0, phStep = (float)((8.0 * M_PI) / 64.0);

        for (i = 0; i < 32; i++)
        {
            objName = (char *)oPGrid->getObjName();
            elemName = new char[strlen(objName) + 4];
            sprintf(elemName, "%s_%d", objName, i);

            grid = genRectGrid(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, numX, numY, numZ, 3214 + i * 2, elemName);
            //grid = genRectGrid(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, numX, numY, numZ, 3214, elemName);
            grid->getAddresses(&xCoord, &yCoord, &zCoord);
            delete[] elemName;

            uData = new float[numX * numY * numZ];
            vData = new float[numX * numY * numZ];
            wData = new float[numX * numY * numZ];

            for (x = 0; x < numX; x++)
            {
                for (y = 0; y < numY; y++)
                {
                    for (z = 0; z < numZ; z++)
                    {

                        // overlayed with orthogonal sine
                        u = xCoord[x];
                        v = -zCoord[z];
                        w = (float)(sin(phase) * 0.3);

                        wData[x * numY * numZ + y * numZ + z] = u;
                        uData[x * numY * numZ + y * numZ + z] = v;
                        vData[x * numY * numZ + y * numZ + z] = w;
                    }
                }
            }
            phase += phStep;

            objName = (char *)oPData->getObjName();
            elemName = new char[strlen(objName) + 4];
            sprintf(elemName, "%s_%d", objName, i);

            data = new coDoVec3(elemName, numX * numY * numZ, uData, vData, wData);

            delete[] elemName;
            delete[] uData;
            delete[] vData;
            delete[] wData;

            setDObjs[i] = data;
            setGObjs[i] = grid;
        }

        setDObjs[i] = NULL;
        setGObjs[i] = NULL;

        coDoSet *tSet = new coDoSet(oPData->getObjName(), setDObjs);
        tSet->addAttribute("TIMESTEP", "1 32");
        for (i = 0; i < 32; i++)
            delete setDObjs[i];
        delete[] setDObjs;
        delete tSet;

        tSet = new coDoSet(oPGrid->getObjName(), setGObjs);
        tSet->addAttribute("TIMESTEP", "1 32");
        for (i = 0; i < 32; i++)
            delete setGObjs[i];
        delete[] setGObjs;
        delete tSet;
    }

    break;

    case 4: // moving (timestep/moving grid+data) [multiblock]
    {
        coDistributedObject **setGObjs = NULL, **setDObjs = NULL;
        char *objName, *elemName;
        int i;

        setGObjs = new coDistributedObject *[6];
        setDObjs = new coDistributedObject *[6];

        float *xCoord, *yCoord, *zCoord;
        float *uData, *vData, *wData;
        int numX, numY, numZ, x, y, z;

        coDoRectilinearGrid *grid;
        coDoVec3 *data;

        for (i = 0; i < 5; i++)
        {
            objName = (char *)oPGrid->getObjName();
            elemName = new char[strlen(objName) + 3];
            sprintf(elemName, "%s_%d", objName, i);

            switch (i)
            {
            case 0:
                numX = 20;
                numY = 24;
                numZ = 16;
                grid = genRectGrid(1.0, 1.0, 0.0, -1.0, 0.0, 1.0, numX, numY, numZ, 3214, elemName);
                break;
            case 1:
                numX = 10;
                numY = 6;
                numZ = 8;
                grid = genRectGrid(0.0, 1.0, 0.0, 1.0, 0.0, -1.0, numX, numY, numZ, 637, elemName);
                break;
            case 2:
                numX = 8;
                numY = 30;
                numZ = 12;
                grid = genRectGrid(1.0, 0.0, 1.0, 0.0, -1.0, -1.0, numX, numY, numZ, 843, elemName);
                break;
            case 3:
                numX = 15;
                numY = 24;
                numZ = 6;
                grid = genRectGrid(-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, numX, numY, numZ, 12, elemName);
                break;
            case 4:
                numX = 8;
                numY = 14;
                numZ = 6;
                grid = genRectGrid(-1.0, 1.0, 0.0, 0.0, -1.0, -1.0, numX, numY, numZ, 96, elemName);
                break;
            default:
                Covise::sendInfo("invalid case selected. no output generated.");
                return;
                break;
            }
            delete[] elemName;
            grid->getAddresses(&xCoord, &yCoord, &zCoord);

            uData = new float[numX * numY * numZ];
            vData = new float[numX * numY * numZ];
            wData = new float[numX * numY * numZ];

            for (x = 0; x < numX; x++)
            {
                for (y = 0; y < numY; y++)
                {
                    for (z = 0; z < numZ; z++)
                    {
                        if (yCoord[y] > 0.5)
                        {
                            wData[x * numY * numZ + y * numZ + z] = xCoord[x] + zCoord[z] * (yCoord[y] - 0.5f);
                            uData[x * numY * numZ + y * numZ + z] = -zCoord[z] + xCoord[x] * (yCoord[y] - 0.5f);
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                        else if (yCoord[y] < -0.5)
                        {
                            wData[x * numY * numZ + y * numZ + z] = xCoord[x] + zCoord[z] * (yCoord[y] + 0.5f);
                            uData[x * numY * numZ + y * numZ + z] = -zCoord[z] + xCoord[x] * (yCoord[y] + 0.5f);
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                        else
                        {
                            wData[x * numY * numZ + y * numZ + z] = xCoord[x];
                            uData[x * numY * numZ + y * numZ + z] = -zCoord[z];
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                    }
                }
            }

            objName = (char *)oPData->getObjName();
            elemName = new char[strlen(objName) + 3];
            sprintf(elemName, "%s_%d", objName, i);

            data = new coDoVec3(elemName, numX * numY * numZ, uData, vData, wData);

            delete[] elemName;
            delete[] uData;
            delete[] vData;
            delete[] wData;

            setGObjs[i] = grid;
            setDObjs[i] = data;

            //delete data;
            //delete grid;
        }

        setGObjs[i] = NULL;
        setDObjs[i] = NULL;

        coDoSet *tSet = new coDoSet(oPGrid->getObjName(), setGObjs);
        for (i = 0; i < 5; i++)
            delete setGObjs[i];
        delete[] setGObjs;
        delete tSet;

        tSet = new coDoSet(oPData->getObjName(), setDObjs);
        for (i = 0; i < 5; i++)
            delete setDObjs[i];
        delete[] setDObjs;
        delete tSet;
    }

    break;

    case 6: // weird (special bonus hype)

    default:
        Covise::sendInfo("invalid case selected. no output generated.");
    }

    return;
}

void TGenDat::strGrid()
{
    switch (pStyleChoice->getValue())
    {
    case 0: // circular (for testing sbc)
    {
        coDistributedObject **setGObjs = NULL, **setDObjs = NULL;
        char *objName, *elemName;
        int i;

        setGObjs = new coDistributedObject *[6];
        setDObjs = new coDistributedObject *[6];

        float *xCoord, *yCoord, *zCoord;
        float *uData, *vData, *wData;
        int numX, numY, numZ, x, y, z;

        coDoStructuredGrid *grid;
        coDoVec3 *data;

        for (i = 0; i < 5; i++)
        {
            objName = (char *)oPGrid->getObjName();
            elemName = new char[strlen(objName) + 3];
            sprintf(elemName, "%s_%d", objName, i);

            switch (i)
            {
            case 0:
                numX = 20;
                numY = 24;
                numZ = 16;
                grid = genStrGrid(1.0, 1.0, 0.0, -1.0, 0.0, 1.0, numX, numY, numZ, 3214, elemName);
                break;
            case 1:
                numX = 10;
                numY = 6;
                numZ = 8;
                grid = genStrGrid(0.0, 1.0, 0.0, 1.0, 0.0, -1.0, numX, numY, numZ, 637, elemName);
                break;
            case 2:
                numX = 8;
                numY = 30;
                numZ = 12;
                grid = genStrGrid(1.0, 0.0, 1.0, 0.0, -1.0, -1.0, numX, numY, numZ, 843, elemName);
                break;
            case 3:
                numX = 15;
                numY = 24;
                numZ = 6;
                grid = genStrGrid(-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, numX, numY, numZ, 12, elemName);
                break;
            case 4:
                numX = 8;
                numY = 14;
                numZ = 6;
                grid = genStrGrid(-1.0, 1.0, 0.0, 0.0, -1.0, -1.0, numX, numY, numZ, 96, elemName);
                break;
            default:
                Covise::sendInfo("invalid case selected. no output generated.");
                return;
                break;
            }
            delete[] elemName;
            grid->getAddresses(&xCoord, &yCoord, &zCoord);

            uData = new float[numX * numY * numZ];
            vData = new float[numX * numY * numZ];
            wData = new float[numX * numY * numZ];

            for (x = 0; x < numX; x++)
            {
                for (y = 0; y < numY; y++)
                {
                    for (z = 0; z < numZ; z++)
                    {
                        if (yCoord[x * numY * numZ + y * numZ + z] > 0.0)
                        {
                            wData[x * numY * numZ + y * numZ + z] = 1.0;
                            uData[x * numY * numZ + y * numZ + z] = 0.0;
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                        else
                        {
                            wData[x * numY * numZ + y * numZ + z] = -1.0;
                            uData[x * numY * numZ + y * numZ + z] = 0.0;
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                    }
                }
            }

            objName = (char *)oPData->getObjName();
            elemName = new char[strlen(objName) + 3];
            sprintf(elemName, "%s_%d", objName, i);

            data = new coDoVec3(elemName, numX * numY * numZ, uData, vData, wData);

            delete[] elemName;
            delete[] uData;
            delete[] vData;
            delete[] wData;

            setGObjs[i] = grid;
            setDObjs[i] = data;

            //delete data;
            //delete grid;
        }

        setGObjs[i] = NULL;
        setDObjs[i] = NULL;

        coDoSet *tSet = new coDoSet(oPGrid->getObjName(), setGObjs);
        for (i = 0; i < 5; i++)
            delete setGObjs[i];
        delete[] setGObjs;
        delete tSet;

        tSet = new coDoSet(oPData->getObjName(), setDObjs);
        for (i = 0; i < 5; i++)
            delete setDObjs[i];
        delete[] setDObjs;
        delete tSet;
    }
    break;

    case 1: // multiblock

    case 2: // transient1

    case 3: // transient2

    case 4: // moving

    case 5: // weird (special bonus hype)
    {
        coDistributedObject **setGObjs = NULL, **setDObjs = NULL;
        char *objName, *elemName;
        int i;

        setGObjs = new coDistributedObject *[6];
        setDObjs = new coDistributedObject *[6];

        float *xCoord, *yCoord, *zCoord;
        float *uData, *vData, *wData;
        int numX, numY, numZ, x, y, z;

        coDoStructuredGrid *grid;
        coDoVec3 *data;

        for (i = 0; i < 5; i++)
        {
            objName = (char *)oPGrid->getObjName();
            elemName = new char[strlen(objName) + 3];
            sprintf(elemName, "%s_%d", objName, i);

            switch (i)
            {
            case 0:
                numX = 20;
                numY = 24;
                numZ = 16;
                grid = genStrGrid(1.0, 1.0, 0.0, -1.0, 0.0, 1.0, numX, numY, numZ, 3214, elemName);
                break;
            case 1:
                numX = 10;
                numY = 6;
                numZ = 8;
                grid = genStrGrid(0.0, 1.0, 0.0, 1.0, 0.0, -1.0, numX, numY, numZ, 637, elemName);
                break;
            case 2:
                numX = 8;
                numY = 30;
                numZ = 12;
                grid = genStrGrid(1.0, 0.0, 1.0, 0.0, -1.0, -1.0, numX, numY, numZ, 843, elemName);
                break;
            case 3:
                numX = 15;
                numY = 24;
                numZ = 6;
                grid = genStrGrid(-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, numX, numY, numZ, 12, elemName);
                break;
            case 4:
                numX = 8;
                numY = 14;
                numZ = 6;
                grid = genStrGrid(-1.0, 1.0, 0.0, 0.0, -1.0, -1.0, numX, numY, numZ, 96, elemName);
                break;
            default:
                Covise::sendInfo("invalid case selected. no output generated.");
                return;
                break;
            }
            delete[] elemName;
            grid->getAddresses(&xCoord, &yCoord, &zCoord);

            uData = new float[numX * numY * numZ];
            vData = new float[numX * numY * numZ];
            wData = new float[numX * numY * numZ];

            for (x = 0; x < numX; x++)
            {
                for (y = 0; y < numY; y++)
                {
                    for (z = 0; z < numZ; z++)
                    {
                        if (yCoord[x * numY * numZ + y * numZ + z] > 0.5)
                        {
                            wData[x * numY * numZ + y * numZ + z] = xCoord[x * numY * numZ + y * numZ + z] + zCoord[x * numY * numZ + y * numZ + z] * (yCoord[x * numY * numZ + y * numZ + z] - 0.5f);
                            uData[x * numY * numZ + y * numZ + z] = -zCoord[x * numY * numZ + y * numZ + z] + xCoord[x * numY * numZ + y * numZ + z] * (yCoord[x * numY * numZ + y * numZ + z] - 0.5f);
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                        else if (yCoord[x * numY * numZ + y * numZ + z] < -0.5)
                        {
                            wData[x * numY * numZ + y * numZ + z] = xCoord[x * numY * numZ + y * numZ + z] + zCoord[x * numY * numZ + y * numZ + z] * (yCoord[x * numY * numZ + y * numZ + z] + 0.5f);
                            uData[x * numY * numZ + y * numZ + z] = -zCoord[x * numY * numZ + y * numZ + z] + xCoord[x * numY * numZ + y * numZ + z] * (yCoord[x * numY * numZ + y * numZ + z] + 0.5f);
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                        else
                        {
                            wData[x * numY * numZ + y * numZ + z] = xCoord[x * numY * numZ + y * numZ + z];
                            uData[x * numY * numZ + y * numZ + z] = -zCoord[x * numY * numZ + y * numZ + z];
                            vData[x * numY * numZ + y * numZ + z] = 0.0;
                        }
                    }
                }
            }

            objName = (char *)oPData->getObjName();
            elemName = new char[strlen(objName) + 3];
            sprintf(elemName, "%s_%d", objName, i);

            data = new coDoVec3(elemName, numX * numY * numZ, uData, vData, wData);

            delete[] elemName;
            delete[] uData;
            delete[] vData;
            delete[] wData;

            setGObjs[i] = grid;
            setDObjs[i] = data;

            //delete data;
            //delete grid;
        }

        setGObjs[i] = NULL;
        setDObjs[i] = NULL;

        coDoSet *tSet = new coDoSet(oPGrid->getObjName(), setGObjs);
        for (i = 0; i < 5; i++)
            delete setGObjs[i];
        delete[] setGObjs;
        delete tSet;

        tSet = new coDoSet(oPData->getObjName(), setDObjs);
        for (i = 0; i < 5; i++)
            delete setDObjs[i];
        delete[] setDObjs;
        delete tSet;
    }

    default:
        Covise::sendInfo("invalid case selected. no output generated.");
    }

    return;
}

void TGenDat::unstrGrid()
{
    switch (pStyleChoice->getValue())
    {
    case 0: // circular

    case 1: // multiblock

    case 2: // transient1

    case 3: // transient2

    case 4: // moving

    case 5: // weird (special bonus hype)

    default:
        Covise::sendInfo("invalid case selected. no output generated.");
    }

    return;
}

coDoRectilinearGrid *TGenDat::genRectGrid(float x0, float y0, float z0,
                                          float x1, float y1, float z1,
                                          int numX, int numY, int numZ, int seed, char *objName)
{
    int i;
    float *xCoord, *yCoord, *zCoord, *tCoord;
    coDoRectilinearGrid *grid = NULL;

    xCoord = new float[numX];
    yCoord = new float[numY];
    zCoord = new float[numZ];

    srand(seed);

    tCoord = new float[numX];
    loadRand(numX, tCoord);
    for (i = 0; i < numX; i++)
        xCoord[i] = x0 + tCoord[i] * (x1 - x0);
    xCoord[0] = x0;
    xCoord[numX - 1] = x1;
    delete[] tCoord;

    tCoord = new float[numY];
    loadRand(numY, tCoord);
    for (i = 0; i < numY; i++)
        yCoord[i] = y0 + tCoord[i] * (y1 - y0);
    yCoord[0] = y0;
    yCoord[numY - 1] = y1;
    delete[] tCoord;

    tCoord = new float[numZ];
    loadRand(numZ, tCoord);
    for (i = 0; i < numZ; i++)
        zCoord[i] = z0 + tCoord[i] * (z1 - z0);
    zCoord[0] = z0;
    zCoord[numZ - 1] = z1;
    delete[] tCoord;

    grid = new coDoRectilinearGrid(objName, numX, numY, numZ, xCoord, yCoord, zCoord);
    delete[] xCoord;
    delete[] yCoord;
    delete[] zCoord;

    return (grid);
}

coDoStructuredGrid *TGenDat::genStrGrid(float x0, float y0, float z0,
                                        float x1, float y1, float z1,
                                        int numX, int numY, int numZ, int seed, char *objName)
{
    int i, j, k;
    float *xCoord, *yCoord, *zCoord, *tCoord;
    coDoStructuredGrid *grid = NULL;

    xCoord = new float[numX * numY * numZ];
    yCoord = new float[numX * numY * numZ];
    zCoord = new float[numX * numY * numZ];

    srand(seed);

    // TODO
    tCoord = new float[numX];
    loadRand(numX, tCoord);
    for (i = 0; i < numX; i++)
    {
        for (j = 0; j < numY; j++)
        {
            for (k = 0; k < numZ; k++)
                xCoord[i * numY * numZ + j * numZ + k] = x0 + tCoord[i] * (x1 - x0);
        }
    }
    delete[] tCoord;

    tCoord = new float[numY];
    loadRand(numY, tCoord);
    for (i = 0; i < numY; i++)
    {
        for (j = 0; j < numX; j++)
        {
            for (k = 0; k < numZ; k++)
                yCoord[j * numY * numZ + i * numZ + k] = y0 + tCoord[i] * (y1 - y0);
        }
    }
    delete[] tCoord;

    tCoord = new float[numZ];
    loadRand(numZ, tCoord);
    for (i = 0; i < numZ; i++)
    {
        for (j = 0; j < numY; j++)
        {
            for (k = 0; k < numX; k++)
                zCoord[k * numY * numZ + j * numZ + i] = z0 + tCoord[i] * (z1 - z0);
        }
    }
    delete[] tCoord;

    grid = new coDoStructuredGrid(objName, numX, numY, numZ, xCoord, yCoord, zCoord);
    delete[] xCoord;
    delete[] yCoord;
    delete[] zCoord;

    return (grid);
}

float TGenDat::getRand()
{
    int r = rand();
    //float fr=(float)r, q=(float)((2^15)-1);
    float fr = (float)r, q = (float)RAND_MAX;
    return (fr / q);
}

void TGenDat::loadRand(int num, float *val)
{
    int i, f;
    float t, n;

    // load values
    for (i = 0; i < num; i++)
        val[i] = getRand();

    // bubblesort
    f = 1;
    while (f)
    {
        f = 0;
        for (i = 1; i < num; i++)
        {
            if (val[i] < val[i - 1])
            {
                t = val[i];
                val[i] = val[i - 1];
                val[i - 1] = t;
                f = 1;
            }
        }
    }

    // make sure we are in [0.0 , 1.0] and have one 0.0 and one 1.0 value
    n = val[0];
    t = 1.0f / (val[num - 1] - n);
    for (i = 0; i < num; i++)
        val[i] = (val[i] - n) * t;

    // done
    return;
}

/*

int tempBla()
{
 //
 // ...... Read Parameters ........
 //
 int i;
 int x,y,z;
 int imin, imax, Size, ySize,zSize;
 int Coord_Type;
int Timestep,tmin,tmax;
int Coord_Representation;
int Coord_Range;
int Function;
int Orientation;
int doneV=0;
char *Mesh,*SDaten,*VDaten; 			// Out Object Names
char *Colorname;
char *COLOR = "COLOR";
float *Datenptr=NULL,*xKoord=NULL,*yKoord=NULL,*zKoord=NULL,*Koordx=NULL,*Koordy=NULL,*Koordz=NULL;
float *Datenptrx,*Datenptry,*Datenptrz;
float startx,starty,startz;
float endx,endy,endz;
float T,a;
coDoFloat      *sdaten=NULL;
coDoVec3      *vdaten=NULL;
coDoStructuredGrid           *gridstruct=NULL;
coDoUniformGrid              *griduni=NULL;
coDoRectilinearGrid          *gridrect=NULL;

// fprintf(stderr, "in compute Callback\n");

Covise::log_message(__LINE__,__FILE__,"Generating Data");

Covise::get_choice_param("Coord_Type", &Coord_Type);
Covise::get_choice_param("Coord_Representation", &Coord_Representation);
Covise::get_choice_param("Coord_Range", &Coord_Range);
Covise::get_choice_param("Function", &Function);
Covise::get_choice_param("Orientation", &Orientation);
Covise::get_slider_param("xSize", &imin, &imax, &xSize);
Covise::get_slider_param("ySize", &imin, &imax, &ySize);
Covise::get_slider_param("zSize", &imin, &imax, &zSize);
Covise::get_slider_param("start", &startx, &starty, &startz);
Covise::get_slider_param("end", &endx, &endy, &endz);
Covise::get_slider_param("timestep", &tmin, &tmax, &Timestep);
Covise::get_string_param("color",&Colorname);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0'))

T=Timestep/2.0;

Mesh       = Covise::get_object_name("Grid");
SDaten      = Covise::get_object_name("Scalar Data");
VDaten      = Covise::get_object_name("Vector Data");

// fprintf(stderr, "nach Parametern\n");

if(Mesh == 0L) {
Covise::sendError("ERROR: Object name not correct for 'grid'");
return;
}
if(SDaten == 0L) {
Covise::sendError("ERROR: Object name not correct for 'Sdata'");
return;
}
if(VDaten == 0L) {
Covise::sendError("ERROR: Object name not correct for 'Vdata'");
return;
}

// cout << "Type: " << Coord_Type << "Funktion: " << Function << "\n";
// cout << "Range: " << Coord_Range << "Rep: " << Coord_Representation << "\n";
// cout << "xSize: " << xSize << "\n";
// cout << "ySize: " << ySize << "\n";
// cout << "zSize: " << zSize << "\n";

if(xSize<1)
xSize=1;
if(ySize<1)
ySize=1;
if(zSize<1)
zSize=1;

float xDiv = (xSize>1) ? (xSize-1) : 1 ;
float yDiv = (ySize>1) ? (ySize-1) : 1 ;
float zDiv = (zSize>1) ? (zSize-1) : 1 ;

switch(Coord_Type)
{
case UNIFORM:
if(Coord_Representation==UNIFORM)
{
//     fprintf(stderr, "vor new coDoUniformGrid %s\n", Mesh);
if(Coord_Range==1)
griduni = new coDoUniformGrid(Mesh,xSize , ySize, zSize, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
else if(Coord_Range==2)
griduni = new coDoUniformGrid(Mesh,xSize , ySize, zSize, 0.0, (float) xSize, 0.0, (float) ySize, 0.0, (float) zSize);
else
griduni = new coDoUniformGrid(Mesh,xSize , ySize, zSize, startx, endx, starty, endy, startz, endz);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0'))
griduni->addAttribute(COLOR,Colorname);

griduni->addAttribute("DataObjectName", SDaten);
//     fprintf(stderr, "nach new coDoUniformGrid\n");
}
else if(Coord_Representation==RECTILINEAR)
{
if((xKoord = new float[xSize])==NULL)
{
cout << "Could not alocate " << xSize*sizeof(float) << " Bytes for xKoord\n";
exit(255);
}
if((yKoord = new float[ySize])==NULL)
{
cout << "Could not alocate " << ySize*sizeof(float) << " Bytes for yKoord\n";
exit(255);
}
if((zKoord = new float[zSize])==NULL)
{
cout << "Could not alocate " << zSize*sizeof(float) << " Bytes for zKoord\n";
exit(255);
}
if(Coord_Range==1)
{
for(i=0;i<xSize;i++)
xKoord[i]= -1.0+((i*2.0)/xDiv);
for(i=0;i<ySize;i++)
yKoord[i]= -1.0+((i*2.0)/yDiv);
for(i=0;i<zSize;i++)
zKoord[i]= -1.0+((i*2.0)/zDiv);
}
else if(Coord_Range==2)
{
for(i=0;i<xSize;i++)
xKoord[i]= (float)i;
for(i=0;i<ySize;i++)
yKoord[i]= (float)i;
for(i=0;i<zSize;i++)
zKoord[i]= (float)i;
}
else
{
for(i=0;i<xSize;i++)
xKoord[i]= startx+((i*endx-startx)/xDiv);
for(i=0;i<ySize;i++)
yKoord[i]= starty+((i*endy-starty)/yDiv);
for(i=0;i<zSize;i++)
zKoord[i]= startz+((i*endz-startz)/zDiv);
}
gridrect = new coDoRectilinearGrid(Mesh,xSize, ySize, zSize, xKoord, yKoord, zKoord);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0'))
gridrect->addAttribute(COLOR,Colorname);
//	 delete[] xKoord;
//	 delete[] yKoord;
//	 delete[] zKoord;
}
else // Structured Grid
{
if((Koordx = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koordx\n";
exit(255);
}
if((Koordy = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koordy\n";
exit(255);
}
if((Koordz = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koordz\n";
exit(255);
}
if(Coord_Range==1)
{
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Koordx[x*ySize*zSize+y*zSize+z]= -1.0+((x*2.0)/xDiv);
Koordy[x*ySize*zSize+y*zSize+z]= -1.0+((y*2.0)/yDiv);
Koordz[x*ySize*zSize+y*zSize+z]= -1.0+((z*2.0)/zDiv);
}
}
else if(Coord_Range==2)
{
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Koordx[x*ySize*zSize+y*zSize+z]=x;
Koordy[x*ySize*zSize+y*zSize+z]=y;
Koordz[x*ySize*zSize+y*zSize+z]=z;
}
}
else
{
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Koordx[x*ySize*zSize+y*zSize+z]=startx+(x*(endx-startx)/xDiv);
Koordy[x*ySize*zSize+y*zSize+z]=starty+(y*(endy-starty)/yDiv);
Koordz[x*ySize*zSize+y*zSize+z]=startz+(z*(endz-startz)/zDiv);
}
}
gridstruct = new coDoStructuredGrid(Mesh,xSize, ySize, zSize, Koordx, Koordy, Koordz);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0'))
gridstruct->addAttribute(COLOR,Colorname);
char abuf[1000];
sprintf(abuf,"M%s\n%s\n%s\nend\n%f\n%f\n%f\nstart\n%f\n%f\n%f\n1\n"
,Covise::get_module(),Covise::get_instance(),Covise::get_host()
,endx,endy,endz,startx,starty,startz);
gridstruct->addAttribute("VECTOR0",abuf);
if(Orientation==OPT2)
{
doneV=1;
// Vektordaten

if((Datenptrx = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for VDatax\n";
exit(255);
}
if((Datenptry = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for VDatay\n";
exit(255);
}
if((Datenptrz = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for VDataz\n";
exit(255);
}

for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
a=(Koordx[x*ySize*zSize+y*zSize+z]-T)*(Koordx[x*ySize*zSize+y*zSize+z]-T)+Koordy[x*ySize*zSize+y*zSize+z]*Koordy[x*ySize*zSize+y*zSize+z];
if(a==0.0)
{
Datenptrx[x*ySize*zSize+y*zSize+z]=0.0;
Datenptry[x*ySize*zSize+y*zSize+z]=0.0;
Datenptrz[x*ySize*zSize+y*zSize+z]=0.0;
}
else
{
Datenptrx[x*ySize*zSize+y*zSize+z]=1+Koordy[x*ySize*zSize+y*zSize+z]/a;
Datenptry[x*ySize*zSize+y*zSize+z]=-1*(Koordx[x*ySize*zSize+y*zSize+z]-T)/a;
Datenptrz[x*ySize*zSize+y*zSize+z]=0.0;
}
}
vdaten = new coDoVec3(VDaten,xSize, ySize, zSize, Datenptrx, Datenptry, Datenptrz);
delete[] Datenptrx;
delete[] Datenptry;
delete[] Datenptrz;
}
delete[] Koordx;
delete[] Koordy;
delete[] Koordz;
}

break;
case RECTILINEAR:
if(Coord_Representation==UNIFORM)
{
// error = Covise::update_choice_param("Coord_Representation", RECTILINEAR);
}
if(Coord_Representation<=RECTILINEAR)
{
if((xKoord = new float[xSize])==NULL)
{
cout << "Could not alocate " << xSize*sizeof(float) << " Bytes for xKoord\n";
exit(255);
}
if((yKoord = new float[ySize])==NULL)
{
cout << "Could not alocate " << ySize*sizeof(float) << " Bytes for yKoord\n";
exit(255);
}
if((zKoord = new float[zSize])==NULL)
{
cout << "Could not alocate " << zSize*sizeof(float) << " Bytes for zKoord\n";
exit(255);
}
if(Coord_Range==1)
{
for(i=0;i<xSize;i++)
if((i-xSize/2.0)<0)
xKoord[i]=pow((i-xSize/2.0),2)/pow((float)((xSize-1)-xSize/2),2);
else
xKoord[i]=pow((i-xSize/2.0),2)/pow((float)((xSize-1)-xSize-xSize/2),2)* -1;
for(i=0;i<ySize;i++)
if((i-ySize/2.0)<0)
yKoord[i]=pow((i-ySize/2.0),2)/pow((float)((ySize-1)-ySize/2),2);
else
yKoord[i]=pow((i-ySize/2.0),2)/pow((float)((ySize-1)-ySize/2),2)* -1;
for(i=0;i<zSize;i++)
if((i-zSize/2.0)<0)
zKoord[i]=pow((i-zSize/2.0),2)/pow((float)((zSize-1)-zSize/2),2);
else
zKoord[i]=pow((i-zSize/2.0),2)/pow((float)((zSize-1)-zSize/2),2)* -1;
}
else
{
for(i=0;i<xSize;i++)
xKoord[i]=pow((float)i,2);
for(i=0;i<ySize;i++)
yKoord[i]=pow((float)i,2);
for(i=0;i<zSize;i++)
zKoord[i]=pow((float)i,2);
}
gridrect = new coDoRectilinearGrid(Mesh,xSize, ySize, zSize, xKoord, yKoord, zKoord);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0'))
gridrect->addAttribute(COLOR,Colorname);
delete[] xKoord;
delete[] yKoord;
delete[] zKoord;
}
else
{
if((Koordx = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordy = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordz = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if(Coord_Range==1)
{
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
if((x-xSize/2.0)<0)
Koordx[x*ySize*zSize+y*zSize+z]=pow((x-xSize/2.0),2)/pow((float)((xSize-1)-xSize/2),2);
else
Koordx[x*ySize*zSize+y*zSize+z]=pow((x-xSize/2.0),2)/pow((float)((xSize-1)-xSize/2),2)* -1;
if((y-ySize/2.0)<0)
Koordy[x*ySize*zSize+y*zSize+z]=pow((y-ySize/2.0),2)/pow((float)((ySize-1)-ySize/2),2);
else
Koordy[x*ySize*zSize+y*zSize+z]=pow((y-ySize/2.0),2)/pow((float)((ySize-1)-ySize/2),2)* -1;
if((z-xSize/2.0)<0)
Koordz[x*ySize*zSize+y*zSize+z]=pow((z-zSize/2.0),2)/pow((float)((zSize-1)-zSize/2),2);
else
Koordz[x*ySize*zSize+y*zSize+z]=pow((z-zSize/2.0),2)/pow((float)((zSize-1)-zSize/2),2)* -1;
}
}
else
{
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Koordx[x*ySize*zSize+y*zSize+z]=pow((float)x,2);
Koordy[x*ySize*zSize+y*zSize+z]=pow((float)y,2);
Koordz[x*ySize*zSize+y*zSize+z]=pow((float)z,2);
}
}
gridstruct = new coDoStructuredGrid(Mesh,xSize, ySize, zSize, Koordx, Koordy, Koordz);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0')) {
gridstruct->addAttribute(COLOR,Colorname);
}
delete[] Koordx;
delete[] Koordy;
delete[] Koordz;
}
break;
case HALF_CYL:
if(Coord_Representation<=RECTILINEAR)
{
// error = Covise::update_choice_param("Coord_Representation", 3);
}
if((Koordx = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordy = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordz = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Koordx[x*ySize*zSize+y*zSize+z]= -1.0+((x*2.0)/xSize);
Koordy[x*ySize*zSize+y*zSize+z]=(float)z/zSize*sin(((float)y/yDiv)*M_PI);
Koordz[x*ySize*zSize+y*zSize+z]=(float)z/zSize*cos(((float)y/yDiv)*M_PI);
}
gridstruct = new coDoStructuredGrid(Mesh,xSize, ySize, zSize, Koordx, Koordy, Koordz);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0'))
gridstruct->addAttribute(COLOR,Colorname);
delete[] Koordx;
delete[] Koordy;
delete[] Koordz;

break;
case RANDOM:
if(Coord_Representation<=RECTILINEAR)
{
// error = Covise::update_choice_param("Coord_Representation", 3);
}
if((Koordx = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordy = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordz = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Koordx[x*ySize*zSize+y*zSize+z]=drand48();
Koordy[x*ySize*zSize+y*zSize+z]=drand48();
Koordz[x*ySize*zSize+y*zSize+z]=drand48();
}
gridstruct = new coDoStructuredGrid(Mesh,xSize, ySize, zSize, Koordx, Koordy, Koordz);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0'))
gridstruct->addAttribute(COLOR,Colorname);
delete[] Koordx;
delete[] Koordy;
delete[] Koordz;
break;
case FULL_CYL:
if(Coord_Representation<=RECTILINEAR)
{
// error = Covise::update_choice_param("Coord_Representation", 3);
}
if((Koordx = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordy = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordz = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Koordx[x*ySize*zSize+y*zSize+z]= -1.0+((x*2.0)/xSize);
Koordy[x*ySize*zSize+y*zSize+z]=(float)z/zSize*sin(((float)y/(ySize-1))*M_PI*2.0);
Koordz[x*ySize*zSize+y*zSize+z]=(float)z/zSize*cos(((float)y/(ySize-1))*M_PI*2.0);
}
gridstruct = new coDoStructuredGrid(Mesh,xSize, ySize, zSize, Koordx, Koordy, Koordz);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0'))
gridstruct->addAttribute(COLOR,Colorname);
delete[] Koordx;
delete[] Koordy;
delete[] Koordz;

break;
case TORUS:
if(Coord_Representation<=RECTILINEAR)
{
// error = Covise::update_choice_param("Coord_Representation", 3);
}
if((Koordx = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordy = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
if((Koordz = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Koord\n";
exit(255);
}
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Koordx[x*ySize*zSize+y*zSize+z]=sin(((float)x/(xSize-1))*M_PI*2.0)*(4+(cos(((float)z/(zSize-1))*M_PI*2.0)*(float)y/ySize));
Koordy[x*ySize*zSize+y*zSize+z]=sin(((float)z/(zSize-1))*M_PI*2.0)*(float)y/ySize;
Koordz[x*ySize*zSize+y*zSize+z]=cos(((float)x/(xSize-1))*M_PI*2.0)*(4+(cos(((float)z/(zSize-1))*M_PI*2.0)*(float)y/ySize));
}
gridstruct = new coDoStructuredGrid(Mesh,xSize, ySize, zSize, Koordx, Koordy, Koordz);
if((strcmp(Colorname,"none")!=0)&&(Colorname[0]!='\0'))
gridstruct->addAttribute(COLOR,Colorname);
delete[] Koordx;
delete[] Koordy;
delete[] Koordz;

break;
}
if((Datenptr = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for Data\n";
exit(255);
}
switch(Function)
{
case SINUS:
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
Datenptr[x*ySize*zSize+y*zSize+z]=sin((float)x)*sin((float)y)*sin((float)z);
break;
case RAMP:
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
Datenptr[x*ySize*zSize+y*zSize+z]=x % 3 + y % 3 + z % 3;
break;
case RANDOM:
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
Datenptr[x*ySize*zSize+y*zSize+z]=drand48();
break;
case PIPE:
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
Datenptr[x*ySize*zSize+y*zSize+z]=sin((float)x)*sin((float)y);
break;
}
sdaten = new coDoFloat(SDaten,xSize, ySize, zSize, Datenptr);
sdaten->addAttribute("DataObjectName", SDaten);
delete[] Datenptr;

if(!doneV)
{
// Vektordaten

if((Datenptrx = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for VDatax\n";
exit(255);
}
if((Datenptry = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for VDatay\n";
exit(255);
}
if((Datenptrz = new float[xSize*ySize*zSize])==NULL)
{
cout << "Could not alocate " << xSize*ySize*zSize*sizeof(float) << " Bytes for VDataz\n";
exit(255);
}
switch(Orientation)
{
case OPT1:
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Datenptrx[x*ySize*zSize+y*zSize+z]=sin(((float)x/(float)xSize)*(M_PI/2.0));
Datenptry[x*ySize*zSize+y*zSize+z]=sin(((float)y/(float)ySize)*(M_PI/2.0));
Datenptrz[x*ySize*zSize+y*zSize+z]=sin(((float)z/(float)zSize)*(M_PI/2.0));
}
break;
case OPT2:
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Datenptrx[x*ySize*zSize+y*zSize+z]=(float)x;
Datenptry[x*ySize*zSize+y*zSize+z]=(float)y;
Datenptrz[x*ySize*zSize+y*zSize+z]=(float)z;
}
break;
case OPT3:
for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{
Datenptrx[x*ySize*zSize+y*zSize+z]=(float)x;
Datenptry[x*ySize*zSize+y*zSize+z]=(float)z;
Datenptrz[x*ySize*zSize+y*zSize+z]=(float)-y;
}
break;
case OPT4:
for(x=0;x<xSize*ySize*zSize;x++)
{
Datenptrx[x]=drand48();
Datenptry[x]=drand48();
Datenptrz[x]=drand48();
}
break;
case CIRCULAR:
{
float v1[3], v2[3], c[3];

v2[0] = 0.0; v2[1] = 1.0; v2[2] = 0.0;  // Normale
c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;   // Mittelpunkt

for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{

v1[0] = xKoord[x]-c[0];
v1[1] = 0.0;
v1[2] = zKoord[z]-c[2];

Datenptrz[x*ySize*zSize+y*zSize+z]=v1[0]*v2[1] - v2[0]*v1[1];
Datenptrx[x*ySize*zSize+y*zSize+z]=v1[1]*v2[2] - v2[1]*v1[2];
Datenptry[x*ySize*zSize+y*zSize+z]=v1[2]*v2[0] - v2[2]*v1[0];
}
}

delete[] xKoord;
delete[] yKoord;
delete[] zKoord;

break;
case EXPANDING:
{
float v1[3], v2[3], c[3];

v2[0] = 0.0; v2[1] = 1.0; v2[2] = 0.0;  // Normale
c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;   // Mittelpunkt

for(x=0;x<xSize;x++)
for(y=0;y<ySize;y++)
for(z=0;z<zSize;z++)
{

v1[0] = xKoord[x]-c[0];
v1[1] = 0.0;
v1[2] = zKoord[z]-c[2];

Datenptrz[x*ySize*zSize+y*zSize+z]=(v1[0]*v2[1] - v2[0]*v1[1])+v1[2]*0.2;
Datenptrx[x*ySize*zSize+y*zSize+z]=(v1[1]*v2[2] - v2[1]*v1[2])+v1[0]*0.2;
Datenptry[x*ySize*zSize+y*zSize+z]=(v1[2]*v2[0] - v2[2]*v1[0])+v1[1]*0.2;
}
}

delete[] xKoord;
delete[] yKoord;
delete[] zKoord;

break;
case OSCILLATE:
{

}
break;
}
vdaten = new coDoVec3(VDaten,xSize, ySize, zSize, Datenptrx, Datenptry, Datenptrz);
delete[] Datenptrx;
delete[] Datenptry;
delete[] Datenptrz;
}

// fprintf(stderr, "nach switch\n");

if(gridstruct!=NULL)
delete gridstruct;
if(griduni!=NULL)
delete griduni;
if(gridrect!=NULL)
delete gridrect;
if(vdaten!=NULL)
delete vdaten;
if(sdaten!=NULL)
delete sdaten;
}
*/

MODULE_MAIN(UnderDev, TGenDat)
