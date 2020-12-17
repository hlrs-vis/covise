/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "CoolEmAllToDatabase, world!" in COVISE API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "CoolEmAllToDatabase.h"
#include "../../../tools/CoolEmAllCasePreparation/CoolEmAllClient.h"
#include <string.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CoolEmAllToDatabase::CoolEmAllToDatabase(int argc, char *argv[])
    : coModule(argc, argv, "CoolEmAllToDatabase, world! program")
{
    // no parameters, no ports...
    p_grid = addInputPort("grid", "UnstructuredGrid", "Distributed Grid");
    p_boco = addInputPort("boco", "USR_FenflossBoco", "Boundary Conditions");
    p_temp = addInputPort("temp", "Float", "Temperature Values");
    p_p = addInputPort("pressure", "Float", "Pressure Values");
    p_velo = addInputPort("velocities", "Vec3", "Velocity Values");
    p_gridOut = addOutputPort("gridout", "UnstructuredGrid", "the computational mesh");

    p_databasePrefix = addStringParam("databasePrefix", "databasePrefix");
    p_databasePrefix->setValue("none");
    p_csvPath = addFileBrowserParam("csvPath", "path to csv file");
    p_csvPath->setValue("/tmp/CoolEmAll.csv", "*.csv");

    p_grid->setRequired(1);
    p_boco->setRequired(1);
    p_temp->setRequired(0);
    p_p->setRequired(0);
    p_velo->setRequired(0);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int CoolEmAllToDatabase::compute(const char *port)
{

    const coDoSet *boco;
    const coDoUnstructuredGrid *grid;
    const coDoFloat *tempObject;
    const coDoFloat *pressureObject;
    const coDoVec3 *veloObject;

    temp = NULL;
    press = NULL;
    grid = dynamic_cast<const coDoUnstructuredGrid *>(p_grid->getCurrentObject());
    boco = dynamic_cast<const coDoSet *>(p_boco->getCurrentObject());
    tempObject = dynamic_cast<const coDoFloat *>(p_temp->getCurrentObject());
    if (tempObject)
    {
        tempObject->getAddress(&temp);
    }
    pressureObject = dynamic_cast<const coDoFloat *>(p_p->getCurrentObject());
    if (pressureObject)
    {
        pressureObject->getAddress(&press);
    }
    vx = NULL;
    vy = NULL;
    vz = NULL;
    veloObject = dynamic_cast<const coDoVec3 *>(p_velo->getCurrentObject());
    if (veloObject)
    {
        veloObject->getAddresses(&vx, &vy, &vz);
    }
    if (boco && grid)
    {
        const coDistributedObject *const *setObj;

        colDiricletVals = 5;
        grid->getAddresses(&elem, &conn, &x, &y, &z);
        grid->getGridSize(&numElem, &numConn, &numCoord);
        grid->getTypeList(&inTypeList);

        setObj = boco->getAllElements(&numObj);

        ///////// Diriclet Nodes
        const coDoIntArr *diricletIndexObj = dynamic_cast<const coDoIntArr *>(setObj[3]);
        if (!diricletIndexObj)
        {
            sendWarning("illegal part type (3) in boco");
            return FAIL;
        }
        else
        {
            colDiriclet = diricletIndexObj->getDimension(0); //2
            numDiriclet = diricletIndexObj->getDimension(1);
            diricletIndex = diricletIndexObj->getAddress();
        }

        //	0,10,20,30,40....32 inletnodes
        numberinletnodes = (numDiriclet * colDiriclet) / (2 * colDiricletVals);
        int *inletnodes = (int *)malloc(numberinletnodes * sizeof(int));

        if (numberinletnodes == 0)
        {
            fprintf(stderr, "missing inlet boundary condition! Stopping. Please generate inlet boco.\n");
            sendError("missing inlet boundary condition! Stopping. Please generate inlet boco.");
            return STOP_PIPELINE;
        }
        ///////// balance indices
        const coDoIntArr *balanceObj = dynamic_cast<const coDoIntArr *>(setObj[6]);
        if (!balanceObj)
        {
            sendWarning("illegal part type (6) in boco");
            return FAIL;
        }
        else
        {
            coDoIntArr *balanceObj = (coDoIntArr *)setObj[6];
            colBalance = balanceObj->getDimension(0);
            numBalance = balanceObj->getDimension(1);
            balance = balanceObj->getAddress();
            int oldType = -2;
            //types: 100-111 rack inlet
            // 200-211 rack outlet
            // 150-154 floor inlet (four different types)
            // 300-
            for (int i = 0; i < numBalance; i++)
            {
                int type = balance[(i * colBalance) + 5];
                if (type != oldType)
                {
                    fprintf(stderr, "%d Type %d\n", i, type);
                    patches.push_back(new BoundaryPatch(this, type, i));
                    oldType = type;
                }
            }
        }
    }
    int nPolOut = 0;
    int *polOut;
    int *vertOut;
    float *xOut;
    float *yOut;
    float *zOut;
    std::list<BoundaryPatch *>::iterator it;
    for (it = patches.begin(); it != patches.end(); it++)
    {
        nPolOut += (*it)->quads.size();
    }
    coDoPolygons *outgrid = new coDoPolygons(p_gridOut->getObjName(), nPolOut * 4, nPolOut * 4, nPolOut);
    outgrid->getAddresses(&xOut, &yOut, &zOut, &vertOut, &polOut);
    p_gridOut->setCurrentObject(outgrid);
    int pn = 0;
    for (it = patches.begin(); it != patches.end(); it++)
    {
        std::list<BoundaryQuad *>::iterator qit;
        for (qit = (*it)->quads.begin(); qit != (*it)->quads.end(); qit++)
        {
            polOut[pn] = pn * 4;
            for (int n = 0; n < 4; n++)
            {
                vertOut[pn * 4 + n] = pn * 4 + n;
                xOut[pn * 4 + n] = (*qit)->corners[n].x();
                yOut[pn * 4 + n] = (*qit)->corners[n].y();
                zOut[pn * 4 + n] = (*qit)->corners[n].z();
            }
            pn++;
        }
    }

    FILE *fp = fopen(p_csvPath->getValue(), "w");
    if (fp)
    {
        // obtain the existing locale name for numbers
        char *oldLocale = setlocale(LC_NUMERIC, NULL);

        // inherit locale from environment
        setlocale(LC_NUMERIC, "German");

        fprintf(fp, ";");
        for (it = patches.begin(); it != patches.end(); it++)
        {
            fprintf(fp, "%s;", (*it)->name.c_str());
        }
        fprintf(fp, "\n");

        fprintf(fp, "Area;");
        for (it = patches.begin(); it != patches.end(); it++)
        {
            fprintf(fp, "%f;", (*it)->area);
        }
        fprintf(fp, "\n");

        fprintf(fp, "Temperature;");
        for (it = patches.begin(); it != patches.end(); it++)
        {
            fprintf(fp, "%f;", (*it)->averageTemperature - 273.15);
        }
        fprintf(fp, "\n");

        fprintf(fp, "Pressure;");
        for (it = patches.begin(); it != patches.end(); it++)
        {
            fprintf(fp, "%f;", (*it)->averagePressure);
        }
        fprintf(fp, "\n");
        fprintf(fp, "Vx;");
        for (it = patches.begin(); it != patches.end(); it++)
        {
            fprintf(fp, "%f;", (*it)->averageVelo[0]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "Vy;");
        for (it = patches.begin(); it != patches.end(); it++)
        {
            fprintf(fp, "%f;", (*it)->averageVelo[1]);
        }
        fprintf(fp, "\n");

        fprintf(fp, "Vz;");
        for (it = patches.begin(); it != patches.end(); it++)
        {
            fprintf(fp, "%f;", (*it)->averageVelo[2]);
        }
        fprintf(fp, "\n");

        fclose(fp);

        // set the locale back
        setlocale(LC_NUMERIC, oldLocale);
    }

    CoolEmAllClient *cc = new CoolEmAllClient("recs1.coolemall.eu");
    for (it = patches.begin(); it != patches.end(); it++)
    {
        char pathName[1000];
        char value[1000];
        sprintf(pathName, "%s/%s", p_databasePrefix->getValue(), (*it)->name.c_str());
        sprintf(value, "%f", (*it)->averageTemperature - 273.15);
        //cc->setValue(pathName,"temperature",value);
        sprintf(value, "%f", (*it)->averagePressure);
        //cc->setValue(pathName,"pressure",value);
        float vol = sqrt((*it)->averageVelo[0] * (*it)->averageVelo[0] + (*it)->averageVelo[1] * (*it)->averageVelo[1] + (*it)->averageVelo[2] * (*it)->averageVelo[2]) / (*it)->area;
        sprintf(value, "%f", vol);
        //cc->setValue(pathName,"airflow_velocity",value);
    }

    for (it = patches.begin(); it != patches.end(); it++)
    {
        delete (*it);
    }
    patches.clear();
    return SUCCESS;
}

BoundaryPatch::BoundaryPatch(CoolEmAllToDatabase *c, int t, int balanceNumber)
{
    cd = c;
    type = t;
    instance = 0;

    std::list<BoundaryPatch *>::iterator it;
    for (it = cd->patches.begin(); it != cd->patches.end(); it++)
    {
        if ((*it)->type == type)
        {
            instance = (*it)->instance + 1;
        }
    }
    area = 0;
    averageTemperature = 0;
    averagePressure = 0;
    averageVelo[0] = 0;
    averageVelo[1] = 0;
    averageVelo[2] = 0;

    //types: 100-111 rack inlet
    // 200-211 rack outlet
    // 150-154 floor inlet (four different types)
    // 300-
    if (type >= 100 && type < 150)
    {
        char tmpName[1000];
        sprintf(tmpName, "Rack_%02d/Inlet_01", type - 100);
        name = tmpName;
    }
    if (type >= 150 && type < 200)
    {
        char tmpName[1000];
        sprintf(tmpName, "Inlet%02d", type - 150);
        name = tmpName;
    }
    if (type >= 300 && type < 400)
    {
        char tmpName[1000];
        sprintf(tmpName, "Outlet%02d_%02d", type - 300, instance);
        name = tmpName;
    }
    if (type >= 200 && type < 300)
    {
        char tmpName[1000];
        sprintf(tmpName, "Rack_%02d/Outlet_01", type - 200);
        name = tmpName;
    }
    for (int i = balanceNumber; i < cd->numBalance; i++)
    {
        int currentType = cd->balance[(i * cd->colBalance) + 5];
        if (currentType != type)
        {
            break;
        }
        int node[4];
        for (int n = 0; n < 4; n++)
        {
            node[n] = cd->balance[((i)*cd->colBalance) + n] - 1;
        }
        BoundaryQuad *bq = new BoundaryQuad();
        float temperatureSumm = 0;
        float pressureSumm = 0;
        float veloSumm[3];
        veloSumm[0] = 0;
        veloSumm[1] = 0;
        veloSumm[2] = 0;
        for (int n = 0; n < 4; n++)
        {
            bq->corners[n][0] = cd->x[node[n]];
            bq->corners[n][1] = cd->y[node[n]];
            bq->corners[n][2] = cd->z[node[n]];
            if (cd->temp)
                temperatureSumm += cd->temp[node[n]];
            if (cd->press)
                pressureSumm += cd->press[node[n]];
            if (cd->vx)
            {
                veloSumm[0] += cd->vx[node[n]];
                veloSumm[1] += cd->vy[node[n]];
                veloSumm[2] += cd->vz[node[n]];
            }
        }
        bq->averageTemperature = temperatureSumm / 4.0;
        bq->averagePressure = pressureSumm / 4.0;
        bq->averageVelo[0] = veloSumm[0] / 4.0;
        bq->averageVelo[1] = veloSumm[1] / 4.0;
        bq->averageVelo[2] = veloSumm[2] / 4.0;
        bq->computeArea();
        quads.push_back(bq);
    }
    area = 0;
    averageTemperature = 0;
    averagePressure = 0;
    averageVelo[0] = 0;
    averageVelo[1] = 0;
    averageVelo[2] = 0;
    std::list<BoundaryQuad *>::iterator qit;
    for (qit = quads.begin(); qit != quads.end(); qit++)
    {
        area += (*qit)->area;
    }
    for (qit = quads.begin(); qit != quads.end(); qit++)
    {
        fprintf(stderr, "area: %f, qArea: %f, temp %f\n", area, (*qit)->area, (*qit)->averageTemperature);
        averageTemperature += (*qit)->averageTemperature / area * (*qit)->area;
        averagePressure += (*qit)->averagePressure / area * (*qit)->area;
        averageVelo[0] += (*qit)->averageVelo[0] / area * (*qit)->area;
        averageVelo[1] += (*qit)->averageVelo[1] / area * (*qit)->area;
        averageVelo[2] += (*qit)->averageVelo[2] / area * (*qit)->area;
    }
    /*averageTemperature *= area;
  averagePressure *= area;
  averageVelo[0] *= area;
  averageVelo[1] *= area;
  averageVelo[2] *= area;*/
}
BoundaryPatch::~BoundaryPatch()
{
    std::list<BoundaryQuad *>::iterator it;
    for (it = quads.begin(); it != quads.end(); it++)
    {
        delete (*it);
    }
    quads.clear();
}

void BoundaryQuad::computeArea()
{
    osg::Vec3f v1 = (corners[2] - corners[0]);
    osg::Vec3f v2 = (corners[2] - corners[1]);
    osg::Vec3f v3 = (corners[3] - corners[0]);
    osg::Vec3f v4 = (corners[3] - corners[2]);
    osg::Vec3f av1 = v1 ^ v2;
    osg::Vec3f av2 = v3 ^ v4;
    float a1 = av1.length();
    float a2 = av2.length();
    area = a1 / 2.0 + a2 / 2.0;
}

MODULE_MAIN(Simulation, CoolEmAllToDatabase)
