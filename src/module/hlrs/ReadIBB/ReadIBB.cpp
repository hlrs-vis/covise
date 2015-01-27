/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                              2008      **
 **                                                                        **
 ** Description:  ReadIBB                                                  **
 **                                                                        **
 ** Covise read module for IBB GID files                                   **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  19.07.08  V1.0                                                  **
 \**************************************************************************/

#include "ReadIBB.h"
#include <math.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>

#define sqr(x) (x * x)

ReadIBB::ReadIBB(int argc, char *argv[])
    : coModule(argc, argv, "Read module for IBB GID files")
{

    char buf[300];

    //ports & parameters

    port_grid = addOutputPort("grid", "UnstructuredGrid", "computation grid");
    port_displacement = addOutputPort("displacements", "Vec3", "grid displacements");
    port_velocity = addOutputPort("velocity", "Vec3", "output velocity");
    port_pressure = addOutputPort("pressure", "Float", "output pressure");
    port_k = addOutputPort("k", "Float", "output k");
    port_eps = addOutputPort("eps", "Float", "output eps");
    //   port_boco = addOutputPort ("boco", "USR_FenflossBoco", "Boundary Conditions");
    port_pressrb = addOutputPort("press_rb", "Polygons", "pressure boundary conditions");
    port_wall = addOutputPort("wall", "Polygons", "wall elements");
    port_bila = addOutputPort("bila_elems", "Polygons", "marked elements");
    port_bcin = addOutputPort("bcin", "Polygons", "inlet elements");

    sprintf(buf, "%s/", getenv("HOME"));
    p_geoFile = addFileBrowserParam("geoFile", "Geometry File");
    p_geoFile->setValue(buf, "*.msh;*.MSH");

    p_simFile = addFileBrowserParam("simFile", "Geometry File");
    p_simFile->setValue(buf, "*.res*;*.RES*");

    p_firstStep = addInt32Param("firstStepNo", "first Step Nr.");
    p_firstStep->setValue(0);

    p_numt = addInt32Param("numt", "Number of Timesteps to read");
    p_numt->setValue(1000);

    p_skip = addInt32Param("skip", "Number of Timesteps to skip");
    p_skip->setValue(0);
}

void ReadIBB::postInst()
{
    p_geoFile->show(); // visible in control panel
    p_simFile->show();
}

void ReadIBB::quit()
{
}

int ReadIBB::compute(const char *)
{

    char buf[1000];

    /*  coDoVec3 *displacement = NULL;
     coDoVec3 *velocity = NULL;
   coDoFloat *k = NULL;
   coDoFloat *eps = NULL;*/
    std::vector<coordinate> coords;
    std::vector<element> elements;
    coords.reserve(10000);
    elements.reserve(10000);
    int minCoord;
    int minElem = -1;

    int skip = p_skip->getValue() + 1;
    if (skip < 0)
    {
        skip = 0;
        p_skip->setValue(skip);
    };

    int numt = p_numt->getValue();
    if (numt < 0)
    {
        numt = 0;
        p_numt->setValue(numt);
    };

    int firststep = p_firstStep->getValue();
    if (firststep < 0)
    {
        firststep = 0;
        p_firstStep->setValue(firststep);
    };

    gridFP = Covise::fopen(p_geoFile->getValue(), "r");
    if (gridFP == NULL)
    {
        return FAIL;
    }
    int currentType = 0;
    int numConn = 0;
    int i = 0;

    while (!feof(gridFP))
    {
        fgets(buf, 1000, gridFP);
        if (strncmp(buf, "COORDINATES", 11) == 0)
        {
            fgets(buf, 1000, gridFP);
            if (strncmp(buf, "END COORDINATES", 15) != 0)
            {
                sscanf(buf, "%d ", &minCoord);

                do
                {
                    int i;
                    float x, y, z;
                    sscanf(buf, "%d %f %f %f", &i, &x, &y, &z);
                    if (i >= coords.size())
                        coords.resize(i);
                    coords[i - 1].set(x, y, z);
                    if (i < minCoord)
                        minCoord = i;
                    fgets(buf, 1000, gridFP);
                } while (!feof(gridFP) && (strncmp(buf, "END COORDINATES", 15) != 0));
                fprintf(stderr, "MinCoord %d CoordSize %zu\n", minCoord, coords.size());
            }
        }
        else if (strncmp(buf, "MESH", 4) == 0)
        {
            char *nnode = strstr(buf, "NNODE");
            sscanf(nnode, "NNODE %d", &currentType);
        }
        else if (strncmp(buf, "ELEMENTS", 8) == 0)
        {
            fgets(buf, 1000, gridFP);
            if (strncmp(buf, "END ELEMENTS", 12) != 0)
            {
                if (minElem < 0)
                    sscanf(buf, "%d ", &minElem);

                do
                {
                    int v[8];
                    if (currentType == 4)
                    {
                        sscanf(buf, "%d %d %d %d %d", &i, v, v + 1, v + 2, v + 3);

                        if (i >= elements.size())
                            elements.resize(i);
                        if (i < minElem)
                            minElem = i;
                        for (int n = 0; n < currentType; n++)
                        {
                            elements[i - 1].v[n] = v[n] - minCoord;
                        }
                        numConn += 4;
                    }
                    else if (currentType == 8)
                    {
                        sscanf(buf, "%d %d %d %d %d %d %d %d %d", &i, v, v + 1, v + 2, v + 3, v + 4, v + 5, v + 6, v + 7);
                        if (i >= elements.size())
                            elements.resize(i);
                        if (i < minElem)
                            minElem = i;

                        for (int n = 0; n < currentType; n++)
                        {
                            elements[i - 1].v[n] = v[n] - minCoord;
                        }
                        numConn += 8;
                    }
                    elements[i - 1].type = currentType;

                    fgets(buf, 1000, gridFP);
                } while (!feof(gridFP) && (strncmp(buf, "END ELEMENTS", 12) != 0));
                fprintf(stderr, "MinElem %d ElemSize %zu\n", minElem, elements.size());
            }
        }
    }

    dataFP = Covise::fopen(p_simFile->getValue(), "r");

    if (dataFP != NULL)
    {

        int timestep = 0;
        float *p, *rx, *ry, *rz;
        int step = 0;
        coDoVec3 *reso = NULL;
        coDoFloat *press = NULL;
        coDoSet *v_set = NULL;
        coDoSet *d_set = NULL;
        coDoSet *p_set = NULL;

        char *Velocity, *Displacement, *Pressure;

        Velocity = Covise::get_object_name("velocity");
        Displacement = Covise::get_object_name("displacements");
        Pressure = Covise::get_object_name("pressure");

        if (Velocity == NULL)
        {
            Covise::sendError("ERROR: Object name not correct for Velocity");
            return FAIL;
        }

        if (Displacement == NULL)
        {
            Covise::sendError("ERROR: Object name not correct for Displacements");
            return FAIL;
        }
        if (Pressure == NULL)
        {
            Covise::sendError("ERROR: Object name not correct for Pressure");
            return FAIL;
        }

        v_set = new coDoSet(Velocity, SET_CREATE);
        if (!v_set->objectOk())
        {
            Covise::sendError("ERROR: creation of set 'velocity' failed");
            return FAIL;
        }

        d_set = new coDoSet(Displacement, SET_CREATE);
        if (!d_set->objectOk())
        {
            Covise::sendError("ERROR: creation of set 'displacements' failed");
            return FAIL;
        }

        p_set = new coDoSet(Pressure, SET_CREATE);
        if (!p_set->objectOk())
        {
            Covise::sendError("ERROR: creation of set 'pressure' failed");
            return FAIL;
        }

        while (!feof(dataFP))
        {
            if (timestep >= numt)
                break;
            fgets(buf, 1000, dataFP);

            bool dispCase, veloCase;
            dispCase = veloCase = false;
            if (strncmp(buf, "# RESULT DISPLACEMENTS", 22) == 0)
                dispCase = true;
            else if (strncmp(buf, "# RESULT velocity", 17) == 0)
                veloCase = true;
            if (dispCase || veloCase)
            {
                std::vector<coordinate> result;
                result.reserve(coords.size() - minCoord + 1);
                coordinate initVec = { 0, 0, 0 };
                result.assign(coords.size() - minCoord + 1, initVec);

                fgets(buf, 1000, dataFP);
                fgets(buf, 1000, dataFP);

                char *tempStr = strstr(buf, "STEP");
                sscanf(tempStr, "STEP %d", &step);

                if ((step >= firststep) && ((step - firststep) % skip == 0))
                {

                    while (!feof(dataFP))
                    {
                        fgets(buf, 1000, dataFP);
                        if (strncmp(buf, "VALUES", 6) == 0)
                        {
                            while (!feof(dataFP))
                            {
                                fgets(buf, 1000, dataFP);
                                if (strncmp(buf, "END VALUES", 10) == 0)
                                    break;

                                int i;
                                float x, y, z;
                                sscanf(buf, "%d %f %f %f", &i, &x, &y, &z);
                                if (i - minCoord + 1 >= result.size())
                                    result.resize(i - minCoord + 1);
                                result[i - minCoord].set(x, y, z);
                            }
                        }
                        if (strncmp(buf, "END VALUES", 10) == 0)
                        {

                            if (dispCase)
                                sprintf(buf, "%s_%d", Displacement, step + 1);
                            else
                                sprintf(buf, "%s_%d", Velocity, step + 1);
                            reso = new coDoVec3(buf, result.size());
                            if (!reso->objectOk())
                            {
                                Covise::sendError("ERROR: creation of data object 'velocity' failed");
                                return FAIL;
                            }
                            reso->getAddresses(&rx, &ry, &rz);
                            for (i = 0; i < result.size(); i++)
                            {
                                rx[i] = result[i].x;
                                ry[i] = result[i].y;
                                rz[i] = result[i].z;
                            }
                            if (dispCase)
                                d_set->addElement(reso);
                            else
                                v_set->addElement(reso);
                            break;
                        }
                    }
                }
            }
            else if (strncmp(buf, "# RESULT pressure", 17) == 0)
            {
                std::vector<float> pressResult;
                pressResult.reserve(coords.size() - minCoord + 1);
                pressResult.assign(coords.size() - minCoord + 1, 0);

                fgets(buf, 1000, dataFP);
                fgets(buf, 1000, dataFP);
                char *tempStr = strstr(buf, "STEP");
                sscanf(tempStr, "STEP %d", &step);
                fprintf(stderr, "Data Timestep: %d\n", step);

                if ((step >= firststep) && ((step - firststep) % skip == 0))
                {
                    fprintf(stderr, "Chosen Timestep: %d\n", timestep);

                    while (!feof(dataFP))
                    {
                        fgets(buf, 1000, dataFP);
                        if (strncmp(buf, "VALUES", 6) == 0)
                        {
                            while (!feof(dataFP))
                            {
                                fgets(buf, 1000, dataFP);
                                if (strncmp(buf, "END VALUES", 10) == 0)
                                    break;

                                int i;
                                float x;
                                sscanf(buf, "%d %f", &i, &x);
                                if (i - minCoord + 1 >= pressResult.size())
                                    pressResult.resize(i - minCoord + 1);
                                pressResult[i - minCoord] = x;
                            }
                        }
                        if (strncmp(buf, "END VALUES", 10) == 0)
                        {
                            sprintf(buf, "%s_%d", Pressure, step + 1);
                            press = new coDoFloat(buf, pressResult.size());
                            if (!press->objectOk())
                            {
                                Covise::sendError("ERROR: creation of data object 'velocity' failed");
                                return FAIL;
                            }
                            press->getAddress(&p);
                            for (i = 0; i < pressResult.size(); i++)
                                p[i] = pressResult[i];
                            p_set->addElement(press);
                            timestep++;
                            break;
                        }
                    }
                }
            }
        }
        if (step + 1 > 1)
        {
            p_set->addAttribute("TIMESTEP", "0 100 0");
            d_set->addAttribute("TIMESTEP", "0 100 0");
            v_set->addAttribute("TIMESTEP", "0 100 0");
        }
        delete p_set;
        delete d_set;
        delete v_set;
    }

    if (gridFP != NULL)
        fclose(gridFP);
    if (dataFP != NULL)
        fclose(dataFP);

    coDoUnstructuredGrid *unsGrd = new coDoUnstructuredGrid(port_grid->getObjName(),
                                                            elements.size() - minElem + 1, // number of elements
                                                            numConn, // number of connectivities
                                                            coords.size() - minCoord + 1, // number of coordinates
                                                            1); // does type list exist?

    int *elem, *conn, *type;
    float *xc, *yc, *zc;

    unsGrd->getAddresses(&elem, &conn, &xc, &yc, &zc);
    unsGrd->getTypeList(&type);

    int el = 0;
    for (i = minElem - 1; i < elements.size(); i++)
    {
        *elem = el;
        elem++;
        int et = elements[i].type;
        el += et;
        for (int n = 0; n < et; n++)
        {
            *conn = elements[i].v[n];
            conn++;
        }
        if (et == 4)
        {
            *type = TYPE_QUAD;
            type++;
        }
        else
        {
            *type = TYPE_HEXAGON;
            type++;
        }
    }
    for (i = minCoord; i <= coords.size(); i++)
    {
        xc[i - minCoord] = coords[i - 1].x;
        yc[i - minCoord] = coords[i - 1].y;
        zc[i - minCoord] = coords[i - 1].z;
    }

    // set out port
    port_grid->setCurrentObject(unsGrd);

    return SUCCESS;
}

MODULE_MAIN(IO, ReadIBB)
