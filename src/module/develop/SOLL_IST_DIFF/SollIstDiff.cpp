/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class SollIstDiff                     ++
// ++                                                                     ++
// ++ Author:  Sven Kufer (sk@vircinity.com)                              ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 04.10.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "SollIstDiff.h"
#include "stdlib.h"
#include "float.h"

float tolerance;
typedef enum
{
    SORT_X,
    SORT_Y,
    SORT_Z
} Modes;
Modes mode = SORT_Z;

//
// Constructor
//
SollIstDiff::SollIstDiff()
    : coSimpleModule("soll ist vergleich")
{
    p_polyIn = addInputPort("polys", "coDoPolygons", " ");

    p_polyOut = addOutputPort("ist_polys", "coDoPolygons", " ");
    p_diffOut = addOutputPort("disp_vec", "coDoVec3", " ");
    p_totalOut = addOutputPort("total_deviation", "coDoFloat", " ");
    p_pointsOut = addOutputPort("points", "coDoPoints", " ");
    p_pointTotalOut = addOutputPort("total_deviation on points", "coDoFloat", " ");

    p_file = addFileBrowserParam("dev_file", "dev");
    p_file->setValue("/data/Kunden/Metris/deviations.txt", "*.txt");

    p_tol = addFloatParam("start tolerance", "tol");
    p_tol->setValue(1.e-4);

    p_onlyGeo = addBooleanParam("only geometry", " ");
    p_onlyGeo->setValue(0);
}

static float
vecabs(Vector *vec)
{
    return ((vec->x * vec->x) + (vec->y * vec->y) + (vec->z * vec->z));
}

static float vecdiff(Vector *v2, Vector *v1)
{
    Vector vec;
    vec.x = v2->x - v1->x;
    vec.y = v2->y - v1->y;
    vec.z = v2->z - v1->z;

    return (vecabs(&vec));
}

int vec_compare(const void *v1, const void *v2)
{
    dinfo *node1 = (dinfo *)v1;
    dinfo *node2 = (dinfo *)v2;

    float vdiff;
    if (mode == SORT_X)
    {
        vdiff = node2->coord.x - node1->coord.x;
    }
    else if (mode == SORT_Y)
    {
        vdiff = node2->coord.y - node1->coord.y;
    }
    else
    {
        vdiff = node2->coord.z - node1->coord.z;
    }

    if (fabs(vdiff) < tolerance)
    {
        return 0;
    }
    else if (vdiff > 0)
    {
        return -1;
    }

    return 1;
}

//
// compute method
//
int
SollIstDiff::compute()
{

    tolerance = p_tol->getValue();

    coDoPolygons *poly = (coDoPolygons *)p_polyIn->getCurrentObject();
    int numPoints = poly->getNumPoints();
    float *x_in, *y_in, *z_in;
    int *dummy;
    poly->getAddresses(&x_in, &y_in, &z_in, &dummy, &dummy);

    data = new dinfo[10 * numPoints];
    if (!data)
    {
        Covise::sendError("Error allocating memory");
        return STOP_PIPELINE;
    }

    int numVert = 0;
    char line[512];
    float fdummy;

    FILE *dispFile = fopen(p_file->getValue(), "r");

    if (!dispFile)
    {
        Covise::sendError("Error opening file");
        return STOP_PIPELINE;
    }
    int i;
    int num_ignore;

#ifdef STIHL
    num_ignore = 6;
#else
    num_ignore = 1;
#endif

    for (i = 0; i < num_ignore; i++)
    {
        fgets(line, 256, dispFile);
    }

    while (!feof(dispFile) && numVert < 10 * numPoints - 1)
    {
        fgets(line, 256, dispFile);
#ifdef STIHL
        sscanf(line, "%d %f %f %f %f %f %f %f %f %f %f", &i,
               &fdummy, &fdummy, &fdummy,
               &data[numVert].coord.x, &data[numVert].coord.y, &data[numVert].coord.z,
               //&fdummy, &fdummy, &fdummy,
               &data[numVert].disp.x, &data[numVert].disp.y, &data[numVert].disp.z, &data[numVert].total);
#else
        sscanf(line, "%f %f %f %f %f %f %f %f %f %f",

               &data[numVert].coord.x, &data[numVert].coord.y, &data[numVert].coord.z,
               &fdummy, &fdummy, &fdummy,
               &data[numVert].disp.x, &data[numVert].disp.y, &data[numVert].disp.z, &data[numVert].total);
#endif

        //cerr << data[numVert].coord.x << data[numVert].coord.y << data[numVert].coord.z << endl;
        numVert++;
    }

    coDoPoints *points_out = new coDoPoints(p_pointsOut->getObjName(), numVert);
    float *p_x, *p_y, *p_z;
    points_out->getAddresses(&p_x, &p_y, &p_z);

    // mesh in order of file

    coDoPolygons *poly_out = new coDoPolygons(p_polyOut->getObjName(), numVert, numVert, numVert / 3);
    float *pol_x, *pol_y, *pol_z;
    int *pl, *cl;
    poly_out->getAddresses(&pol_x, &pol_y, &pol_z, &cl, &pl);
    for (i = 0; i < numVert / 3; i++)
    {
        pl[i] = i * 3;
    }

    coDoFloat *ptotal = new coDoFloat(p_pointTotalOut->getObjName(), numVert);
    float *ptotal_dev;
    ptotal->getAddress(&ptotal_dev);

    for (i = 0; i < numVert; i++)
    {
        p_x[i] = pol_x[i] = data[i].coord.x;
        p_y[i] = pol_y[i] = data[i].coord.y;
        p_z[i] = pol_z[i] = data[i].coord.z;
        ptotal_dev[i] = data[i].total;
        cl[i] = i;
    }

    p_polyOut->setCurrentObject(poly_out);
    p_pointsOut->setCurrentObject(points_out);
    p_pointTotalOut->setCurrentObject(ptotal);

    if (p_onlyGeo->getValue())
    {
        delete[] data;
        return CONTINUE_PIPELINE;
    }

    coDoVec3 *disp = new coDoVec3(p_diffOut->getObjName(), numPoints);
    float *xdiff, *ydiff, *zdiff;
    disp->getAddresses(&xdiff, &ydiff, &zdiff);

    coDoFloat *total = new coDoFloat(p_totalOut->getObjName(), numPoints);
    float *total_dev;
    total->getAddress(&total_dev);

    dinfo search;

    int run = 0;
    bool finished = false;
    search.disp.x = search.disp.y = search.disp.z = 0.;

    while (!finished && run < 10)
    {

        tolerance *= pow(1.5, run);

        cerr << "Tol: " << tolerance << endl;
        run++;

        qsort(data, numVert, sizeof(dinfo), &vec_compare);

        for (i = 0; i < numPoints; i++)
        {
            search.coord.x = x_in[i];
            search.coord.y = y_in[i];
            search.coord.z = z_in[i];

            int index, best_index;
            //cerr << "s " << i << " " << x_in[i]  << " " << y_in[i] << " " << z_in[i] <<endl;
            //Bsearch( 0, numVert, search.coord.x, &index );
            float csearch;
            switch (mode)
            {
            case SORT_X:
                csearch = search.coord.x;
                break;
            case SORT_Y:
                csearch = search.coord.y;
                break;
            case SORT_Z:
                csearch = search.coord.z;
                break;
            }

            if (!Bsearch(0, numVert, csearch, &index))
            {
                best_index = bestFit(&search.coord, index, numVert);
                //cerr << "best " << index << " " << data[index].coord.x  << " " << data[index].coord.y << " " << data[index].coord.z <<endl;

                xdiff[i] = -data[best_index].disp.x;
                ydiff[i] = -data[best_index].disp.y;
                zdiff[i] = -data[best_index].disp.z;

                total_dev[i] = data[best_index].total;

                if (fabs(y_in[i] - data[best_index].coord.y) > 1.0 || fabs(z_in[i] - data[best_index].coord.z) > 1.0)
                {
                    //cerr << "SPoint " << i << " " << x_in[i]  << " " << y_in[i] << " " << z_in[i] <<endl;
                    //cerr << "found  " << data[best_index].coord.x << " " << data[best_index].coord.y << " " << data[best_index].coord.z << " " << endl;

                    //break;
                }
            }
            else
            {
                cerr << "not found Point " << i << " " << x_in[i] << " " << y_in[i] << " " << z_in[i] << endl;
                //xdiff[i] = ydiff[i] = zdiff[i] = total_dev[i] = 0;
                break;
            }
        }
        if (i == numPoints)
        {
            finished = true;
        }
    }

    p_diffOut->setCurrentObject(disp);
    p_totalOut->setCurrentObject(total);

    delete[] data;

    return CONTINUE_PIPELINE;
}

int
SollIstDiff::Bsearch(int begin, int end, float xkey, int *pos)
{
    int mid;
    float diff;

    switch (mode)
    {
    case SORT_X:
        diff = fabs(data[begin].coord.x - xkey);
        break;
    case SORT_Y:
        diff = fabs(data[begin].coord.y - xkey);
        break;
    case SORT_Z:
        diff = fabs(data[begin].coord.z - xkey);
        break;
    }

    //cerr << "looking for " << xkey << " between " << begin << " : " << end << endl;
    if (diff < tolerance)
    {
        *pos = begin;
        return 0;
    }
    else if (diff < tolerance)
    {
        *pos = end;
        return 0;
    }
    else
    {
        if (end - begin <= 1)
        {
            //cerr << data[begin].coord.x-xkey << " " << xkey << " " << data[end].coord.x-xkey << endl;
            *pos = begin;
            //return -1;
            return 0;
        }
        mid = begin + (end - begin) / 2;
        //cerr << mid << " " << data[mid].coord.x << endl;
        float comp;

        switch (mode)
        {
        case SORT_X:
            diff = fabs(data[mid].coord.x - xkey);
            comp = data[mid].coord.x;
            break;
        case SORT_Y:
            diff = fabs(data[mid].coord.y - xkey);
            comp = data[mid].coord.y;
            break;
        case SORT_Z:
            diff = fabs(data[mid].coord.z - xkey);
            comp = data[mid].coord.z;
            break;
        }
        if (diff < tolerance)
        {
            *pos = mid;
            return 0;
        }
        else if (comp > xkey)
        {
            return Bsearch(begin, mid, xkey, pos);
        }
        else
        {
            return Bsearch(mid, end, xkey, pos);
        }
    }
    return 0;
}

int
SollIstDiff::bestFit(Vector *p, int pos, int numPoints)
{
    int i, retpos;
    float min = FLT_MAX;
    float delta;
    Vector vert;

    int range = 1000;

    int min_pos = pos - range;
    if (min_pos < 0)
    {
        min_pos = 0;
    }

    int max_pos = pos + range;
    if (max_pos > numPoints)
    {
        max_pos = numPoints;
    }

    //cerr << "look for " << p->x << " " << p->y << " " << p->z << " " << endl;

    for (i = min_pos; i < max_pos; i++)
    {
        delta = vecdiff(p, &(data[i].coord));
        //cerr << "checked " << data[i].coord.x << " " << data[i].coord.y << " " << data[i].coord.z << ": " << delta << endl;
        if (delta < min)
        {
            min = delta;
            retpos = i;
        }
    }

    return retpos;
}

//
// Destructor
//
SollIstDiff::~SollIstDiff()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char *argv[])

{
    // create the module
    SollIstDiff *application = new SollIstDiff;

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
