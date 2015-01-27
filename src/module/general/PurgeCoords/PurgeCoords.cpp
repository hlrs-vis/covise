/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PurgeCoords.h"
#include <do/coDoData.h>
#include <util/coviseCompat.h>

#include <float.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
PurgeCoords::PurgeCoords(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Remove unused Coordinates")
{
    p_GridIn = addInputPort("GridIn0", "Polygons|UnstructuredGrid", "Geo input");
    p_GridIn->setRequired(1);

    p_ScalDataIn = addInputPort("DataIn0", "Int|Float|Vec2|Vec3|RGBA|Mat3|Tensor", "Scalar input");
    p_ScalDataIn->setRequired(0);

    p_VecDataIn = addInputPort("DataIn1", "Int|Float|Vec2|Vec3|RGBA|Mat3|Tensor", "Vector input");
    p_VecDataIn->setRequired(0);

    p_GridOut = addOutputPort("GridOut0", "Polygons|UnstructuredGrid", "Geo output");

    p_ScalDataOut = addOutputPort("DataOut0", "Int|Float|Vec2|Vec3|RGBA|Mat3|Tensor", "Scalar output");
    p_VecDataOut = addOutputPort("DataOut1", "Int|Float|Vec2|Vec3|RGBA|Mat3|Tensor", "Vector output");
}

int PurgeCoords::compute(const char *)
{

    const coDistributedObject *GeoObj = p_GridIn->getCurrentObject();
    if (!GeoObj)
    {
        sendError("Did not receive object at port '%s'", p_GridIn->getName());
        return FAIL;
    }

    coDoPolygons *polyOut = NULL;
    coDoUnstructuredGrid *gridOut = NULL;
    int *mapping = NULL;
    int n_usedPoints = 0;

    int ncoord = 0, nelem = 0;

    if (const coDoPolygons *polyIn = dynamic_cast<const coDoPolygons *>(GeoObj))
    {
        ncoord = polyIn->getNumPoints();
        int npoly = polyIn->getNumPolygons();
        nelem = npoly;
        int nvert = polyIn->getNumVertices();
        int *vl, *pl;
        float *x, *y, *z;

        polyIn->getAddresses(&x, &y, &z, &vl, &pl);

        int *usedPoints = new int[ncoord];
        memset(usedPoints, 0, ncoord * sizeof(int));

        for (int i = 0; i < nvert; i++)
        {
            usedPoints[vl[i]] = 1;
        }

        float *xout, *yout, *zout;
        int *vlout, *plout;

        n_usedPoints = 0;

        mapping = new int[ncoord];
        memset(mapping, -1, ncoord * sizeof(int));

        for (int i = 0; i < ncoord; i++)
        {
            if (usedPoints[i] == 1)
            {
                mapping[i] = n_usedPoints;
                n_usedPoints++;
            }
        }

        polyOut = new coDoPolygons(p_GridOut->getObjName(), n_usedPoints, nvert, npoly);
        polyOut->getAddresses(&xout, &yout, &zout, &vlout, &plout);

        for (int i = 0; i < ncoord; i++)
        {
            if (mapping[i] != -1)
            {
                xout[mapping[i]] = x[i];
                yout[mapping[i]] = y[i];
                zout[mapping[i]] = z[i];
            }
        }
        for (int i = 0; i < nvert; i++)
        {
            vlout[i] = mapping[vl[i]];
        }
        for (int i = 0; i < npoly; i++)
        {
            plout[i] = pl[i];
        }

        delete[] usedPoints;
    }
    else if (const coDoUnstructuredGrid *usgIn = dynamic_cast<const coDoUnstructuredGrid *>(GeoObj))
    {
        int nconn;
        int *elemlist, *connlist, *typelist;
        float *x, *y, *z;

        usgIn->getAddresses(&elemlist, &connlist, &x, &y, &z);
        usgIn->getGridSize(&nelem, &nconn, &ncoord);
        usgIn->getTypeList(&typelist);

        int *usedPoints = new int[ncoord];
        memset(usedPoints, 0, ncoord * sizeof(int));

        for (int i = 0; i < nconn; i++)
        {
            usedPoints[connlist[i]] = 1;
        }

        int *elemlistout, *connlistout, *typelistout;
        float *xout, *yout, *zout;

        n_usedPoints = 0;

        mapping = new int[ncoord];
        memset(mapping, -1, ncoord * sizeof(int));

        for (int i = 0; i < ncoord; i++)
        {
            if (usedPoints[i] == 1)
            {
                mapping[i] = n_usedPoints;
                n_usedPoints++;
            }
        }

        gridOut = new coDoUnstructuredGrid(p_GridOut->getObjName(), nelem, nconn, n_usedPoints, 1);
        gridOut->getAddresses(&elemlistout, &connlistout, &xout, &yout, &zout);
        gridOut->getTypeList(&typelistout);

        for (int i = 0; i < ncoord; i++)
        {
            if (mapping[i] != -1)
            {
                xout[mapping[i]] = x[i];
                yout[mapping[i]] = y[i];
                zout[mapping[i]] = z[i];
            }
        }
        for (int i = 0; i < nelem; i++)
        {
            elemlistout[i] = elemlist[i];
            typelistout[i] = typelist[i];
        }
        for (int i = 0; i < nconn; i++)
        {
            connlistout[i] = mapping[connlist[i]];
        }

        delete[] usedPoints;
    }
    else
    {
        sendError("Did not receive a compatible object at '%s'", p_GridIn->getName());
        return FAIL;
    }

    const coDoAbstractData *dataObj0 = dynamic_cast<const coDoAbstractData *>(p_ScalDataIn->getCurrentObject());
    const coDoAbstractData *dataObj1 = dynamic_cast<const coDoAbstractData *>(p_VecDataIn->getCurrentObject());

    if (dataObj0)
    {
        int ndata = dataObj0->getNumPoints();
        if ((ndata != ncoord) && (ndata != nelem))
        {
            sendError("data and grid dimensions do not fit");
            return STOP_PIPELINE;
        }

        coDoAbstractData *dataObj0_out = dataObj0->cloneType(p_ScalDataOut->getObjName(), n_usedPoints);
        for (int i = 0; i < ndata; i++)
        {
            if (mapping[i] != -1)
            {
                dataObj0_out->cloneValue(mapping[i], dataObj0, i);
            }
        }

        p_ScalDataOut->setCurrentObject(dataObj0_out);
    }

    if (dataObj1)
    {
        int ndata = dataObj1->getNumPoints();
        if ((ndata != ncoord) && (ndata != nelem))
        {
            sendError("data and grid dimensions do not fit");
            return STOP_PIPELINE;
        }

        coDoAbstractData *dataObj1_out = dataObj1->cloneType(p_VecDataOut->getObjName(), n_usedPoints);
        for (int i = 0; i < ndata; i++)
        {
            if (mapping[i] != -1)
            {
                dataObj1_out->cloneValue(mapping[i], dataObj1, i);
            }
        }

        p_VecDataOut->setCurrentObject(dataObj1_out);
    }

    delete[] mapping;

    if (polyOut)
        p_GridOut->setCurrentObject(polyOut);
    else if (gridOut)
        p_GridOut->setCurrentObject(gridOut);

    return SUCCESS;
}

void PurgeCoords::quit()
{
}

void PurgeCoords::postInst()
{
}

MODULE_MAIN(Filter, PurgeCoords)
